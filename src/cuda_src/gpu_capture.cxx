#ifndef _GPU_CAPTURE_CXX
#define _GPU_CAPTURE_CXX

#include "gpu_capture.h"
#include "defines.h"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sys/stat.h>

namespace gpucap
{

namespace
{
std::mutex g_mtx;
std::map<std::string, int> g_seq;      // records written per op

const char *CaptureDir()
{
    static const char *d = std::getenv("TORTOISE_GPU_CAPTURE");
    return d;
}

int MaxPerOp()
{
    static int n = []{
        const char *e = std::getenv("TORTOISE_GPU_CAPTURE_N");
        return e ? std::atoi(e) : 2;
    }();
    return n;
}

// Dense host copy of a CUDAIMAGE, from either its pitched buffer or, once
// CreateTexture() has freed that, from the backing cudaArray.
bool Readback(const CUDAIMAGE::Pointer img, std::vector<float> &out)
{
    if(!img)
        return false;

    const size_t nvox = (size_t)img->sz.x * img->sz.y * img->sz.z;
    const size_t n    = nvox * img->components_per_voxel;
    if(n == 0)
        return false;
    out.resize(n);

    cudaPitchedPtr p = img->getFloatdata();
    if(p.ptr)
    {
        cudaExtent ext = make_cudaExtent(img->components_per_voxel * sizeof(float) * img->sz.x,
                                         img->sz.y, img->sz.z);
        cudaMemcpy3DParms cp = {0};
        cp.srcPtr = p;
        cp.dstPtr = make_cudaPitchedPtr((void *)out.data(),
                                        img->components_per_voxel * sizeof(float) * img->sz.x,
                                        img->components_per_voxel * img->sz.x, img->sz.y);
        cp.kind   = cudaMemcpyDeviceToHost;
        cp.extent = ext;
        return cudaMemcpy3D(&cp) == cudaSuccess;
    }

    cudaArray *arr = img->GetArray();
    if(!arr)
        return false;
    cudaMemcpy3DParms cp = {0};
    cp.srcArray = arr;
    cp.dstPtr   = make_cudaPitchedPtr((void *)out.data(), sizeof(float) * img->sz.x,
                                      img->sz.x, img->sz.y);
    cp.kind     = cudaMemcpyDeviceToHost;
    cp.extent   = make_cudaExtent(img->sz.x, img->sz.y, img->sz.z);   // array extent: elements
    return cudaMemcpy3D(&cp) == cudaSuccess;
}

// FNV-1a over the blob. Not cryptographic - this only has to detect a truncated
// or corrupted record, which it does.
std::string Digest(const std::vector<float> &v)
{
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)v.data();
    const size_t n = v.size() * sizeof(float);
    for(size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016lx", (unsigned long)h);
    return std::string(buf);
}
} // anonymous namespace

bool Enabled() { return CaptureDir() != nullptr; }

struct Rec::Impl
{
    std::string op;
    std::string dir;
    int seq{0};
    json rec;
    std::vector<std::pair<std::string, std::vector<float> > > blobs;   // filename -> data
};

Rec::Rec(const std::string &op, int variant)
{
    if(!Enabled())
        return;

    const std::string key = (variant < 0) ? op : op + "#v" + std::to_string(variant);
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        int &s = g_seq[key];
        if(s >= MaxPerOp())
            return;                       // enough of this op already
        d = std::make_shared<Impl>();
        d->seq = s++;
    }

    active  = true;
    d->op   = op;
    d->dir  = std::string(CaptureDir()) + "/" + op +
              (variant < 0 ? "" : ".v" + std::to_string(variant)) + "." + std::to_string(d->seq);
    d->rec["schema_version"] = SCHEMA_VERSION;
    d->rec["op"]             = op;
    d->rec["seq"]            = d->seq;
    d->rec["params"]         = json::object();
    d->rec["scalars"]        = json::object();
    d->rec["tensors"]        = json::array();
}

Rec::~Rec() {}

Rec &Rec::param(const std::string &k, double v)
{ if(active) d->rec["params"][k] = v; return *this; }

Rec &Rec::param(const std::string &k, const std::string &v)
{ if(active) d->rec["params"][k] = v; return *this; }

Rec &Rec::param(const std::string &k, const std::vector<double> &v)
{ if(active) d->rec["params"][k] = v; return *this; }

Rec &Rec::param(const std::string &k, float3 v)
{ if(active) d->rec["params"][k] = std::vector<double>{v.x, v.y, v.z}; return *this; }

Rec &Rec::scalar(const std::string &k, double v)
{ if(active) d->rec["scalars"][k] = v; return *this; }

Rec &Rec::in(const std::string &name, const CUDAIMAGE::Pointer img)
{ return addTensor("in", name, img); }

Rec &Rec::out(const std::string &name, const CUDAIMAGE::Pointer img)
{ return addTensor("out", name, img); }

Rec &Rec::geom(const std::string &name, const CUDAIMAGE::Pointer img)
{
    if(!active || !img)
        return *this;
    json t;
    t["role"]    = "geom";
    t["name"]    = name;
    t["dims"]    = std::vector<int>{img->sz.x, img->sz.y, img->sz.z};
    t["ncomp"]   = img->components_per_voxel;
    t["spacing"] = std::vector<double>{img->spc.x, img->spc.y, img->spc.z};
    t["origin"]  = std::vector<double>{img->orig.x, img->orig.y, img->orig.z};
    std::vector<double> dirv(9);
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            dirv[3 * r + c] = img->dir(r, c);
    t["direction"] = dirv;
    d->rec["tensors"].push_back(t);
    return *this;
}

Rec &Rec::addTensor(const char *role, const std::string &name,
                    const CUDAIMAGE::Pointer img)
{
    if(!active)
        return *this;

    std::vector<float> data;
    if(!Readback(img, data))
    {
        // Record the absence explicitly rather than writing a silently empty
        // tensor: a replay must fail loudly on an incomplete record.
        json t;
        t["role"] = role; t["name"] = name; t["unreadable"] = true;
        d->rec["tensors"].push_back(t);
        return *this;
    }

    const std::string file = std::string(role) + "_" + name + ".f32";
    json t;
    t["role"]      = role;
    t["name"]      = name;
    t["dims"]      = std::vector<int>{img->sz.x, img->sz.y, img->sz.z};
    t["ncomp"]     = img->components_per_voxel;
    t["spacing"]   = std::vector<double>{img->spc.x, img->spc.y, img->spc.z};
    t["origin"]    = std::vector<double>{img->orig.x, img->orig.y, img->orig.z};
    std::vector<double> dirv(9);
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            dirv[3 * r + c] = img->dir(r, c);
    t["direction"] = dirv;
    t["file"]      = file;
    t["bytes"]     = (double)(data.size() * sizeof(float));
    t["digest"]    = Digest(data);
    d->rec["tensors"].push_back(t);
    d->blobs.push_back(std::make_pair(file, std::move(data)));
    return *this;
}

void Rec::save()
{
    if(!active)
        return;

    ::mkdir(CaptureDir(), 0755);
    ::mkdir(d->dir.c_str(), 0755);

    for(size_t i = 0; i < d->blobs.size(); i++)
    {
        std::ofstream f((d->dir + "/" + d->blobs[i].first).c_str(), std::ios::binary);
        f.write((const char *)d->blobs[i].second.data(),
                d->blobs[i].second.size() * sizeof(float));
    }
    std::ofstream f((d->dir + "/record.json").c_str());
    f << d->rec.dump(2) << std::endl;

    // Append to a manifest so coverage can be inventoried without a directory walk.
    std::lock_guard<std::mutex> lk(g_mtx);
    std::ofstream m((std::string(CaptureDir()) + "/manifest.jsonl").c_str(), std::ios::app);
    // Use the real directory name: variant records are op.v<N>.<seq>, so emitting
    // op.<seq> here would make them unlocatable from the manifest.
    const std::string leaf = d->dir.substr(d->dir.find_last_of('/') + 1);
    m << "{\"op\":\"" << d->op << "\",\"seq\":" << d->seq
      << ",\"dir\":\"" << leaf << "\"}" << std::endl;

    active = false;      // saved once
}

} // namespace gpucap

#endif
