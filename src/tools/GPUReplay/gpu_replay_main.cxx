// gpu_replay - replay a captured GPU golden-vector record in isolation.
//
//   gpu_replay <record_dir> [<record_dir> ...]      replay specific records
//   gpu_replay --all <capture_dir>                  replay every record found
//   gpu_replay --list <capture_dir>                 coverage inventory, no replay
//
// The captured inputs are reloaded onto the device, the operation is re-run
// through an explicit registry, and the result is compared against the captured
// output. Under CUDA the comparison is exact (bitwise): the same kernels on the
// same inputs must reproduce the same bytes. The WebGPU backend will reuse this
// harness with the §6.2 tolerances instead.
//
// Exit codes: 0 all replays matched, 1 a replay mismatched or a record was
// unusable (unknown op, bad schema, truncated blob).

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "gpu_record.h"
#include "../../cuda_src/cuda_image.h"
#include "../../cuda_src/gpu_capture.h"

static_assert(RECORD_SCHEMA_VERSION == gpucap::SCHEMA_VERSION,
              "record schema version drift between capture and replay");
#include "../../cuda_src/warp_image.h"
#include "../../cuda_src/resample_image.h"
#include "../../cuda_src/gaussian_smooth_image.h"
#include "../../cuda_src/cuda_image_utilities.h"
#include "../../cuda_src/compute_metric.h"
#include "../../cuda_src/compute_entropy.h"
#include "../../cuda_src/quadratic_transform_image.h"

#include <fstream>
#include <iostream>
#include <functional>
#include <map>
#include <dirent.h>
#include <sys/stat.h>

// ------------------------------------------------------------ device helpers

static CUDAIMAGE::Pointer ToDevice(const Tensor &t)
{
    CUDAIMAGE::Pointer img = CUDAIMAGE::New();
    img->sz.x = t.dims[0]; img->sz.y = t.dims[1]; img->sz.z = t.dims[2];
    img->spc.x = t.spacing[0]; img->spc.y = t.spacing[1]; img->spc.z = t.spacing[2];
    img->orig.x = t.origin[0]; img->orig.y = t.origin[1]; img->orig.z = t.origin[2];
    CUDAIMAGE::ImageType3D::DirectionType dir;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            dir(r, c) = t.direction[3 * r + c];
    img->dir = dir;
    img->components_per_voxel = t.ncomp;
    img->Allocate();

    cudaExtent ext = make_cudaExtent(t.ncomp * sizeof(float) * img->sz.x, img->sz.y, img->sz.z);
    cudaMemcpy3DParms cp = {0};
    cp.srcPtr = make_cudaPitchedPtr((void *)t.data.data(), t.ncomp * sizeof(float) * img->sz.x,
                                    t.ncomp * img->sz.x, img->sz.y);
    cp.dstPtr = img->getFloatdata();
    cp.kind   = cudaMemcpyHostToDevice;
    cp.extent = ext;
    gpuErrchk(cudaMemcpy3D(&cp));
    return img;
}

// Build an image that carries only geometry (the wrappers that take one never
// read its buffer, they sample onto its grid).
static CUDAIMAGE::Pointer GeomToDevice(const Tensor &t)
{
    CUDAIMAGE::Pointer img = CUDAIMAGE::New();
    img->sz.x = t.dims[0]; img->sz.y = t.dims[1]; img->sz.z = t.dims[2];
    img->spc.x = t.spacing[0]; img->spc.y = t.spacing[1]; img->spc.z = t.spacing[2];
    img->orig.x = t.origin[0]; img->orig.y = t.origin[1]; img->orig.z = t.origin[2];
    CUDAIMAGE::ImageType3D::DirectionType dir;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            dir(r, c) = t.direction[3 * r + c];
    img->dir = dir;
    img->components_per_voxel = t.ncomp;
    img->Allocate();
    return img;
}

static std::vector<float> FromDevice(CUDAIMAGE::Pointer img)
{
    const size_t n = (size_t)img->sz.x * img->sz.y * img->sz.z * img->components_per_voxel;
    std::vector<float> out(n);
    cudaExtent ext = make_cudaExtent(img->components_per_voxel * sizeof(float) * img->sz.x,
                                     img->sz.y, img->sz.z);
    cudaMemcpy3DParms cp = {0};
    cp.srcPtr = img->getFloatdata();
    cp.dstPtr = make_cudaPitchedPtr((void *)out.data(),
                                    img->components_per_voxel * sizeof(float) * img->sz.x,
                                    img->components_per_voxel * img->sz.x, img->sz.y);
    cp.kind   = cudaMemcpyDeviceToHost;
    cp.extent = ext;
    gpuErrchk(cudaMemcpy3D(&cp));
    return out;
}

// ---------------------------------------------------------- operation registry

typedef std::function<void(const Record &, Outcome &)> OpFn;

#define NEED_IN(var, nm)                                                        \
    const Tensor *t_##var = rec.find("in", nm);                                 \
    if(!t_##var) throw std::runtime_error("record is missing input '" nm "'");   \
    CUDAIMAGE::Pointer var = ToDevice(*t_##var);

#define NEED_GEOM(var, nm)                                                      \
    const Tensor *t_##var = rec.find("geom", nm);                               \
    if(!t_##var) throw std::runtime_error("record is missing geometry '" nm "'"); \
    CUDAIMAGE::Pointer var = GeomToDevice(*t_##var);

#define CHECK_OUT(img, nm)                                                       \
    {                                                                            \
        const Tensor *t_o = rec.find("out", nm);                                 \
        if(!t_o) throw std::runtime_error("record is missing output '" nm "'");   \
        out.check(nm, Compare(FromDevice(img), t_o->data));                       \
    }

static float3 F3(const json &j)
{
    std::vector<double> v = j.get<std::vector<double> >();
    float3 r; r.x = v[0]; r.y = v[1]; r.z = v[2];
    return r;
}

// Allocate a zeroed image matching a captured tensor's geometry.
static CUDAIMAGE::Pointer AllocLike(const Tensor &t)
{
    CUDAIMAGE::Pointer img = CUDAIMAGE::New();
    img->sz.x = t.dims[0]; img->sz.y = t.dims[1]; img->sz.z = t.dims[2];
    img->spc.x = t.spacing[0]; img->spc.y = t.spacing[1]; img->spc.z = t.spacing[2];
    img->orig.x = t.origin[0]; img->orig.y = t.origin[1]; img->orig.z = t.origin[2];
    CUDAIMAGE::ImageType3D::DirectionType dir;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            dir(r, c) = t.direction[3 * r + c];
    img->dir = dir;
    img->components_per_voxel = t.ncomp;
    img->Allocate();
    return img;
}

static std::vector<float> Taps(const Record &rec)
{
    std::vector<double> k = rec.params["kernel"].get<std::vector<double> >();
    return std::vector<float>(k.begin(), k.end());
}

#define DIR9(im) im->dir(0,0),im->dir(0,1),im->dir(0,2),                        \
                 im->dir(1,0),im->dir(1,1),im->dir(1,2),                        \
                 im->dir(2,0),im->dir(2,1),im->dir(2,2)

#define NEED_OUTBUF(var, nm)                                                     \
    const Tensor *tb_##var = rec.find("out", nm);                                \
    if(!tb_##var) throw std::runtime_error("record is missing output '" nm "'");  \
    CUDAIMAGE::Pointer var = AllocLike(*tb_##var);

static std::map<std::string, OpFn> BuildRegistry()
{
    std::map<std::string, OpFn> R;

    R["WarpImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_image, "main_image"); NEED_IN(field_image, "field_image");
        main_image->CreateTexture();
        CUDAIMAGE::Pointer o = WarpImage(main_image, field_image);
        CHECK_OUT(o, "output");
    };
    R["ResampleImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_field, "main_field"); NEED_GEOM(virtual_img, "virtual_img");
        CUDAIMAGE::Pointer o = ResampleImage(main_field, virtual_img);
        CHECK_OUT(o, "output");
    };
    R["GaussianSmoothImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_image, "main_image");
        CUDAIMAGE::Pointer o = GaussianSmoothImage(main_image, (float)rec.params["std"]);
        CHECK_OUT(o, "output");
    };
    R["PreprocessImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(img, "img");
        CUDAIMAGE::Pointer o = PreprocessImage(img, (float)rec.params["low_val"],
                                                    (float)rec.params["up_val"]);
        CHECK_OUT(o, "output");
    };
    R["ComposeFields"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_field, "main_field"); NEED_IN(update_field, "update_field");
        CUDAIMAGE::Pointer o = ComposeFields(main_field, update_field);
        CHECK_OUT(o, "output");
    };
    R["NegateField"] = [](const Record &rec, Outcome &out) {
        NEED_IN(field, "field");
        CUDAIMAGE::Pointer o = NegateField(field);
        CHECK_OUT(o, "output");
    };
    R["AddImages"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1"); NEED_IN(im2, "im2");
        CUDAIMAGE::Pointer o = AddImages(im1, im2);
        CHECK_OUT(o, "output");
    };
    R["MultiplyImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1");
        CUDAIMAGE::Pointer o = MultiplyImage(im1, (float)rec.params["factor"]);
        CHECK_OUT(o, "output");
    };
    R["MultiplyImages"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1"); NEED_IN(im2, "im2");
        CUDAIMAGE::Pointer o = MultiplyImages(im1, im2);
        CHECK_OUT(o, "output");
    };
    R["InvertField"] = [](const Record &rec, Outcome &out) {
        NEED_IN(field, "field");
        CUDAIMAGE::Pointer init = nullptr;
        const Tensor *ti = rec.find("in", "initial_estimate");
        if(ti) init = ToDevice(*ti);
        CUDAIMAGE::Pointer o = InvertField(field, init);
        CHECK_OUT(o, "output");
    };
    R["ComputeImageGradientImg"] = [](const Record &rec, Outcome &out) {
        NEED_IN(img, "img");
        std::vector<CUDAIMAGE::Pointer> g = ComputeImageGradientImg(img);
        CHECK_OUT(g[0], "grad_x"); CHECK_OUT(g[1], "grad_y"); CHECK_OUT(g[2], "grad_z");
    };
    R["ScaleUpdateField"] = [](const Record &rec, Outcome &out) {
        NEED_IN(field, "field");
        ScaleUpdateField(field, (float)rec.params["scale_factor"]);
        CHECK_OUT(field, "field");
    };
    R["RestrictPhase"] = [](const Record &rec, Outcome &out) {
        NEED_IN(field, "field");
        RestrictPhase(field, F3(rec.params["phase"]));
        CHECK_OUT(field, "field");
    };
    R["ContrainDefFields"] = [](const Record &rec, Outcome &out) {
        NEED_IN(ufield, "ufield"); NEED_IN(dfield, "dfield");
        ContrainDefFields(ufield, dfield);
        CHECK_OUT(ufield, "ufield"); CHECK_OUT(dfield, "dfield");
    };
    R["AddToUpdateField"] = [](const Record &rec, Outcome &out) {
        NEED_IN(updateField, "updateField"); NEED_IN(updateField_temp, "updateField_temp");
        AddToUpdateField(updateField, updateField_temp, (float)rec.params["weight"],
                         rec.params["normalize"] != 0);
        CHECK_OUT(updateField, "updateField");
    };
    R["SumImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1");
        out.checkScalar("sum", SumImage(im1), rec.scalars["sum"]);
    };
    R["ComputeEntropy"] = [](const Record &rec, Outcome &out) {
        NEED_IN(img, "img");
        const float v = ComputeEntropy(img, (int)rec.params["Nbins"],
                                       (float)rec.params["low_lim"], (float)rec.params["high_lim"]);
        out.checkScalar("entropy", v, rec.scalars["entropy"]);
    };
    R["ComputeJointEntropy"] = [](const Record &rec, Outcome &out) {
        NEED_IN(img1, "img1"); NEED_IN(img2, "img2");
        std::vector<double> L = rec.params["lims"].get<std::vector<double> >();
        float ej, e1, e2;
        ComputeJointEntropy(img1, L[0], L[1], img2, L[2], L[3],
                            (int)rec.params["Nbins"], ej, e1, e2);
        out.checkScalar("entropy_joint", ej, rec.scalars["entropy_joint"]);
        out.checkScalar("entropy_img1",  e1, rec.scalars["entropy_img1"]);
        out.checkScalar("entropy_img2",  e2, rec.scalars["entropy_img2"]);
    };
    R["ComputeMetric_CC"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img");
        CUDAIMAGE::Pointer uF, uM;
        const float v = ComputeMetric_CC(up_img, down_img, uF, uM);
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };
    R["ComputeMetric_CCSK"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img"); NEED_IN(str_img, "str_img");
        CUDAIMAGE::Pointer uF, uM;
        const float v = ComputeMetric_CCSK(up_img, down_img, str_img, uF, uM,
                                           (float)rec.params["t"]);
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };
    R["ComputeMetric_MSJac"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img");
        NEED_IN(def_FINV, "def_FINV"); NEED_IN(def_MINV, "def_MINV");
        NEED_OUTBUF(uF, "updateFieldF"); NEED_OUTBUF(uM, "updateFieldM");
        std::vector<float> k = Taps(rec);
        float v = 0;
        ComputeMetric_MSJac_cuda(up_img->getFloatdata(), down_img->getFloatdata(),
                                 up_img->sz, up_img->spc, DIR9(up_img),
                                 def_FINV->getFloatdata(), def_MINV->getFloatdata(),
                                 uF->getFloatdata(), uM->getFloatdata(),
                                 F3(rec.params["phase_vector"]), (int)k.size(), k.data(), v);
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };
    R["ComputeMetric_CCJacS"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img"); NEED_IN(str_img, "str_img");
        NEED_IN(def_FINV, "def_FINV"); NEED_IN(def_MINV, "def_MINV");
        NEED_OUTBUF(uF, "updateFieldF"); NEED_OUTBUF(uM, "updateFieldM");
        std::vector<float> k = Taps(rec);
        float v = 0;
        ComputeMetric_CCJacS_cuda(up_img->getFloatdata(), down_img->getFloatdata(),
                                  str_img->getFloatdata(),
                                  up_img->sz, up_img->spc, DIR9(up_img),
                                  def_FINV->getFloatdata(), def_MINV->getFloatdata(),
                                  uF->getFloatdata(), uM->getFloatdata(),
                                  F3(rec.params["phase_vector"]), (int)k.size(), k.data(), v);
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };
    R["QuadraticTransformImageC"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_image, "main_image"); NEED_GEOM(target_img, "target_img");
        std::vector<double> pr = rec.params["params"].get<std::vector<double> >();
        TransformType::Pointer tp = TransformType::New();
        tp->SetPhase((short)(double)rec.params["phase"]);
        TransformType::ParametersType par(TransformType::NQUADPARAMS);
        for(int i = 0; i < TransformType::NQUADPARAMS; i++)
            par[i] = pr[i];
        tp->SetParameters(par);

        // The kernel consumes the transform's matrix, so verify the rebuilt
        // transform reproduces the captured one rather than assuming it does.
        std::vector<double> mref = rec.params["matrix"].get<std::vector<double> >();
        TransformType::MatrixType m = tp->GetMatrix();
        for(int r = 0; r < 3; r++)
            for(int c = 0; c < 3; c++)
                if((float)m(r, c) != (float)mref[3 * r + c])
                    throw std::runtime_error("rebuilt transform matrix differs from the captured one");
        main_image->CreateTexture();
        CUDAIMAGE::Pointer o = QuadraticTransformImageC(main_image, tp, target_img);
        CHECK_OUT(o, "output");
    };
    return R;
}

// ------------------------------------------------------------------- driver

static std::vector<std::string> ListRecordDirs(const std::string &root)
{
    std::vector<std::string> dirs;
    DIR *d = opendir(root.c_str());
    if(!d) return dirs;
    struct dirent *e;
    while((e = readdir(d)))
    {
        if(e->d_name[0] == '.') continue;
        struct stat st;
        const std::string p = root + "/" + e->d_name;
        if(stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        {
            struct stat js;
            if(stat((p + "/record.json").c_str(), &js) == 0)
                dirs.push_back(p);
        }
    }
    closedir(d);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

// CUDA-side counterpart of webgpu_replay's --bench. Same measurement, same
// included host-side Compare() term, so the two are directly comparable and the
// WebGPU/CUDA ratio is conservative. See webgpu_replay_main.cxx for the caveat.
static int RunBench(const std::vector<std::string> &dirs,
                    std::map<std::string, OpFn> &registry, int reps)
{
    std::cout << "per-op timing over " << reps << " reps (best of), including the\n"
                 "harness's host-side comparison.\n\n";
    struct Row { double best_ms; int calls; };
    std::map<std::string, Row> rows;
    for(size_t i = 0; i < dirs.size(); i++)
    {
        Record rec;
        try { rec = LoadRecord(dirs[i]); } catch(std::exception &) { continue; }
        std::map<std::string, OpFn>::iterator it = registry.find(rec.op);
        if(it == registry.end()) continue;
        try { Outcome warm; it->second(rec, warm); } catch(std::exception &) { continue; }

        double best = 1e300;
        bool   died = false;
        for(int r = 0; r < reps; r++)
        {
            Outcome o;
            const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            try { it->second(rec, o); } catch(std::exception &) { died = true; break; }
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            if(ms < best) best = ms;
        }
        if(died) continue;
        if(!rows.count(rec.op)) { Row z; z.best_ms = 0; z.calls = 0; rows[rec.op] = z; }
        rows[rec.op].best_ms += best;
        rows[rec.op].calls++;
    }
    double grand = 0.0;
    for(std::map<std::string, Row>::iterator it = rows.begin(); it != rows.end(); ++it)
        grand += it->second.best_ms;
    printf("  %-28s %10s %8s\n", "op", "best_ms", "records");
    for(std::map<std::string, Row>::iterator it = rows.begin(); it != rows.end(); ++it)
        printf("  %-28s %10.2f %8d\n", it->first.c_str(), it->second.best_ms, it->second.calls);
    printf("  %-28s %10.2f\n", "TOTAL", grand);
    return 0;
}

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        std::cerr << "usage: gpu_replay <record_dir>...\n"
                  << "       gpu_replay --all   <capture_dir>\n"
                  << "       gpu_replay --list  <capture_dir>\n"
                  << "       gpu_replay --bench <capture_dir> [reps]\n";
        return 1;
    }

    // This tool links gpu_capture.cxx and calls the HOOKED wrappers, so with
    // TORTOISE_GPU_CAPTURE set it would REWRITE the very golden vectors it is
    // replaying - and the quota exactly matches the record count, so the overwrite
    // is complete and silent, after which "44/44 bit-exact" means nothing. Refuse.
    if(std::getenv("TORTOISE_GPU_CAPTURE"))
    {
        std::cerr << "gpu_replay: TORTOISE_GPU_CAPTURE is set. This tool links the capture\n"
                     "hooks, so replaying would overwrite the golden vectors with this\n"
                     "binary's own output and the comparison would be vacuous. Unset it.\n";
        return 2;
    }

    std::map<std::string, OpFn> registry = BuildRegistry();

    const std::string mode = argv[1];
    std::vector<std::string> dirs;
    if(mode == "--all" || mode == "--list" || mode == "--bench")
    {
        if(argc < 3) { std::cerr << "need a capture dir\n"; return 1; }
        dirs = ListRecordDirs(argv[2]);
        if(dirs.empty()) { std::cerr << "no records under " << argv[2] << "\n"; return 1; }
    }
    else
        for(int i = 1; i < argc; i++) dirs.push_back(argv[i]);

    if(mode == "--bench")
        return RunBench(dirs, registry, argc > 3 ? atoi(argv[3]) : 5);

    if(mode == "--list")
    {
        std::map<std::string, int> per_op;
        for(size_t i = 0; i < dirs.size(); i++)
        {
            try { per_op[LoadRecord(dirs[i]).op]++; }
            catch(std::exception &e) { std::cout << "  BAD " << dirs[i] << ": " << e.what() << "\n"; }
        }
        std::cout << "captured operations (" << per_op.size() << "):\n";
        for(std::map<std::string, int>::iterator it = per_op.begin(); it != per_op.end(); ++it)
            std::cout << "  " << (registry.count(it->first) ? "[replayable] " : "[NO REPLAY ] ")
                      << it->first << "  x" << it->second << "\n";
        for(std::map<std::string, OpFn>::iterator it = registry.begin(); it != registry.end(); ++it)
            if(!per_op.count(it->first))
                std::cout << "  [NOT CAPTURED] " << it->first << "\n";
        return 0;
    }

    int pass = 0, fail = 0, bad = 0;
    for(size_t i = 0; i < dirs.size(); i++)
    {
        Record rec;
        try { rec = LoadRecord(dirs[i]); }
        catch(std::exception &e)
        {
            std::cout << "BAD    " << dirs[i] << ": " << e.what() << "\n";
            bad++;
            continue;
        }

        std::map<std::string, OpFn>::iterator it = registry.find(rec.op);
        if(it == registry.end())
        {
            std::cout << "BAD    " << rec.op << " [" << dirs[i] << "]: no replay registered for this op\n";
            bad++;
            continue;
        }

        Outcome out;
        try { it->second(rec, out); }
        catch(std::exception &e)
        {
            std::cout << "BAD    " << rec.op << " [" << dirs[i] << "]: " << e.what() << "\n";
            bad++;
            continue;
        }

        if(out.ok) { std::cout << "PASS   " << rec.op << "." << rec.seq << "\n"; pass++; }
        else       { std::cout << "FAIL   " << rec.op << "." << rec.seq << ": " << out.detail << "\n"; fail++; }
    }

    std::cout << "\n" << pass << " passed, " << fail << " failed, " << bad << " unusable\n";
    return (fail || bad) ? 1 : 0;
}
