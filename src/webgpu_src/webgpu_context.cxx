#ifndef _WEBGPU_CONTEXT_CXX
#define _WEBGPU_CONTEXT_CXX

#include "webgpu_context.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <atomic>
#include <mutex>
#include <sstream>
#include <vector>

#include <dawn/native/DawnNative.h>

namespace wgpuctx
{

namespace
{

std::mutex  g_mtx;                      // serializes submission and the pipeline cache
bool        g_init{false};
bool        g_ok{false};
AdapterInfo g_info;
uint32_t g_max_storage_buffers = 0;
std::atomic<uint64_t> g_bytes{0};
std::atomic<uint64_t> g_errors{0};

// Dawn objects are deliberately leaked. As namespace statics they would be
// destroyed at exit in an order Dawn does not tolerate - the device outlives its
// native instance and the process faults after main() returns. Holding them in
// never-freed heap objects removes the teardown entirely; the OS reclaims at
// exit either way.
template <typename T> T &Leaked()
{
    static T *p = new T();
    return *p;
}
#define g_instance  Leaked<wgpu::Instance>()
#define g_device    Leaked<wgpu::Device>()
#define g_queue     Leaked<wgpu::Queue>()
#define g_pipelines Leaked<std::map<std::string, wgpu::ComputePipeline> >()
#define g_native    Leaked<std::unique_ptr<dawn::native::Instance> >()

const char *Env(const char *k) { return std::getenv(k); }

bool EnvOn(const char *k)
{
    const char *v = Env(k);
    return v && *v && std::strcmp(v, "0") != 0;
}

std::string TypeName(wgpu::AdapterType t)
{
    switch(t)
    {
        case wgpu::AdapterType::DiscreteGPU:   return "discrete";
        case wgpu::AdapterType::IntegratedGPU: return "integrated";
        case wgpu::AdapterType::CPU:           return "cpu";
        default:                               return "unknown";
    }
}

std::string BackendName(wgpu::BackendType b)
{
    switch(b)
    {
        case wgpu::BackendType::Vulkan: return "vulkan";
        case wgpu::BackendType::Metal:  return "metal";
        case wgpu::BackendType::D3D12:  return "d3d12";
        case wgpu::BackendType::OpenGL: return "opengl";
        case wgpu::BackendType::OpenGLES: return "opengles";
        default: return "other";
    }
}

// Selection policy (CLAUDE.md 3).
//
// Require a discrete adapter from the configured vendor, on Vulkan. This host
// also enumerates an AMD integrated GPU and llvmpipe; both would run the kernels
// correctly and report meaningless performance, so neither may ever be chosen by
// accident. TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER=1 relaxes the policy for testing
// the rejection path on machines without an NVIDIA card - never for real runs.
// Requested backend, as a lowercase string. Read in two places (the selection
// policy's diagnostics and the adapter enumeration), so it lives in one function
// rather than being parsed twice and drifting.
static std::string BackendPref()
{
    const char *b = Env("TORTOISE_WEBGPU_BACKEND");
    return b ? std::string(b) : std::string("vulkan");
}

bool SelectAdapter(std::vector<dawn::native::Adapter> &adapters, size_t &chosen,
                   std::string &why)
{
    uint32_t want_vendor = 0x10DE;
    if(const char *v = Env("TORTOISE_WEBGPU_VENDOR_ID"))
        want_vendor = (uint32_t)strtoul(v, nullptr, 0);
    const bool allow_any = EnvOn("TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER");

    const std::string bks = BackendPref();

    // TORTOISE_WEBGPU_ADAPTER_TYPE = discrete (default) | integrated | any.
    // benchmark/scripts/gpu_env.sh has exported this since the port began and
    // NOTHING READ IT - it is now honoured. Default is unchanged, so this host
    // still requires a discrete GPU and the benchmark numbers stay comparable.
    // "integrated" exists for Apple Silicon, whose GPU reports as IntegratedGPU;
    // requiring discrete there rejects the only GPU in the machine.
    const char *at = Env("TORTOISE_WEBGPU_ADAPTER_TYPE");
    std::string ats = at ? at : "discrete";
    if(ats != "discrete" && ats != "integrated" && ats != "any")
    {
        std::cerr << "WebGPU: TORTOISE_WEBGPU_ADAPTER_TYPE='" << ats
                  << "' is not one of discrete|integrated|any\n";
        std::exit(1);
    }

    std::ostringstream seen;
    int best = -1;      // a policy-conforming adapter
    int fallback = -1;  // first adapter seen, used only under ALLOW_ANY_ADAPTER
    for(size_t i = 0; i < adapters.size(); i++)
    {
        wgpu::AdapterInfo ai;
        wgpu::Adapter(adapters[i].Get()).GetInfo(&ai);
        seen << "    [" << i << "] vendor=0x" << std::hex << ai.vendorID << std::dec
             << " type=" << TypeName(ai.adapterType)
             << " backend=" << BackendName(ai.backendType)
             << "  " << std::string(ai.device.data, ai.device.length) << "\n";

        const bool vendor_ok  = (ai.vendorID == want_vendor);
        const bool type_ok    = (ats == "any") ||
                                (ats == "discrete"   && ai.adapterType == wgpu::AdapterType::DiscreteGPU) ||
                                (ats == "integrated" && ai.adapterType == wgpu::AdapterType::IntegratedGPU);
        // Whatever backend was actually requested - EnumerateAdapters already
        // filtered to it, so an explicit re-check would reject nothing; "auto"
        // deliberately accepts what Dawn picked for the platform.
        const bool backend_ok = true;

        // A CONFORMING adapter always wins, even under the override. The previous
        // form was
        //     if((vendor_ok && type_ok && backend_ok) || (allow_any && best < 0))
        //         if(best < 0) best = (int)i;
        // which, with allow_any set, latched adapter[0] on the first iteration and
        // never looked further - so on a host whose iGPU enumerates first (this one:
        // AMD at [0], NVIDIA at [1]) the override selected the integrated GPU even
        // though the policy-conforming card was present. That is exactly the outcome
        // the policy exists to prevent.
        if(vendor_ok && type_ok && backend_ok)
        {
            if(best < 0) best = (int)i;     // first conforming adapter
        }
        else if(allow_any && fallback < 0)
        {
            fallback = (int)i;              // remembered, used only if none conform
        }
    }
    if(best < 0 && allow_any && fallback >= 0)
        best = fallback;

    if(best < 0)
    {
        std::ostringstream m;
        m << "no adapter satisfies the selection policy (vendor 0x" << std::hex << want_vendor
          << std::dec << ", " << ats << ", " << bks << "). Adapters seen:\n" << seen.str()
          << "  Refusing to run: an integrated or software adapter would produce\n"
             "  plausible results and meaningless performance. Set\n"
             "  TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER=1 only to test this path.";
        why = m.str();
        return false;
    }
    chosen = (size_t)best;
    return true;
}

} // anonymous namespace

std::string AdapterInfo::Describe() const
{
    std::ostringstream o;
    o << name << " [vendor=0x" << std::hex << vendorID << std::dec
      << " type=" << type << " backend=" << backend;
    if(!driver.empty())
        o << " driver=" << driver;
    o << "]";
    return o.str();
}

bool Init()
{
    std::lock_guard<std::mutex> lk(g_mtx);
    if(g_init)
        return g_ok;
    g_init = true;

    g_native.reset(new dawn::native::Instance());
    g_instance = wgpu::Instance(g_native->Get());

    // Backend defaults to Vulkan - unchanged behaviour on this host, where the GL
    // backends are compiled out of Dawn anyway. It is settable because it was
    // previously HARDCODED, which made the macOS/Metal target - the entire reason
    // this port exists - impossible to reach without editing source.
    //   TORTOISE_WEBGPU_BACKEND = vulkan (default) | metal | d3d12 | opengl | auto
    // "auto" leaves backendType Undefined and lets Dawn choose for the platform.
    wgpu::RequestAdapterOptions opts{};
    const std::string bks = BackendPref();
    if     (bks == "vulkan") opts.backendType = wgpu::BackendType::Vulkan;
    else if(bks == "metal")  opts.backendType = wgpu::BackendType::Metal;
    else if(bks == "d3d12")  opts.backendType = wgpu::BackendType::D3D12;
    else if(bks == "opengl") opts.backendType = wgpu::BackendType::OpenGL;
    else if(bks == "auto")   opts.backendType = wgpu::BackendType::Undefined;
    else
    {
        std::cerr << "WebGPU: TORTOISE_WEBGPU_BACKEND='" << bks
                  << "' is not one of vulkan|metal|d3d12|opengl|auto\n";
        std::exit(1);
    }
    std::vector<dawn::native::Adapter> adapters = g_native->EnumerateAdapters(&opts);

    if(adapters.empty())
    {
        std::cerr << "WebGPU: no " << bks << " adapters found.\n";
        if(EnvOn("TORTOISE_WEBGPU_PROBE_ONLY"))
            return false;
        std::exit(1);
    }

    size_t chosen = 0;
    std::string why;
    if(!SelectAdapter(adapters, chosen, why))
    {
        std::cerr << "WebGPU: " << why << std::endl;
        if(EnvOn("TORTOISE_WEBGPU_PROBE_ONLY"))
            return false;
        std::exit(1);
    }

    wgpu::Adapter adapter(adapters[chosen].Get());
    wgpu::AdapterInfo ai;
    adapter.GetInfo(&ai);
    g_info.name     = std::string(ai.device.data, ai.device.length);
    g_info.backend  = BackendName(ai.backendType);
    g_info.type     = TypeName(ai.adapterType);
    g_info.vendorID = ai.vendorID;
    g_info.deviceID = ai.deviceID;
    g_info.driver   = std::string(ai.description.data, ai.description.length);

    // Negotiate limits rather than assuming them: DRBUDDI peaks near 6 GiB of
    // buffers on `medium`, and the elementwise kernels want a 1024-invocation
    // workgroup budget. Ask for what the adapter actually supports.
    wgpu::Limits supported{};
    adapter.GetLimits(&supported);

    wgpu::Limits want{};
    want.maxComputeInvocationsPerWorkgroup = supported.maxComputeInvocationsPerWorkgroup;
    want.maxComputeWorkgroupSizeX          = supported.maxComputeWorkgroupSizeX;
    want.maxComputeWorkgroupSizeY          = supported.maxComputeWorkgroupSizeY;
    want.maxComputeWorkgroupSizeZ          = supported.maxComputeWorkgroupSizeZ;
    want.maxComputeWorkgroupsPerDimension  = supported.maxComputeWorkgroupsPerDimension;
    want.maxStorageBufferBindingSize       = supported.maxStorageBufferBindingSize;
    want.maxBufferSize                     = supported.maxBufferSize;
    want.maxStorageBuffersPerShaderStage   = supported.maxStorageBuffersPerShaderStage;

    wgpu::DeviceDescriptor ddesc{};
    ddesc.requiredLimits = &want;
#ifdef __APPLE__
    // Required before a shader module may request strict math - see Pipeline().
    static const wgpu::FeatureName kStrictMathFeature =
        wgpu::FeatureName::ShaderModuleCompilationOptions;
    if(adapter.HasFeature(kStrictMathFeature))
    {
        ddesc.requiredFeatures     = &kStrictMathFeature;
        ddesc.requiredFeatureCount = 1;
    }
    else
    {
        // Relaxed math measurably breaks the 1e-5 gate on the guarded ops; running
        // anyway would report a tolerance failure as if it were a porting fault.
        std::cerr << "WebGPU: adapter lacks ShaderModuleCompilationOptions, so strict "
                     "math cannot be requested - aborting" << std::endl;
        exit(1);
    }
#endif
    ddesc.SetUncapturedErrorCallback(
        [](const wgpu::Device &, wgpu::ErrorType type, wgpu::StringView msg) {
            g_errors++;
            std::cerr << "WebGPU error (" << (int)type << "): "
                      << std::string(msg.data, msg.length) << std::endl;
        });
    ddesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device &, wgpu::DeviceLostReason r, wgpu::StringView msg) {
            // Everything computed after device loss is meaningless. cuda_utils.h's
            // gpuErrchk aborts on a device error; match that rather than letting
            // the pipeline continue on a dead device.
            // Destroyed and CallbackCancelled are normal teardown. FailedCreation
            // must fall through to the !g_device check below, which honours
            // TORTOISE_WEBGPU_PROBE_ONLY; aborting here would make that path dead.
            if(r != wgpu::DeviceLostReason::Unknown)
                return;
            std::cerr << "WebGPU device lost (" << (int)r << "): "
                      << std::string(msg.data, msg.length) << " - aborting" << std::endl;
            std::_Exit(1);
        });

    g_device = adapter.CreateDevice(&ddesc);
    if(!g_device)
    {
        std::cerr << "WebGPU: device creation failed on " << g_info.Describe() << std::endl;
        if(EnvOn("TORTOISE_WEBGPU_PROBE_ONLY"))
            return false;
        std::exit(1);
    }
    g_queue = g_device.GetQueue();

    g_max_storage_buffers = supported.maxStorageBuffersPerShaderStage;

    std::cout << "WebGPU adapter: " << g_info.Describe() << std::endl;
    g_ok = true;
    return true;
}

const AdapterInfo &Info() { return g_info; }
wgpu::Device       Device() { Init(); return g_device; }
wgpu::Queue        Queue()  { Init(); return g_queue; }
uint64_t           BytesAllocatedCumulative() { return g_bytes; }
uint64_t           ErrorCount() { return g_errors; }

bool TexFilterEmulation()
{
    return std::getenv("TORTOISE_WEBGPU_CUDA_TEXFILTER") != nullptr;
}

// Checked lazily by the one caller that needs more than the WebGPU default of 8,
// rather than at startup: a run whose metric set never reaches CCJacS is
// unaffected by the limit and must not be refused for it.
void RequireStorageBuffers(uint32_t n, const char *who)
{
    Init();
    if(g_max_storage_buffers < n)
    {
        std::cerr << "WebGPU: " << who << " needs " << n
                  << " storage buffers per shader stage but this adapter allows only "
                  << g_max_storage_buffers << ". Cannot run this metric." << std::endl;
        std::_Exit(1);
    }
}

wgpu::Buffer CreateStorage(size_t bytes, bool copy_src, bool copy_dst)
{
    Init();
    wgpu::BufferDescriptor d{};
    d.size  = (bytes + 3u) & ~3ull;         // WebGPU requires 4-byte sized buffers
    d.usage = wgpu::BufferUsage::Storage;
    if(copy_src) d.usage |= wgpu::BufferUsage::CopySrc;
    if(copy_dst) d.usage |= wgpu::BufferUsage::CopyDst;
    // Under g_mtx like every other device call in this file. Dawn's DeviceBase is
    // internally thread-safe, so this is discipline rather than a fix for a live
    // race - but this was the one place the file's own rule was not followed.
    wgpu::Buffer b;
    { std::lock_guard<std::mutex> lk(g_mtx); b = g_device.CreateBuffer(&d); }
    if(b)
        g_bytes += d.size;
    return b;
}

void Upload(const wgpu::Buffer &dst, const void *src, size_t bytes)
{
    Init();
    const size_t padded = (bytes + 3u) & ~3ull;
    std::lock_guard<std::mutex> lk(g_mtx);
    if(padded == bytes)
    {
        g_queue.WriteBuffer(dst, 0, src, bytes);
        return;
    }
    // Rounding the length up would read past the caller's buffer; stage through a
    // padded host copy instead.
    std::vector<uint8_t> tmp(padded, 0);
    std::memcpy(tmp.data(), src, bytes);
    g_queue.WriteBuffer(dst, 0, tmp.data(), padded);
}

// Device-side clear. This was a host round-trip (allocate a vector of zeros and
// WriteBuffer it), which costs O(bytes) of PCIe traffic on EVERY image
// allocation. ClearBuffer is queue-ordered exactly as WriteBuffer was, so a
// subsequent dispatch sees the cleared buffer without any host sync - the
// observable semantics are unchanged.
void Zero(const wgpu::Buffer &buf, size_t bytes)
{
    Init();
    const uint64_t errors_before = g_errors.load();
    std::lock_guard<std::mutex> lk(g_mtx);
    wgpu::CommandEncoder enc = g_device.CreateCommandEncoder();
    enc.ClearBuffer(buf, 0, (bytes + 3u) & ~3ull);
    wgpu::CommandBuffer cb = enc.Finish();
    g_queue.Submit(1, &cb);
    // Same reasoning as DispatchAt: a rejected clear leaves the buffer at whatever
    // it held, which is indistinguishable from a legitimate result downstream.
    if(g_errors.load() != errors_before)
    {
        std::cerr << "WebGPU: buffer clear failed validation - aborting" << std::endl;
        std::_Exit(1);      // g_mtx is held
    }
}

// Device-to-device copy, queue-ordered like Zero. Replaces a Download+Upload
// pair (a blocking map, a spin, and two full PCIe transits) in image duplication.
void CopyBuffer(const wgpu::Buffer &dst, const wgpu::Buffer &src, size_t bytes)
{
    Init();
    const uint64_t errors_before = g_errors.load();
    std::lock_guard<std::mutex> lk(g_mtx);
    wgpu::CommandEncoder enc = g_device.CreateCommandEncoder();
    enc.CopyBufferToBuffer(src, 0, dst, 0, (bytes + 3u) & ~3ull);
    wgpu::CommandBuffer cb = enc.Finish();
    g_queue.Submit(1, &cb);
    // A rejected copy would leave the destination at its zero-fill, which reads as
    // a legitimate all-zero image.
    if(g_errors.load() != errors_before)
    {
        std::cerr << "WebGPU: device-to-device copy failed validation - aborting" << std::endl;
        std::_Exit(1);      // g_mtx is held
    }
}

wgpu::Buffer CreateUniform(const void *src, size_t bytes)
{
    Init();
    wgpu::BufferDescriptor d{};
    d.size  = (bytes + 15u) & ~15ull;      // uniform buffers align to 16 bytes
    d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer b;   // under g_mtx, as above
    { std::lock_guard<std::mutex> lk(g_mtx); b = g_device.CreateBuffer(&d); }
    if(b)
    {
        g_bytes += d.size;
        std::vector<uint8_t> padded(d.size, 0);
        std::memcpy(padded.data(), src, bytes);
        std::lock_guard<std::mutex> lk(g_mtx);
        g_queue.WriteBuffer(b, 0, padded.data(), d.size);
    }
    return b;
}

void Download(const wgpu::Buffer &src, void *dst, size_t bytes)
{
    Init();
    const size_t n = (bytes + 3u) & ~3ull;

    // Reductions read back a single float, and creating plus destroying a staging
    // buffer for 4 bytes costs far more than the transfer. Cache one staging buffer
    // per small size and reuse it; anything larger keeps the allocate-and-destroy
    // path, where the transfer dominates and caching would just pin memory.
    //
    // Bounded deliberately: the per-op allocate/free pattern is what keeps peak GPU
    // at 0.93x CUDA, so this must not grow into a general buffer pool.
    const size_t kCacheMax = 64u * 1024u;
    const bool   cacheable = (n <= kCacheMax);

    // The whole operation is serialised for a cached buffer: the map and its
    // ProcessEvents spin sit outside the submit lock in the uncached path, which is
    // fine for a private buffer but would race on a shared one.
    std::unique_lock<std::mutex> cache_lk(g_mtx, std::defer_lock);
    wgpu::Buffer staging;
    if(cacheable)
    {
        cache_lk.lock();
        wgpu::Buffer &slot = Leaked<std::map<size_t, wgpu::Buffer> >()[n];
        if(!slot)
        {
            wgpu::BufferDescriptor sd{};
            sd.size  = n;
            sd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
            slot = g_device.CreateBuffer(&sd);
        }
        staging = slot;
    }
    else
    {
        wgpu::BufferDescriptor sd{};
        sd.size  = n;
        sd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        staging = g_device.CreateBuffer(&sd);
    }

    {
        std::unique_lock<std::mutex> lk(g_mtx, std::defer_lock);
        if(!cacheable) lk.lock();          // cached path already holds g_mtx
        wgpu::CommandEncoder enc = g_device.CreateCommandEncoder();
        enc.CopyBufferToBuffer(src, 0, staging, 0, n);
        wgpu::CommandBuffer cmd = enc.Finish();
        g_queue.Submit(1, &cmd);
    }

    // INVARIANT - do not break this. MapAsync waits on the BUFFER's last-usage
    // serial, not on the queue's global serial, and performs no implicit flush. The
    // global sync every caller relies on exists only because this function always
    // submits its OWN CopyBufferToBuffer immediately above: that copy takes the
    // newest serial, and serials complete in order, so waiting on it implies all
    // earlier work finished. If a future change maps a persistent staging buffer
    // without a fresh copy, or skips the copy when data "hasn't changed", that
    // guarantee silently disappears.
    bool done = false;
    bool ok   = false;
    std::string why;
    wgpu::Future f = staging.MapAsync(
        wgpu::MapMode::Read, 0, n, wgpu::CallbackMode::AllowProcessEvents,
        [&done, &ok, &why](wgpu::MapAsyncStatus status, wgpu::StringView msg) {
            ok   = (status == wgpu::MapAsyncStatus::Success);
            why  = std::string(msg.data, msg.length);
            done = true;
        });
    while(!done)
        g_instance.ProcessEvents();
    (void)f;

    // On device loss or a validation failure the range is not mapped;
    // dereferencing it would turn a reportable GPU error into a null-pointer
    // crash, so fail loudly instead.
    const void *src_range = ok ? staging.GetConstMappedRange(0, n) : nullptr;
    if(!src_range)
    {
        // cuda_utils.h's gpuErrchk aborts on a failed transfer; do the same rather
        // than handing back zeros that would look like a legitimate result.
        g_errors++;
        std::cerr << "WebGPU: buffer readback failed (" << why << ") - aborting" << std::endl;
        std::_Exit(1);
    }

    std::memcpy(dst, src_range, bytes);
    staging.Unmap();
    // Cached buffers are reused; only the one-shot ones are destroyed.
    if(!cacheable)
        staging.Destroy();
}

wgpu::ComputePipeline Pipeline(const std::string &key, const char *wgsl,
                               const char *entry_point, uint32_t wgx, uint32_t wgy, uint32_t wgz,
                               const std::vector<std::pair<const char *, double> > &extra)
{
    Init();
    std::ostringstream k;
    k << key << ":" << entry_point << ":" << wgx << "x" << wgy << "x" << wgz;
    for(size_t i = 0; i < extra.size(); i++)
        k << ":" << extra[i].first << "=" << extra[i].second;

    std::lock_guard<std::mutex> lk(g_mtx);
    std::map<std::string, wgpu::ComputePipeline>::iterator it = g_pipelines.find(k.str());
    if(it != g_pipelines.end())
        return it->second;

    wgpu::ShaderSourceWGSL src{};
    src.code = wgsl;
    wgpu::ShaderModuleDescriptor smd{};
    smd.nextInChain = &src;
#ifdef __APPLE__
    // Dawn's Metal backend emits `#pragma METAL fp math_mode(relaxed)` unless strict
    // math is requested (ShaderModuleMTL.mm), which lets the shader compiler reassociate
    // and contract. Measured cost of leaving it relaxed: ComposeFields and InvertField
    // exceed the 1e-5 gate on 6.2% of elements instead of Linux's 0.05%. Apple-only
    // because the Linux/Vulkan configuration is validated and cannot be re-verified here.
    wgpu::ShaderModuleCompilationOptions strict{};
    strict.strictMath = true;
    src.nextInChain = &strict;
#endif
    const uint64_t errors_before_module = g_errors.load();
    wgpu::ShaderModule mod = g_device.CreateShaderModule(&smd);

    // Workgroup dims arrive as WGSL override constants so one shader serves any
    // tiling without string substitution.
    std::vector<wgpu::ConstantEntry> consts(3 + extra.size());
    consts[0].key = "WGX"; consts[0].value = (double)wgx;
    consts[1].key = "WGY"; consts[1].value = (double)wgy;
    consts[2].key = "WGZ"; consts[2].value = (double)wgz;
    for(size_t i = 0; i < extra.size(); i++)
    {
        consts[3 + i].key   = extra[i].first;
        consts[3 + i].value = extra[i].second;
    }

    wgpu::ComputePipelineDescriptor pd{};
    pd.compute.module        = mod;
    pd.compute.entryPoint    = entry_point;
    pd.compute.constants     = consts.data();
    pd.compute.constantCount = consts.size();

    wgpu::ComputePipeline p = g_device.CreateComputePipeline(&pd);

    // Surface a WGSL compile or pipeline-creation failure here, where the shader
    // and entry point are known, rather than as a rejected dispatch later.
    //
    // Verified against this Dawn revision: the uncaptured-error callback is
    // SYNCHRONOUS - Device::HandleError invokes the function pointer directly and
    // never goes through CallbackTaskManager (the only AddCallbackTask sites are
    // work-done and map-async). Encoder-time errors are held until Finish(). So the
    // error-counter delta below is reliable, not best-effort.
    //
    // (An earlier comment here claimed Dawn defers these to a later ProcessEvents
    // and that DispatchAt was the real backstop. Both were wrong, and the second
    // credited a ProcessEvents pump that no longer exists.)
    //
    // The !p test is near-dead: createComputePipeline returns a non-null INVALID
    // object on failure rather than null, so the counter is what actually catches it.
    if(!p || g_errors.load() != errors_before_module)
    {
        std::cerr << "WebGPU: failed to build pipeline '" << k.str()
                  << "' - shader compile or pipeline creation error above." << std::endl;
        std::_Exit(1);
    }

    g_pipelines[k.str()] = p;
    return p;
}

void Dispatch(const wgpu::ComputePipeline &pipeline, const std::vector<wgpu::Buffer> &bindings,
              uint32_t gx, uint32_t gy, uint32_t gz)
{
    std::vector<std::pair<uint32_t, wgpu::Buffer> > at(bindings.size());
    for(size_t i = 0; i < bindings.size(); i++)
        at[i] = std::make_pair((uint32_t)i, bindings[i]);
    DispatchAt(pipeline, at, gx, gy, gz);
}

void DispatchAt(const wgpu::ComputePipeline &pipeline,
                const std::vector<std::pair<uint32_t, wgpu::Buffer> > &bindings,
                uint32_t gx, uint32_t gy, uint32_t gz)
{
    Init();
    const uint64_t errors_before = g_errors.load();
    std::vector<wgpu::BindGroupEntry> entries(bindings.size());
    for(size_t i = 0; i < bindings.size(); i++)
    {
        entries[i].binding = bindings[i].first;
        entries[i].buffer  = bindings[i].second;
        entries[i].offset  = 0;
        entries[i].size    = bindings[i].second.GetSize();
    }

    std::lock_guard<std::mutex> lk(g_mtx);
    wgpu::BindGroupDescriptor bgd{};
    bgd.layout     = pipeline.GetBindGroupLayout(0);
    bgd.entryCount = entries.size();
    bgd.entries    = entries.data();
    wgpu::BindGroup bg = g_device.CreateBindGroup(&bgd);

    wgpu::CommandEncoder enc = g_device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups(gx, gy, gz);
    pass.End();
    wgpu::CommandBuffer cmd = enc.Finish();
    g_queue.Submit(1, &cmd);

    // NO host wait here. The port originally spun on OnSubmittedWorkDone after
    // every dispatch to mirror the CUDA wrappers' cudaDeviceSynchronize(). That
    // mirrored the shape of the reference but not its cost: a CUDA sync on
    // already-complete work is microseconds, while a WebGPU submit plus event-loop
    // round-trip is orders of magnitude more, and DRBUDDI issues many dispatches
    // per iteration.
    //
    // Dropping it is safe because the wait was never load-bearing for
    // correctness: queue operations execute in submission order and Dawn inserts
    // the read-after-write barriers, so a later dispatch always observes an
    // earlier one's writes. The only place ordering must reach the HOST is
    // Download(), which maps the buffer and therefore already waits for all prior
    // queue work.
    //
    // Set TORTOISE_WEBGPU_SYNC_EACH_DISPATCH=1 to restore the old behaviour when
    // bisecting a suspected ordering problem.
    static const bool sync_each = EnvOn("TORTOISE_WEBGPU_SYNC_EACH_DISPATCH");
    if(sync_each)
    {
        bool done = false;
        bool queue_ok = false;
        g_queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowProcessEvents,
            [&done, &queue_ok](wgpu::QueueWorkDoneStatus st, wgpu::StringView) {
                queue_ok = (st == wgpu::QueueWorkDoneStatus::Success);
                done = true;
            });
        while(!done)
            g_instance.ProcessEvents();
        if(!queue_ok)
        {
            std::cerr << "WebGPU: queue work did not complete successfully - aborting" << std::endl;
            std::_Exit(1);      // g_mtx is held here
        }
    }

    // A rejected dispatch leaves its output buffer untouched - in DRBUDDI that is
    // indistinguishable from a legitimately zero update field, so the pipeline
    // would keep running and silently produce wrong results. Fail here instead of
    // relying on every caller to remember to look.
    //
    // Still meaningful without the wait above: WebGPU validation happens during
    // encoding and Submit, not during execution, so a rejected dispatch has
    // already raised its error by this point.
    if(g_errors.load() != errors_before)
    {
        std::cerr << "WebGPU: dispatch failed validation; results would be silently "
                     "wrong, aborting." << std::endl;
        std::_Exit(1);      // g_mtx is held here; std::exit would run its destructor
    }
}

} // namespace wgpuctx

#endif
