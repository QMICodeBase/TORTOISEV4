#ifndef _WEBGPU_CONTEXT_H
#define _WEBGPU_CONTEXT_H

// Process-wide WebGPU device for the USEWEBGPU backend (CLAUDE.md 2.2).
//
// One instance, one adapter, one device, one queue. DIFFPREP hands GPU volumes
// only to the first NGPUs OMP threads (=1), so a single serialized device is
// sufficient; submission is mutex-guarded anyway so a future caller cannot race.
//
// Adapter selection is deliberately strict - see SelectAdapter() in the .cxx.
// This host enumerates an AMD iGPU and llvmpipe alongside the NVIDIA card, and
// either would produce plausible-but-worthless results.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace wgpuctx
{

struct AdapterInfo
{
    std::string name;
    std::string backend;
    std::string type;          // "discrete" | "integrated" | "cpu" | "unknown"
    uint32_t    vendorID{0};
    uint32_t    deviceID{0};
    std::string driver;
    std::string Describe() const;
};

// Initialise on first use. Aborts with a diagnostic if no adapter satisfies the
// selection policy. Returns false only when TORTOISE_WEBGPU_PROBE_ONLY is set,
// so the no-suitable-adapter path stays testable without killing the process.
bool Init();

const AdapterInfo &Info();
wgpu::Device       Device();
wgpu::Queue        Queue();

// ---- buffers -------------------------------------------------------------
wgpu::Buffer CreateStorage(size_t bytes, bool copy_src = true, bool copy_dst = true);
void Upload(const wgpu::Buffer &dst, const void *src, size_t bytes);
void Download(const wgpu::Buffer &src, void *dst, size_t bytes);   // blocking
void Zero(const wgpu::Buffer &buf, size_t bytes);
// Device-to-device copy; both buffers need CopySrc/CopyDst (CreateStorage's default).
void CopyBuffer(const wgpu::Buffer &dst, const wgpu::Buffer &src, size_t bytes);
wgpu::Buffer CreateUniform(const void *src, size_t bytes);   // kernel parameter block

// ---- compute -------------------------------------------------------------
// Pipelines are cached by (key, entry point, workgroup dims). `wgsl` is only
// consulted on a cache miss, so passing a static string is free.
wgpu::ComputePipeline Pipeline(const std::string &key, const char *wgsl,
                               const char *entry_point,
                               uint32_t wgx, uint32_t wgy, uint32_t wgz,
                               const std::vector<std::pair<const char *, double> > &extra =
                                   std::vector<std::pair<const char *, double> >());

// Bind buffers in binding order, dispatch, submit and wait.
void Dispatch(const wgpu::ComputePipeline &pipeline,
              const std::vector<wgpu::Buffer> &bindings,
              uint32_t gx, uint32_t gy, uint32_t gz);

// Same, but with explicit binding numbers - needed where an entry point uses a
// non-contiguous subset of a module's bindings.
void DispatchAt(const wgpu::ComputePipeline &pipeline,
                const std::vector<std::pair<uint32_t, wgpu::Buffer> > &bindings,
                uint32_t gx, uint32_t gy, uint32_t gz);

// Bytes ever allocated through CreateStorage/CreateUniform. This is CUMULATIVE:
// wgpu::Buffer is reference-counted and releases without notifying us, so this
// counter never decreases and must not be read as live usage. Peak device memory
// for the benchmark gate comes from the nvidia-smi sampler instead (CLAUDE.md 7.4).
uint64_t BytesAllocatedCumulative();

// Count of uncaptured WebGPU errors (validation failures, device loss). A
// rejected dispatch silently leaves buffers untouched, which then looks exactly
// like a porting bug - so callers must treat any nonzero count as a hard failure.
// Aborts if the adapter cannot supply n storage buffers in one shader stage.
// Call from the specific op that needs more than the WebGPU default of 8.
void RequireStorageBuffers(uint32_t n, const char *who);

uint64_t ErrorCount();

// Whether this backend is emulating CUDA's quantised texture filter weights
// (CLAUDE.md 5.2). Each backend owns its own switch; the shared replay harness must
// ask rather than guess at an environment variable it does not own.
bool TexFilterEmulation();

} // namespace wgpuctx

#endif
