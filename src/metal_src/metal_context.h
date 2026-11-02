#ifndef _METAL_CONTEXT_H
#define _METAL_CONTEXT_H

// Process-wide Metal device for the USEMETAL backend (CLAUDE.md 5.1).
//
// Mirrors webgpu_context.h function for function so the ported ops read the same;
// only the namespace and the handle types differ. One device, one command queue.
//
// Handles are shared_ptr<void> over the Objective-C objects so this header stays
// pure C++ (main/ includes it through gpu_image.h) and copies stay reference
// counted, which is what wgpu::Buffer gave the WebGPU backend.

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mtlctx
{

struct AdapterInfo
{
    std::string name;
    std::string backend;       // always "metal"
    std::string type;          // "discrete" | "integrated" | "unknown"
    uint32_t    vendorID{0};
    uint32_t    deviceID{0};
    std::string driver;
    std::string Describe() const;
};

using Buffer          = std::shared_ptr<void>;   // id<MTLBuffer>
using ComputePipeline = std::shared_ptr<void>;   // Pipe (state + threadgroup dims)

// Initialise on first use. Aborts with a diagnostic if no Metal device is usable.
// Returns false only when TORTOISE_METAL_PROBE_ONLY is set, so the no-device path
// stays testable.
bool Init();

const AdapterInfo &Info();

// ---- buffers -------------------------------------------------------------
Buffer CreateStorage(size_t bytes, bool copy_src = true, bool copy_dst = true);
void   Upload(const Buffer &dst, const void *src, size_t bytes);
void   Download(const Buffer &src, void *dst, size_t bytes);   // blocking
// Zero a buffer the caller just created. No queue interaction - see the .mm.
void   ZeroFresh(const Buffer &buf, size_t bytes);
void   CopyBuffer(const Buffer &dst, const Buffer &src, size_t bytes);
Buffer CreateUniform(const void *src, size_t bytes);           // kernel parameter block
// Allocate a buffer already holding `src`. Skips the zero-fill that a full overwrite
// makes pointless and the queue drain that a brand-new buffer cannot need.
Buffer CreateStorageFrom(const void *src, size_t bytes);

// ---- compute -------------------------------------------------------------
// Cached by (key, entry point, workgroup dims, extra constants); `msl` is only
// consulted on a cache miss. Workgroup dims and `extra` become MSL function
// constants, and the dims also set threadsPerThreadgroup - so a shader's
// threadgroup geometry matches its WGSL @workgroup_size exactly, which is what
// keeps reduction summation order comparable (CLAUDE.md 5.1).
ComputePipeline Pipeline(const std::string &key, const char *msl,
                         const char *entry_point,
                         uint32_t wgx, uint32_t wgy, uint32_t wgz,
                         const std::vector<std::pair<const char *, double> > &extra =
                             std::vector<std::pair<const char *, double> >());

// Bind buffers in binding order, dispatch and submit. gx/gy/gz are THREADGROUP
// counts, matching the WebGPU backend's workgroup counts.
void Dispatch(const ComputePipeline &pipeline,
              const std::vector<Buffer> &bindings,
              uint32_t gx, uint32_t gy, uint32_t gz);

// Same, with explicit binding numbers.
void DispatchAt(const ComputePipeline &pipeline,
                const std::vector<std::pair<uint32_t, Buffer> > &bindings,
                uint32_t gx, uint32_t gy, uint32_t gz);

// Bytes ever allocated here. CUMULATIVE - buffers are refcounted and release
// without notifying us, so this never decreases and is not live usage.
uint64_t BytesAllocatedCumulative();

// High-water mark of MTLDevice.currentAllocatedSize. Unlike the counter above this
// IS an allocation measure the driver maintains, and is the backend-lifetime signal
// the loop protocol records (CLAUDE.md 7.4).
uint64_t PeakDeviceBytes();
void     SampleDeviceAllocation();

// Aborts if the device cannot bind n buffers to one compute stage.
void RequireStorageBuffers(uint32_t n, const char *who);

uint64_t ErrorCount();

// Whether this backend is emulating CUDA's quantised texture filter weights
// (CLAUDE.md 5.2). Each backend owns its own switch; the shared replay harness must
// ask rather than guess at an environment variable it does not own.
bool TexFilterEmulation();

// Print device and per-pipeline capacity limits, and check the ones this backend
// actually depends on. maxTotalThreadsPerThreadgroup is a PER-PIPELINE value that falls
// with register pressure, so the device figure alone does not tell you whether the
// 1024-thread reduction will build on a given GPU.
void ReportLimits();

} // namespace mtlctx

#endif
