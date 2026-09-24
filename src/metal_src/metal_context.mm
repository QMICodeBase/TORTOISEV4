// Metal implementation of the mtlctx API. See metal_context.h.
//
// Error discipline matches the WebGPU backend and, behind it, cuda_utils.h's
// gpuErrchk: any device or compile failure prints and exit()s rather than
// leaving buffers untouched, which is indistinguishable from a legitimate result.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_context.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <sstream>

namespace mtlctx
{
namespace
{

id<MTLDevice>       g_device = nil;
id<MTLCommandQueue> g_queue  = nil;
AdapterInfo         g_info;
bool                g_ready  = false;
std::mutex          g_mtx;

std::atomic<uint64_t> g_bytes{0};
std::atomic<uint64_t> g_errors{0};
std::atomic<uint64_t> g_peak{0};

// A pipeline plus the threadgroup geometry it was built for. Metal takes the
// threads-per-threadgroup at dispatch time, unlike WGSL's @workgroup_size, so the
// dims travel with the pipeline to keep the call sites identical to the WebGPU ones.
struct Pipe
{
    id<MTLComputePipelineState> state = nil;
    uint32_t wgx = 1, wgy = 1, wgz = 1;
};

std::map<std::string, ComputePipeline> g_pipelines;
std::map<std::string, id<MTLLibrary> > g_libraries;

void Fail(const char *what, NSError *err)
{
    g_errors++;
    std::fprintf(stderr, "Metal: %s%s%s - aborting\n", what,
                 err ? ": " : "",
                 err ? [[err localizedDescription] UTF8String] : "");
    std::_Exit(1);
}

id<MTLBuffer> B(const Buffer &b) { return (__bridge id<MTLBuffer>)b.get(); }


// Bound on command buffers committed but not yet completed.
//
// RESTORED after removal. It was deleted as unjustified complexity because it measured
// as no-op on memory - but that measurement was taken while an unbounded MTLBuffer leak
// dominated the total, so it could not have shown an effect. With the leak fixed the
// bound is real: a committed command buffer RETAINS every resource it references until
// the GPU completes it, so a CPU running far ahead of the GPU raises peak memory with
// nothing to stop it. Autorelease pools do not help - they release *our* reference, not
// the queue's. TORTOISE_METAL_INFLIGHT overrides for A/B.
dispatch_semaphore_t Inflight()
{
    // Function-local static: never nil regardless of call order, unlike a global that
    // only Init() populated.
    static dispatch_semaphore_t sem = []{
        long n = 16;
        if(const char *e = std::getenv("TORTOISE_METAL_INFLIGHT"))
        {
            const long v = std::atol(e);
            if(v > 0) n = v;
        }
        return dispatch_semaphore_create(n);
    }();
    return sem;
}

// Serialises Sync() and guards g_last_cb. Sync is already a drain, so holding this
// across the wait costs nothing and stops a second thread concluding the queue is
// empty while the first is still emptying it.
std::mutex g_sync_mtx;
id<MTLCommandBuffer> g_last_cb = nil;   // retained; the most recently committed buffer

// Commit/completion counters. Waiting on the last command buffer proves every earlier
// one COMPLETED, but not that their completion handlers - where a failure is recorded -
// have RUN. Metal does not document cross-buffer handler delivery order, so Sync()
// waits for the handler count to catch up instead of assuming it. That makes failure
// observation synchronous with the drain without retaining every buffer (which would
// hold their resources and defeat the in-flight bound).
std::atomic<uint64_t> g_committed{0};
std::atomic<uint64_t> g_completed{0};

// Commit, and observe the result. Every command buffer's status is checked here.
void CommitTracked(id<MTLCommandBuffer> cb)
{
    dispatch_semaphore_wait(Inflight(), DISPATCH_TIME_FOREVER);
    // EVERY tracked command buffer's status is checked, not just the sentinel Sync()
    // submits. Without this a failed dispatch was silently followed by a successful
    // sentinel and the host consumed stale data as a result - the one failure mode this
    // backend must never have (CLAUDE.md 0.0: gpuErrchk aborts on any device error).
    // Recording here and aborting in Sync() keeps ONE error path; the wrappers stay clean.
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if([done status] == MTLCommandBufferStatusError)
        {
            g_errors++;
            NSError *e = [done error];
            std::fprintf(stderr, "Metal: command buffer failed: %s\n",
                         e ? [[e localizedDescription] UTF8String] : "(unspecified)");
        }
        g_completed++;
        dispatch_semaphore_signal(Inflight());
    }];
    // The commit and the g_last_cb publish must be ONE critical section. With commit
    // outside the lock, two threads can commit in one order and publish in the other,
    // leaving g_last_cb pointing at an EARLIER buffer than the last committed one - so
    // Sync() waits on a buffer that completes first and the host then memcpys shared
    // storage another thread's kernel is still writing. Silent wrong data, which is the
    // exact failure the shared-storage ordering rule exists to prevent.
    // Uncontended today (only OMP thread 0 reaches the GPU); this is the NGPUs>1 path.
    std::lock_guard<std::mutex> lk(g_sync_mtx);
    [cb commit];
    g_committed++;
    if(g_last_cb) [g_last_cb release];
    g_last_cb = [cb retain];
}

void SyncLocked()   // g_sync_mtx must be held
{
    @autoreleasepool
    {
        // Wait on the LAST COMMITTED buffer, not a fresh sentinel. Command buffers on
        // one queue complete in commit order, so this implies every earlier one has
        // finished AND gives us its status directly - rather than relying on an async
        // completion handler having already run, which Metal does not promise.
        if(g_last_cb)
        {
            [g_last_cb waitUntilCompleted];
            if([g_last_cb status] == MTLCommandBufferStatusError)
                Fail("command buffer failed", [g_last_cb error]);
            [g_last_cb release];
            g_last_cb = nil;
        }
    }
    // Every committed buffer has completed (queue order), so every handler is either
    // done or imminent. Waiting for the counts to meet is what makes a failure in an
    // EARLIER buffer observable here rather than one Sync() later.
    const uint64_t want = g_committed.load();
    while(g_completed.load() < want)
        std::this_thread::yield();
    // Checked on EVERY path, including the one where nothing was queued: a failure
    // recorded by an earlier handler must still stop the run before the host reads.
    if(g_errors.load())
    {
        std::fprintf(stderr, "Metal: %llu command buffer failure(s) - results are "
                             "meaningless, aborting\n",
                     (unsigned long long)g_errors.load());
        // _Exit: this runs under g_sync_mtx, and exit() would run static destructors
        // that can re-enter this backend.
        std::_Exit(1);
    }
}

void Sync()
{
    std::lock_guard<std::mutex> lk(g_sync_mtx);
    SyncLocked();
}

Buffer Wrap(id<MTLBuffer> buf)
{
    // The shared_ptr takes over the +1 that newBufferWithLength: already gave us and
    // releases it on the last reference, so a Buffer copies with the same semantics
    // wgpu::Buffer had.
    //
    // __bridge, NOT __bridge_retained. This file is compiled WITHOUT -fobjc-arc, so
    // `new...` already returns +1 owned; __bridge_retained added a SECOND retain that
    // the single CFRelease below could never balance, and every MTLBuffer ever
    // allocated leaked. Measured: peak RSS 16.9 GiB against Dawn's 4.64 GiB on the same
    // pipeline, and it got worse as the backend got faster - more throughput, more
    // allocations, same leak per allocation. If ARC is ever enabled for this target,
    // this must become __bridge_retained again.
    return Buffer((__bridge void *)buf,
                  [](void *p) { if(p) CFRelease(p); });
}

void Track(size_t bytes)
{
    g_bytes += bytes;
    SampleDeviceAllocation();
}

} // namespace

std::string AdapterInfo::Describe() const
{
    std::ostringstream o;
    o << name << " [vendor=0x" << std::hex << vendorID << std::dec
      << " type=" << type << " backend=" << backend << " driver=" << driver << "]";
    return o.str();
}

bool Init()
{
    std::lock_guard<std::mutex> lk(g_mtx);
    if(g_ready)
        return true;
    // May first run on an OMP worker, which has no pool of its own; -name and
    // -operatingSystemVersionString are autoreleased.
    @autoreleasepool {

    g_device = MTLCreateSystemDefaultDevice();
    if(!g_device)
    {
        if(std::getenv("TORTOISE_METAL_PROBE_ONLY"))
            return false;
        Fail("no Metal device available", nil);
    }

    // Optional pin by name, for a Mac with more than one GPU. Unset on Apple
    // silicon, where there is exactly one device.
    if(const char *want = std::getenv("TORTOISE_METAL_DEVICE_NAME"))
    {
        // Copy rule: +1 owned, so this is released on every exit path below.
        NSArray<id<MTLDevice> > *all = MTLCopyAllDevices();
        id<MTLDevice> match = nil;
        for(id<MTLDevice> d in all)
            if(std::strstr([[d name] UTF8String], want))
                { match = d; break; }
        if(!match)
        {
            std::fprintf(stderr, "Metal: no device matching TORTOISE_METAL_DEVICE_NAME='%s'.\n"
                                 "Available:\n", want);
            for(id<MTLDevice> d in all)
                std::fprintf(stderr, "  %s\n", [[d name] UTF8String]);
            [all release];
            if(std::getenv("TORTOISE_METAL_PROBE_ONLY"))
                return false;
            std::_Exit(1);
        }
        [g_device release];      // the default device this replaces was +1 owned
        g_device = match;
        [match retain];          // outlives the array
        [all release];
    }

    g_queue = [g_device newCommandQueue];
    if(!g_queue)
        Fail("could not create a command queue", nil);

    g_info.name     = [[g_device name] UTF8String];
    g_info.backend  = "metal";
    g_info.type     = [g_device hasUnifiedMemory] ? "integrated" : "discrete";
    g_info.vendorID = 0x106B;   // Apple
    g_info.driver   = [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String];

    g_ready = true;
    std::fprintf(stderr, "Metal device: %s\n", g_info.Describe().c_str());
    }
    return true;
}

const AdapterInfo &Info() { return g_info; }

// ---- buffers -------------------------------------------------------------

Buffer CreateStorage(size_t bytes, bool, bool)
{
    Init();
    if(bytes == 0)
        bytes = 4;
    // Shared storage: on unified memory the CPU and GPU address the same pages, so
    // Upload/Download are memcpy. This does NOT change the explicit upload/download
    // contract the port preserves - M7 is where those calls are removed, if ever.
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if(!buf)
    {
        std::fprintf(stderr, "Metal: failed to allocate %zu bytes\n", bytes);
        std::_Exit(1);
    }
    Track(bytes);
    return Wrap(buf);
}

void ZeroFresh(const Buffer &buf, size_t bytes)
{
    // Only for a buffer the caller just created, where no queued GPU work can reference
    // it - the same argument as CreateStorageFrom below. The precondition is a property
    // of the BUFFER, not of the queue: keying it on global queue state was unsound under
    // concurrent Sync().
    if(bytes)
        std::memset([B(buf) contents], 0, bytes);
}

Buffer CreateStorageFrom(const void *src, size_t bytes)
{
    // For a buffer whose entire contents are about to be written, Allocate()+Upload()
    // does two avoidable things: it zero-fills bytes that are immediately overwritten,
    // and Upload() drains the queue to order the host write against GPU work that
    // cannot reference a buffer this new. This does neither.
    Buffer b = CreateStorage(bytes);
    if(bytes)
        std::memcpy([B(b) contents], src, bytes);
    return b;
}

Buffer CreateUniform(const void *src, size_t bytes)
{
    Buffer b = CreateStorage(bytes);
    // Deliberately NOT Upload(): that drains the queue, and a buffer created on the
    // line above cannot be referenced by any queued GPU work, so there is nothing to
    // order against. Every op builds a parameter block per dispatch, so going through
    // Upload() made each dispatch wait for the whole pipeline to empty - measured at
    // 1.7-2.0x Dawn on the dispatch-heavy ops (InvertField, ComputeJointEntropy,
    // ComposeFields) while single-kernel ops were at parity or faster.
    if(bytes)
        std::memcpy([B(b) contents], src, bytes);
    return b;
}

void Upload(const Buffer &dst, const void *src, size_t bytes)
{
    if(!bytes) return;
    // Drain and copy under ONE lock. Releasing between them would let another thread
    // commit work naming this buffer in the gap, which is the race Sync() exists to
    // prevent. SyncLocked() assumes g_sync_mtx is held.
    std::lock_guard<std::mutex> lk(g_sync_mtx);
    SyncLocked();
    std::memcpy([B(dst) contents], src, bytes);
}

void Download(const Buffer &src, void *dst, size_t bytes)
{
    if(!bytes) return;
    std::lock_guard<std::mutex> lk(g_sync_mtx);
    SyncLocked();
    std::memcpy(dst, [B(src) contents], bytes);
}


void CopyBuffer(const Buffer &dst, const Buffer &src, size_t bytes)
{
    if(!bytes) return;
    @autoreleasepool
    {
        id<MTLCommandBuffer>      cb  = [g_queue commandBuffer];
        if(!cb) Fail("could not create a command buffer", nil);
        id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        [enc copyFromBuffer:B(src) sourceOffset:0 toBuffer:B(dst)
          destinationOffset:0 size:bytes];
        [enc endEncoding];
        CommitTracked(cb);
    }
}

// ---- compute -------------------------------------------------------------

ComputePipeline Pipeline(const std::string &key, const char *msl,
                         const char *entry_point,
                         uint32_t wgx, uint32_t wgy, uint32_t wgz,
                         const std::vector<std::pair<const char *, double> > &extra)
{
    Init();
    std::ostringstream k;
    k << key << ":" << entry_point << ":" << wgx << "x" << wgy << "x" << wgz;
    for(size_t i = 0; i < extra.size(); i++)
        k << ":" << extra[i].first << "=" << extra[i].second;

    std::lock_guard<std::mutex> lk(g_mtx);
    std::map<std::string, ComputePipeline>::iterator it = g_pipelines.find(k.str());
    if(it != g_pipelines.end())
        return it->second;

    @autoreleasepool
    {
        std::map<std::string, id<MTLLibrary> >::iterator li = g_libraries.find(key);
        id<MTLLibrary> lib = (li == g_libraries.end()) ? nil : li->second;
        if(!lib)
        {
            MTLCompileOptions *opts = [MTLCompileOptions new];
            // Metal defaults to fast math, which reassociates and flushes denormals.
            // Measured on the Dawn control: relaxed math alone puts ComposeFields and
            // InvertField over the 1e-5 gate on 6.2% of elements (CLAUDE.md 5.1).
            //
            // BOTH properties are required. The deprecated fastMathEnabled was split in
            // two: mathMode governs reassociation/denormals/IEEE arithmetic, while
            // mathFloatingPointFunctions selects the math LIBRARY and defaults to
            // ...Fast. Setting only mathMode left every sqrt() and log() compiling
            // against the low-precision variants on macOS 15+, while the macOS-14
            // branch below (fastMathEnabled = NO) set both - so the same source produced
            // different numbers depending on host OS version.
            // Measured: 1154 of 4096 sqrt samples misrounded; setting Precise makes 11
            // more fast records bit-exact with CUDA, including ScaleUpdateField, whose
            // magnitude normalises the update field on EVERY DRBUDDI iteration.
            if(@available(macOS 15.0, *))
            {
                opts.mathMode = MTLMathModeSafe;
                opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            }
            else
                opts.fastMathEnabled = NO;
            NSError *err = nil;
            lib = [g_device newLibraryWithSource:[NSString stringWithUTF8String:msl]
                                         options:opts
                                           error:&err];
            if(!lib)
            {
                std::fprintf(stderr, "Metal: failed to compile shader '%s'\n", key.c_str());
                Fail("shader compilation", err);
            }
            // newLibraryWithSource: is already +1 owned and the map holds it for the
            // process lifetime; an extra CFRetain here just leaked it (see Wrap()).
            g_libraries[key] = lib;
            [opts release];      // `new` is +1 owned; the enclosing pool does not cover it
        }

        // Workgroup dims and every `extra` are uint function constants. All of the
        // WGSL overrides they replace are integral (WGX/WGY/WGZ, OP,
        // TEXFILTER_QUANTISE), so one type covers them and no name->type table is
        // needed; a shader wanting a boolean tests `!= 0`.
        // `new` is +1 owned and this file is not ARC, so both fc and fn below are
        // released explicitly. leaks(1) reported them as ROOT LEAKs - bounded (one per
        // pipeline variant, ~33 KB total) but the same ownership mistake as Wrap().
        MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
        uint32_t dims[3] = {wgx, wgy, wgz};
        const char *dim_names[3] = {"WGX", "WGY", "WGZ"};
        for(int i = 0; i < 3; i++)
            [fc setConstantValue:&dims[i] type:MTLDataTypeUInt
                        withName:[NSString stringWithUTF8String:dim_names[i]]];
        std::vector<uint32_t> vals(extra.size());
        for(size_t i = 0; i < extra.size(); i++)
        {
            vals[i] = (uint32_t)extra[i].second;
            [fc setConstantValue:&vals[i] type:MTLDataTypeUInt
                        withName:[NSString stringWithUTF8String:extra[i].first]];
        }

        NSError *err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry_point]
                                       constantValues:fc
                                                error:&err];
        if(!fn)
        {
            std::fprintf(stderr, "Metal: no entry point '%s' in shader '%s'\n",
                         entry_point, key.c_str());
            Fail("function specialisation", err);
        }
        id<MTLComputePipelineState> st = [g_device newComputePipelineStateWithFunction:fn
                                                                                error:&err];
        [fn release];
        [fc release];
        if(!st)
        {
            std::fprintf(stderr, "Metal: pipeline '%s' entry '%s'\n", key.c_str(), entry_point);
            Fail("pipeline creation", err);
        }
        // maxTotalThreadsPerThreadgroup is PER PIPELINE and falls with register
        // pressure, so the device-level 1024 does not guarantee a given kernel fits.
        // TORTOISE_METAL_DEBUG reports the headroom for every pipeline, which is how
        // you check portability to an older GPU without owning one.
        if(std::getenv("TORTOISE_METAL_DEBUG"))
            std::fprintf(stderr, "[METAL] %-28s %-22s want %4u  max %4lu  simd %lu\n",
                         key.c_str(), entry_point, wgx * wgy * wgz,
                         (unsigned long)[st maxTotalThreadsPerThreadgroup],
                         (unsigned long)[st threadExecutionWidth]);
        if(wgx * wgy * wgz > [st maxTotalThreadsPerThreadgroup])
        {
            std::fprintf(stderr, "Metal: pipeline '%s' entry '%s' wants %ux%ux%u threads per "
                                 "threadgroup but the limit is %lu - the WGSL geometry cannot "
                                 "be reproduced, so summation order would differ\n",
                         key.c_str(), entry_point, wgx, wgy, wgz,
                         (unsigned long)[st maxTotalThreadsPerThreadgroup]);
            std::_Exit(1);
        }

        Pipe *p = new Pipe();
        // newComputePipelineStateWithFunction: is +1 owned; Pipe takes that reference
        // and its deleter releases it. No extra retain (see Wrap()).
        p->state = st;
        p->wgx = wgx; p->wgy = wgy; p->wgz = wgz;
        ComputePipeline out(p, [](void *v) {
            Pipe *q = (Pipe *)v;
            if(q->state) CFRelease((__bridge CFTypeRef)q->state);
            delete q;
        });
        g_pipelines[k.str()] = out;
        return out;
    }
}

void DispatchAt(const ComputePipeline &pipeline,
                const std::vector<std::pair<uint32_t, Buffer> > &bindings,
                uint32_t gx, uint32_t gy, uint32_t gz)
{
    if(!gx || !gy || !gz)
        return;
    Pipe *p = (Pipe *)pipeline.get();
    @autoreleasepool
    {
    id<MTLCommandBuffer>        cb  = [g_queue commandBuffer];
    if(!cb) Fail("could not create a command buffer", nil);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p->state];
    for(size_t i = 0; i < bindings.size(); i++)
        [enc setBuffer:B(bindings[i].second) offset:0 atIndex:bindings[i].first];
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
        threadsPerThreadgroup:MTLSizeMake(p->wgx, p->wgy, p->wgz)];
    [enc endEncoding];
    // No wait. Work on one queue completes in submission order and Download
    // synchronises before it reads - the same reasoning that removed the
    // per-dispatch host wait from the WebGPU backend (CLAUDE.md 7.1).
    // The pool drains here: the queue holds its own reference to a committed command
    // buffer until it completes, so releasing ours is correct and does not wait.
    CommitTracked(cb);
    }
}

void Dispatch(const ComputePipeline &pipeline,
              const std::vector<Buffer> &bindings,
              uint32_t gx, uint32_t gy, uint32_t gz)
{
    std::vector<std::pair<uint32_t, Buffer> > at(bindings.size());
    for(size_t i = 0; i < bindings.size(); i++)
        at[i] = std::make_pair((uint32_t)i, bindings[i]);
    DispatchAt(pipeline, at, gx, gy, gz);
}

// ---- accounting ----------------------------------------------------------

uint64_t BytesAllocatedCumulative() { return g_bytes.load(); }

void SampleDeviceAllocation()
{
    if(!g_device) return;
    const uint64_t cur = (uint64_t)[g_device currentAllocatedSize];
    uint64_t prev = g_peak.load();
    while(cur > prev && !g_peak.compare_exchange_weak(prev, cur)) {}
}

uint64_t PeakDeviceBytes() { SampleDeviceAllocation(); return g_peak.load(); }

void RequireStorageBuffers(uint32_t n, const char *who)
{
    // Metal guarantees 31 buffer binding slots per stage on every family this
    // targets; the WebGPU default of 8 was the constraint, not the hardware.
    if(n > 31)
    {
        std::fprintf(stderr, "Metal: %s needs %u buffer bindings, limit is 31\n", who, n);
        std::_Exit(1);
    }
}

uint64_t ErrorCount() { return g_errors.load(); }

bool TexFilterEmulation()
{
    return std::getenv("TORTOISE_METAL_CUDA_TEXFILTER") != nullptr;
}

void ReportLimits()
{
    Init();
    @autoreleasepool
    {
        const struct { const char *n; MTLGPUFamily f; } fam[] = {
            {"Apple7 (M1)", MTLGPUFamilyApple7}, {"Apple8 (M2)", MTLGPUFamilyApple8},
            {"Apple9 (M3/M4)", MTLGPUFamilyApple9},
        };
        std::printf("PROBE: families:");
        for(size_t i = 0; i < sizeof(fam)/sizeof(fam[0]); i++)
            if([g_device supportsFamily:fam[i].f]) std::printf(" %s", fam[i].n);
        std::printf("\n");
        std::printf("PROBE: maxThreadsPerThreadgroup %lux%lux%lu  threadgroupMemory %lu KiB"
                    "  maxBufferLength %.1f GiB\n",
                    (unsigned long)[g_device maxThreadsPerThreadgroup].width,
                    (unsigned long)[g_device maxThreadsPerThreadgroup].height,
                    (unsigned long)[g_device maxThreadsPerThreadgroup].depth,
                    (unsigned long)([g_device maxThreadgroupMemoryLength] / 1024),
                    [g_device maxBufferLength] / 1073741824.0);
    }
}

} // namespace mtlctx
