#ifndef _REDUCTION_SCRATCH_H
#define _REDUCTION_SCRATCH_H

// Reused per-thread scratch for the fixed-size reduction output buffers.
//
// Every reduction wrapper used to cudaMalloc/cudaFree a gSize-element buffer (96 bytes)
// around two kernel launches. nsys measured 70,626 such pairs in a 45 s DRBUDDI window,
// and allocation accounted for ~80% of that window's idle GPU time (PERF_NOTES.md 2.2).
//
// Reuse is bit-exact, not merely "probably fine": every call site launches its first-stage
// kernel with <<<gSize, bSize>>> and each block writes gOut[blockIdx.x] unconditionally,
// so all gSize elements are overwritten before the <<<1, bSize>>> stage reads them. There
// is no path that reads a stale element.
//
// thread_local, because DIFFPREP registration calls these wrappers from several OMP
// threads at once; a shared buffer would race. Each thread is pinned to one device for
// the life of the loop, so the pointer stays valid for its device.

#include <cuda_runtime.h>

// Byte-sized scratch. One SLOT PER CALL SITE, never shared between sites: that makes
// aliasing impossible by construction, so no reasoning about overlapping lifetimes is
// needed. Grows on demand and never shrinks; sizes are stable within a registration.
static constexpr int CUDA_SCRATCH_NSLOTS = 32;

static inline void*&  CudaScratchSlot(int i)
{
    static thread_local void* buf[CUDA_SCRATCH_NSLOTS] = {nullptr};
    return buf[i];
}
static inline size_t& CudaScratchCap(int i)
{
    static thread_local size_t cap[CUDA_SCRATCH_NSLOTS] = {0};
    return cap[i];
}

static inline void* CudaScratch(size_t nbytes, int slot)
{
    if(CudaScratchCap(slot) < nbytes)
    {
        if(CudaScratchSlot(slot))
            cudaFree(CudaScratchSlot(slot));
        cudaMalloc(&CudaScratchSlot(slot), nbytes);
        CudaScratchCap(slot) = nbytes;
    }
    return CudaScratchSlot(slot);
}

// Release this thread's cached blocks. Call at the end of a phase that used large
// scratch, so it does not sit resident through a later, more memory-hungry phase.
// Without this the MI-metric histograms allocated during DIFFPREP registration stayed
// live through DRBUDDI and pushed peak GPU up 10% on `slow` (PERF_NOTES.md 11.1).
static inline void CudaScratchRelease()
{
    for(int i=0;i<32;i++)
    {
        void* b = CudaScratchSlot(i);
        if(b)
        {
            cudaFree(b);
            CudaScratchSlot(i) = nullptr;
            CudaScratchCap(i)  = 0;
        }
    }
}

static inline float* ReductionScratch(int nfloats, int slot = 0)
{
    return (float*)CudaScratch(sizeof(float)*(size_t)nfloats, slot);
}

#endif
