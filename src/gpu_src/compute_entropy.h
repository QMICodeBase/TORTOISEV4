#ifndef _GPUSHIM_COMPUTE_ENTROPY_H
#define _GPUSHIM_COMPUTE_ENTROPY_H
// Backend shim: selects the CUDA, WebGPU or Metal implementation of this interface.
// Each exposes the same declarations, so main/ needs no other change.
#ifdef USEMETAL
    #include "../metal_src/compute_entropy.h"
#elif defined(USEWEBGPU)
    #include "../webgpu_src/compute_entropy.h"
#else
    #include "../cuda_src/compute_entropy.h"
#endif
#endif
