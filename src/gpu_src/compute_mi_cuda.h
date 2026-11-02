#ifndef _GPUSHIM_COMPUTE_MI_CUDA_H
#define _GPUSHIM_COMPUTE_MI_CUDA_H
// Unreachable from the default DIFFPREP+DRBUDDI path (CLAUDE.md 1.1); CUDA only.
#if !defined(USEWEBGPU) && !defined(USEMETAL)
    #include "../cuda_src/compute_mi_cuda.h"
#endif
#endif
