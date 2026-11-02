#ifndef _GPUSHIM_GAUSSIAN_SMOOTH_IMAGE_H
#define _GPUSHIM_GAUSSIAN_SMOOTH_IMAGE_H
// Backend shim: selects the CUDA, WebGPU or Metal implementation of this interface.
// Each exposes the same declarations, so main/ needs no other change.
#ifdef USEMETAL
    #include "../metal_src/gaussian_smooth_image.h"
#elif defined(USEWEBGPU)
    #include "../webgpu_src/gaussian_smooth_image.h"
#else
    #include "../cuda_src/gaussian_smooth_image.h"
#endif
#endif
