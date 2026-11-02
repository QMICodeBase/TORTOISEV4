#ifndef _GPUSHIM_QUADRATIC_TRANSFORM_IMAGE_H
#define _GPUSHIM_QUADRATIC_TRANSFORM_IMAGE_H
// Backend shim: selects the CUDA, WebGPU or Metal implementation of this interface.
// Each exposes the same declarations, so main/ needs no other change.
#ifdef USEMETAL
    #include "../metal_src/quadratic_transform_image.h"
#elif defined(USEWEBGPU)
    #include "../webgpu_src/quadratic_transform_image.h"
#else
    #include "../cuda_src/quadratic_transform_image.h"
#endif
#endif
