#ifndef _GPUSHIM_CUDA_IMAGE_UTILITIES_H
#define _GPUSHIM_CUDA_IMAGE_UTILITIES_H
// Backend shim: selects the CUDA, WebGPU or Metal implementation of this interface.
// Each exposes the same declarations, so main/ needs no other change.
#ifdef USEMETAL
    #include "../metal_src/image_utilities_all.h"
#elif defined(USEWEBGPU)
    #include "../webgpu_src/image_utilities_all.h"
#else
    #include "../cuda_src/cuda_image_utilities.h"
#endif
#endif
