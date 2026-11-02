#ifndef _GPUSHIM_WARP_IMAGE_H
#define _GPUSHIM_WARP_IMAGE_H
// Backend shim: selects the CUDA, WebGPU or Metal implementation of this interface.
// Each exposes the same declarations, so main/ needs no other change.
#ifdef USEMETAL
    #include "../metal_src/warp_image.h"
#elif defined(USEWEBGPU)
    #include "../webgpu_src/warp_image.h"
#else
    #include "../cuda_src/warp_image.h"
#endif
#endif
