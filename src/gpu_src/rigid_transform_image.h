#ifndef _GPUSHIM_RIGID_TRANSFORM_IMAGE_H
#define _GPUSHIM_RIGID_TRANSFORM_IMAGE_H
// Unreachable from the default DIFFPREP+DRBUDDI path (CLAUDE.md 1.1); CUDA only.
#if !defined(USEWEBGPU) && !defined(USEMETAL)
    #include "../cuda_src/rigid_transform_image.h"
#endif
#endif
