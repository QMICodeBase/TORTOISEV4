#ifndef _WGPU_WARPIMAGE_H
#define _WGPU_WARPIMAGE_H
#include "gpu_image.h"
CUDAIMAGE::Pointer WarpImage(CUDAIMAGE::Pointer main_image, CUDAIMAGE::Pointer field_image);
#endif
