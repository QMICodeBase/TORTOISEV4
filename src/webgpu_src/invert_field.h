#ifndef _WGPU_INVERTFIELD_H
#define _WGPU_INVERTFIELD_H
#include "gpu_image.h"
CUDAIMAGE::Pointer InvertField(CUDAIMAGE::Pointer field,
                               CUDAIMAGE::Pointer initial_estimate = nullptr);
#endif
