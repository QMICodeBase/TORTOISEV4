#ifndef _WGPU_COMPOSEFIELDS_H
#define _WGPU_COMPOSEFIELDS_H
#include "gpu_image.h"
CUDAIMAGE::Pointer ComposeFields(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer update_field);
#endif
