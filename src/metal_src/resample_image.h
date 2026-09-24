#ifndef _MTL_RESAMPLEIMAGE_H
#define _MTL_RESAMPLEIMAGE_H
#include "gpu_image.h"
CUDAIMAGE::Pointer ResampleImage(CUDAIMAGE::Pointer main_image, CUDAIMAGE::Pointer virtual_img);
#endif
