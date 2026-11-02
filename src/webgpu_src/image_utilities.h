#ifndef _WGPU_IMAGE_UTILITIES_H
#define _WGPU_IMAGE_UTILITIES_H
#include "gpu_image.h"

CUDAIMAGE::Pointer AddImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2);
CUDAIMAGE::Pointer MultiplyImage(CUDAIMAGE::Pointer im1, float factor);
CUDAIMAGE::Pointer MultiplyImages(CUDAIMAGE::Pointer im1, CUDAIMAGE::Pointer im2);
CUDAIMAGE::Pointer PreprocessImage(CUDAIMAGE::Pointer img, float low_val, float up_val);
void RestrictPhase(CUDAIMAGE::Pointer field, float3 phase);
void ContrainDefFields(CUDAIMAGE::Pointer ufield, CUDAIMAGE::Pointer dfield);
void ScaleUpdateField(CUDAIMAGE::Pointer field, float scale_factor);
void AddToUpdateField(CUDAIMAGE::Pointer updateField, CUDAIMAGE::Pointer updateField_temp,
                      float weight, bool normalize = true);
float SumImage(CUDAIMAGE::Pointer im1);
#endif
