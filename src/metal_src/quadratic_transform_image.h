#ifndef _MTL_QUADTRANSFORM_H
#define _MTL_QUADTRANSFORM_H
#include "gpu_image.h"
#include "itkOkanQuadraticTransform.h"
using TransformType = itk::OkanQuadraticTransform<double,3,3>;
CUDAIMAGE::Pointer QuadraticTransformImageC(CUDAIMAGE::Pointer img, TransformType::Pointer tr,
                                            CUDAIMAGE::Pointer target_img);
#endif
