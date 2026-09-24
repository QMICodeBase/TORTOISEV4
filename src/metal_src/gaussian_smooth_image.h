#ifndef _MTL_GAUSSIANSMOOTH_H
#define _MTL_GAUSSIANSMOOTH_H
#include "gpu_image.h"
CUDAIMAGE::Pointer GaussianSmoothImage(CUDAIMAGE::Pointer main_image, float std);

// Same smoothing with the kernel taps supplied directly. The production wrapper
// derives taps from `std` via itk::GaussianOperator and delegates here; tests use
// it to pin the convolution against an exact reference without having to
// reproduce ITK's tap generation.
#include <vector>
CUDAIMAGE::Pointer GaussianSmoothImageWithTaps(CUDAIMAGE::Pointer main_image,
                                               const std::vector<float> &taps, float std);
#endif
