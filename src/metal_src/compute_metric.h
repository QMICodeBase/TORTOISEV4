#ifndef _MTL_COMPUTEMETRIC_H
#define _MTL_COMPUTEMETRIC_H
#include "gpu_image.h"
#include "itkGaussianOperator.h"
#include <vector>

float ComputeMetric_CC(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                       CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM);

float ComputeMetric_CCSK(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                         const CUDAIMAGE::Pointer str_img,
                         CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                         float t = 0.5);

float ComputeMetric_MSJac(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                          const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                          CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                          float3 phase_vector, itk::GaussianOperator<float, 3> &oper);

float ComputeMetric_CCJacS(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                           const CUDAIMAGE::Pointer str_img,
                           const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                           CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                           float3 phase_vector, itk::GaussianOperator<float, 3> &oper);

// Taps-based entry points. The operator-taking wrappers above extract the taps
// and delegate here; tests use these directly rather than reconstructing an ITK
// operator, whose CreateToRadius builds a SQUARE neighbourhood, not the 1-D
// directional one these kernels expect.
float ComputeMetric_MSJacWithTaps(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                                  const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                                  CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                                  float3 phase_vector, const std::vector<float> &taps);

float ComputeMetric_CCJacSWithTaps(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                                   const CUDAIMAGE::Pointer str_img,
                                   const CUDAIMAGE::Pointer def_FINV, const CUDAIMAGE::Pointer def_MINV,
                                   CUDAIMAGE::Pointer &updateFieldF, CUDAIMAGE::Pointer &updateFieldM,
                                   float3 phase_vector, const std::vector<float> &taps);
#endif
