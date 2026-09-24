#ifndef _WGPU_REDUCTIONS_H
#define _WGPU_REDUCTIONS_H
#include "gpu_image.h"

// Two-stage reductions matching cuda_image_utilities.cu's shape (24 workgroups
// of 1024, then a single workgroup over the 24 partials).
enum class ReduceOp { Sum = 0, SumSq = 1, Max = 2, MaxFieldNorm = 3, Min = 4 };

// `grid`/`block` default to cuda_image_utilities.cu's 24x1024. compute_entropy.cu
// uses a single block of 256 (ScalarFindSum2); pass 1,256 to match its summation
// order, which matters because float addition is not associative.
float Reduce(const wgpu::Buffer &buf, size_t n_elements, ReduceOp op,
             float3 spc = float3{1.f, 1.f, 1.f},
             uint32_t grid = 24, uint32_t block = 1024);
#endif
