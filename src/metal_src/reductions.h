#ifndef _MTL_REDUCTIONS_H
#define _MTL_REDUCTIONS_H
#include "gpu_image.h"

// Two-stage reductions matching cuda_image_utilities.cu's shape (24 workgroups
// of 1024, then a single workgroup over the 24 partials).
enum class ReduceOp { Sum = 0, SumSq = 1, Max = 2, MaxFieldNorm = 3, Min = 4 };

// `grid`/`block` default to cuda_image_utilities.cu's 24x1024. compute_entropy.cu
// uses a single block of 256 (ScalarFindSum2); pass 1,256 to match its summation
// order, which matters because float addition is not associative.
float Reduce(const mtlctx::Buffer &buf, size_t n_elements, ReduceOp op,
             float3 spc = float3{1.f, 1.f, 1.f},
             uint32_t grid = 24, uint32_t block = 1024);

// Same reduction, but the scalar is LEFT ON THE DEVICE at element 0 of the returned
// buffer instead of being read back. Reduce() is this plus a Download.
//
// Why it exists: ComputeJointEntropy needs three of its sums only as a divisor
// inside the next shader. Reading those back cost a full queue drain each - and a
// drain waits for every OMP thread's queued work, not just this one's. Consuming the
// float on the device is bit-identical (it is the same float32 the reduction wrote)
// and removes the round-trip.
mtlctx::Buffer ReduceToBuffer(const mtlctx::Buffer &buf, size_t n_elements, ReduceOp op,
                              float3 spc = float3{1.f, 1.f, 1.f},
                              uint32_t grid = 24, uint32_t block = 1024);
#endif
