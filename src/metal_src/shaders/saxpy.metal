// Smoke-test kernel: proves the MSL embedding path and fp32 arithmetic.
// Mirrors shaders/saxpy.wgsl in the WebGPU backend.
#include <metal_stdlib>
using namespace metal;

constant uint WGX [[function_constant(0)]];
constant uint WGY [[function_constant(1)]];
constant uint WGZ [[function_constant(2)]];

kernel void main_saxpy(device const float *src [[buffer(0)]],
                       device       float *dst [[buffer(1)]],
                       constant     uint  &n   [[buffer(2)]],
                       uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if(i < n)
        dst[i] = src[i] * 2.0f + 1.0f;
}
