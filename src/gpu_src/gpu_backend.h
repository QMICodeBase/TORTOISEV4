#ifndef _GPUSHIM_GPU_BACKEND_H
#define _GPUSHIM_GPU_BACKEND_H
// Backend binding for tools that drive every ported operation (the replay
// harness). One place selects the backend's headers and gives its context
// namespace a stable alias, so the harness itself - record loading, comparison,
// tolerance selection, the driver - is backend-agnostic and compiled once per
// target rather than duplicated per backend.
#ifdef USEMETAL
    #include "../metal_src/gpu_image.h"
    #include "../metal_src/metal_context.h"
    #include "../metal_src/resample_image.h"
    #include "../metal_src/warp_image.h"
    #include "../metal_src/quadratic_transform_image.h"
    #include "../metal_src/gaussian_smooth_image.h"
    #include "../metal_src/image_utilities.h"
    #include "../metal_src/compose_fields.h"
    #include "../metal_src/invert_field.h"
    #include "../metal_src/compute_entropy.h"
    #include "../metal_src/compute_metric.h"
    namespace gpuctx = mtlctx;
    #define GPU_BACKEND_NAME "Metal"
#elif defined(USEWEBGPU)
    #include "../webgpu_src/gpu_image.h"
    #include "../webgpu_src/webgpu_context.h"
    #include "../webgpu_src/resample_image.h"
    #include "../webgpu_src/warp_image.h"
    #include "../webgpu_src/quadratic_transform_image.h"
    #include "../webgpu_src/gaussian_smooth_image.h"
    #include "../webgpu_src/image_utilities.h"
    #include "../webgpu_src/compose_fields.h"
    #include "../webgpu_src/invert_field.h"
    #include "../webgpu_src/compute_entropy.h"
    #include "../webgpu_src/compute_metric.h"
    namespace gpuctx = wgpuctx;
    #define GPU_BACKEND_NAME "WebGPU"
#else
    // No silent default: a translation unit that reaches here would otherwise be
    // graded as WebGPU while linking against whatever backend the target supplies.
    #error "gpu_backend.h requires USEMETAL or USEWEBGPU"
#endif
#endif
