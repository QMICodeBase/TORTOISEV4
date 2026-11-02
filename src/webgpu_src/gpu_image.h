#ifndef _GPU_IMAGE_H
#define _GPU_IMAGE_H

// GPUIMAGE - the WebGPU counterpart of CUDAIMAGE, with the same public surface.
//
// Under USEWEBGPU this is typedef'd to CUDAIMAGE at the bottom of the file, so
// DRBUDDI_Diffeo.h's `using CurrentImageType = CUDAIMAGE` and every call site in
// main/ compile unchanged. Keeping the CUDA-flavoured names is deliberate:
// renaming the type across the tree would be a large, non-surgical diff.
//
// Differences from CUDAIMAGE that matter:
//   * no pitched allocation - rows are exactly components*sz.x*4 bytes, so the
//     kernels' explicit pitch arithmetic carries over unchanged and there is no
//     padding for reductions to pick up (CLAUDE.md 5.5)
//   * CreateTexture() is a no-op. WebGPU has no clamp-to-border address mode, so
//     sampling is done manually in WGSL over the same storage buffer (CLAUDE.md 5.2)

#include <memory>

#include "itkImage.h"
#include "itkDisplacementFieldTransform.h"

#include "webgpu_context.h"

// CUDA vector types used in signatures across main/; provided here so those
// files need no edits under USEWEBGPU.
#ifndef __CUDACC__
struct float3 { float x, y, z; };
struct int3   { int   x, y, z; };
inline float3 make_float3(float x, float y, float z) { float3 v; v.x = x; v.y = y; v.z = z; return v; }
inline int3   make_int3(int x, int y, int z)         { int3   v; v.x = x; v.y = y; v.z = z; return v; }
#endif

// Stands in for cudaPitchedPtr: a buffer plus the row stride in bytes.
struct GPUData
{
    wgpu::Buffer buf;
    size_t       pitch{0};      // bytes per row  (== components*sz.x*sizeof(float))
    // Mirrors cudaPitchedPtr::ptr so existing `getFloatdata().ptr != nullptr`
    // allocation tests in main/ keep working unchanged.
    void        *ptr{nullptr};
    explicit operator bool() const { return (bool)buf; }
};

class GPUIMAGE
{
public:
    using DataType    = float;
    using ImageType3D = itk::Image<DataType, 3>;
    using ImageType4D = itk::Image<DataType, 4>;
    typedef itk::DisplacementFieldTransform<double, 3> DisplacementTransformType;
    typedef itk::DisplacementFieldTransform<float, 3>  DisplacementTransformTypeFloat;
    typedef DisplacementTransformType::DisplacementFieldType      DisplacementFieldType;
    typedef DisplacementTransformTypeFloat::DisplacementFieldType DisplacementFieldTypeFloat;

    using InternalMatrixType = vnl_matrix_fixed<double, 3, 3>;
    using DTMatrixImageType  = itk::Image<InternalMatrixType, 3>;
    using TensorVectorType      = itk::Vector<float, 6>;
    using TensorVectorImageType = itk::Image<TensorVectorType, 3>;

    using Self    = GPUIMAGE;
    using Pointer = std::shared_ptr<Self>;
    static Pointer New() { return std::make_shared<GPUIMAGE>(); }

    GPUIMAGE() {}
    ~GPUIMAGE() {}

    void DuplicateFromCUDAImage(GPUIMAGE::Pointer cp_img);
    void SetImageFromITK(ImageType3D::Pointer itk_image, bool create_texture = false);
    void SetImageFromITK(DisplacementFieldType::Pointer itk_field);
    void SetTImageFromITK(DTMatrixImageType::Pointer tensor_img);

    void FillBuffer(float val);
    void Allocate();

    ImageType3D::Pointer          CudaImageToITKImage();
    TensorVectorImageType::Pointer CudaImageToITKImage4D();
    DisplacementFieldType::Pointer CudaImageToITKField();

    GPUData getFloatdata() { return data; }
    void    SetFloatDataPointer(GPUData d) { data = d; }
    ImageType3D::DirectionType GetDirection() { return dir; }

    // No sampler is used (WebGPU has no border address mode), so the "texture"
    // is the same storage buffer and this is a no-op. It stays in the API so the
    // call sites in main/ and the wrappers need no edits.
    void    CreateTexture() {}
    GPUData GetTexture() { return data; }

    size_t NumVoxels() const { return (size_t)sz.x * sz.y * sz.z; }
    size_t NumFloats() const { return NumVoxels() * components_per_voxel; }
    size_t Bytes()     const { return NumFloats() * sizeof(float); }

    ImageType3D::DirectionType dir;
    float3 orig{};
    float3 spc{};
    int3   sz{};
    int    components_per_voxel{1};

private:
    GPUData data;
};

using CUDAIMAGE = GPUIMAGE;

#endif
