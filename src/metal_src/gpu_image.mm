#ifndef _GPU_IMAGE_CXX
#define _GPU_IMAGE_CXX

#include "gpu_image.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "itkImportImageFilter.h"

void GPUIMAGE::Allocate()
{
    data.pitch = (size_t)components_per_voxel * sz.x * sizeof(float);
    data.buf   = mtlctx::CreateStorage(Bytes());
    data.ptr   = data.buf.get();
    mtlctx::ZeroFresh(data.buf, Bytes());
}

void GPUIMAGE::AllocateFrom(const void *src)
{
    data.pitch = (size_t)components_per_voxel * sz.x * sizeof(float);
    data.buf   = mtlctx::CreateStorageFrom(src, Bytes());
    data.ptr   = data.buf.get();
}

void GPUIMAGE::FillBuffer(float val)
{
    // FAITHFUL MIMICRY OF A REFERENCE BUG - do not "fix" this (CLAUDE.md 0.0).
    //
    // CUDAIMAGE::FillBuffer (cuda_image.h:51-55) calls
    //     cudaMemset3D(PitchedFloatData, val, extent)
    // whose second parameter is an **int byte value**, not a float. So the float
    // argument is truncated to int and its low byte is written to every BYTE of
    // the region - it does NOT set each float to `val`:
    //
    //     FillBuffer(0)   -> every byte 0x00 -> 0.0f          (agrees by luck)
    //     FillBuffer(1)   -> every byte 0x01 -> 2.3694e-38    (NOT 1.0f)
    //     FillBuffer(-1)  -> every byte 0xFF -> NaN           (NOT -1.0f)
    //
    // This port previously filled each float with `val`, which is what the call
    // site plainly intends and is what ITK's FillBuffer does - but it is NOT what
    // the reference computes.
    //
    // REACHABILITY, checked rather than assumed: this overload has NO call site in
    // the GPU pipeline at all. Every apparent one binds to ITK's FillBuffer
    // instead - they pass an itk::Vector (run_drbuddi_stage.cxx:193-220, inside the
    // #else CPU branch that closes at :222) or act on an ImageType3D
    // (create_mask.cxx:107, DIFFPREP.cxx:1321, FINALDATA.cxx:985,
    // compute_metrics_*.h). So the divergence is LATENT, not merely dormant, and no
    // golden vector covers this function. Reproducing the byte fill makes
    // port == reference by construction if anyone ever wires it up.
    const unsigned char byte = (unsigned char)(int)val;
    std::vector<unsigned char> v(Bytes(), byte);
    mtlctx::Upload(data.buf, v.data(), Bytes());
}

void GPUIMAGE::DuplicateFromCUDAImage(GPUIMAGE::Pointer cp)
{
    dir  = cp->dir;
    orig = cp->orig;
    spc  = cp->spc;
    sz   = cp->sz;
    components_per_voxel = cp->components_per_voxel;
    Allocate();

    // Device-to-device. This was Download-then-Upload, i.e. a blocking buffer map
    // plus two full host transits of the whole image, several times per DRBUDDI
    // iteration.
    mtlctx::CopyBuffer(data.buf, cp->getFloatdata().buf, Bytes());
}

void GPUIMAGE::SetImageFromITK(ImageType3D::Pointer itk_image, bool)
{
    dir = itk_image->GetDirection();
    ImageType3D::PointType   o = itk_image->GetOrigin();
    ImageType3D::SpacingType s = itk_image->GetSpacing();
    ImageType3D::SizeType    n = itk_image->GetLargestPossibleRegion().GetSize();
    orig = make_float3(o[0], o[1], o[2]);
    spc  = make_float3(s[0], s[1], s[2]);
    sz   = make_int3(n[0], n[1], n[2]);
    components_per_voxel = 1;

    AllocateFrom(itk_image->GetBufferPointer());
}

void GPUIMAGE::SetImageFromITK(DisplacementFieldType::Pointer itk_field)
{
    dir = itk_field->GetDirection();
    DisplacementFieldType::PointType   o = itk_field->GetOrigin();
    DisplacementFieldType::SpacingType s = itk_field->GetSpacing();
    DisplacementFieldType::SizeType    n = itk_field->GetLargestPossibleRegion().GetSize();
    orig = make_float3(o[0], o[1], o[2]);
    spc  = make_float3(s[0], s[1], s[2]);
    sz   = make_int3(n[0], n[1], n[2]);
    components_per_voxel = 3;

    // The ITK field stores double-precision vectors; the device side is fp32.
    const size_t nv = NumVoxels();
    std::vector<float> host(nv * 3);
    DisplacementFieldType::PixelType *src = itk_field->GetBufferPointer();
    for(size_t i = 0; i < nv; i++)
    {
        host[3 * i + 0] = (float)src[i][0];
        host[3 * i + 1] = (float)src[i][1];
        host[3 * i + 2] = (float)src[i][2];
    }
    AllocateFrom(host.data());
}

void GPUIMAGE::SetTImageFromITK(DTMatrixImageType::Pointer tensor_img)
{
    dir = tensor_img->GetDirection();
    DTMatrixImageType::PointType   o = tensor_img->GetOrigin();
    DTMatrixImageType::SpacingType s = tensor_img->GetSpacing();
    DTMatrixImageType::SizeType    n = tensor_img->GetLargestPossibleRegion().GetSize();
    orig = make_float3(o[0], o[1], o[2]);
    spc  = make_float3(s[0], s[1], s[2]);
    sz   = make_int3(n[0], n[1], n[2]);
    components_per_voxel = 6;

    const size_t nv = NumVoxels();
    std::vector<float> host(nv * 6);
    DTMatrixImageType::PixelType *src = tensor_img->GetBufferPointer();
    for(size_t i = 0; i < nv; i++)
    {
        host[6 * i + 0] = (float)src[i](0, 0);
        host[6 * i + 1] = (float)src[i](0, 1);
        host[6 * i + 2] = (float)src[i](0, 2);
        host[6 * i + 3] = (float)src[i](1, 1);
        host[6 * i + 4] = (float)src[i](1, 2);
        host[6 * i + 5] = (float)src[i](2, 2);
    }
    AllocateFrom(host.data());
}

GPUIMAGE::ImageType3D::Pointer GPUIMAGE::CudaImageToITKImage()
{
    ImageType3D::SizeType n;
    n[0] = sz.x; n[1] = sz.y; n[2] = sz.z;
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start, n);
    ImageType3D::PointType   o;   o[0]  = orig.x; o[1]  = orig.y; o[2]  = orig.z;
    ImageType3D::SpacingType sp;  sp[0] = spc.x;  sp[1] = spc.y;  sp[2] = spc.z;

    // Reads NumVoxels() floats regardless of components_per_voxel, exactly as
    // CUDAIMAGE::CudaImageToITKImage does. Only ever called on scalar images.
    const size_t nv = NumVoxels();
    // ITK takes ownership of this buffer (SetImportPointer with the transfer flag).
    float *buf = new float[nv];
    mtlctx::Download(data.buf, buf, nv * sizeof(float));

    typedef itk::ImportImageFilter<float, 3> ImportFilterType;
    ImportFilterType::Pointer imp = ImportFilterType::New();
    imp->SetRegion(reg);
    imp->SetOrigin(o);
    imp->SetSpacing(sp);
    imp->SetDirection(dir);
    imp->SetImportPointer(buf, nv, true);
    imp->Update();
    return imp->GetOutput();
}

GPUIMAGE::DisplacementFieldType::Pointer GPUIMAGE::CudaImageToITKField()
{
    // The reference returns nullptr on a non-3-component image
    // (cuda_image.cxx:328-329). Without this the port issues a readback of 3x the
    // buffer size, which here is a memcpy that would read out of bounds - i.e. the
    // port would be LESS defensive than the reference, which CLAUDE.md 0.0 forbids in
    // both directions. (Distinct from CudaImageToITKImage, where the reference
    // genuinely has no check and the port's was correctly removed.)
    if(components_per_voxel != 3)
        return nullptr;

    DisplacementFieldType::SizeType n;
    n[0] = sz.x; n[1] = sz.y; n[2] = sz.z;
    DisplacementFieldType::IndexType start; start.Fill(0);
    DisplacementFieldType::RegionType reg(start, n);

    DisplacementFieldType::Pointer f = DisplacementFieldType::New();
    f->SetRegions(reg);
    DisplacementFieldType::PointType   o;  o[0]  = orig.x; o[1]  = orig.y; o[2]  = orig.z;
    DisplacementFieldType::SpacingType sp; sp[0] = spc.x;  sp[1] = spc.y;  sp[2] = spc.z;
    f->SetOrigin(o);
    f->SetSpacing(sp);
    f->SetDirection(dir);
    f->Allocate();

    const size_t nv = NumVoxels();
    std::vector<float> host(nv * 3);
    mtlctx::Download(data.buf, host.data(), host.size() * sizeof(float));

    DisplacementFieldType::PixelType *dst = f->GetBufferPointer();
    for(size_t i = 0; i < nv; i++)
    {
        dst[i][0] = host[3 * i + 0];
        dst[i][1] = host[3 * i + 1];
        dst[i][2] = host[3 * i + 2];
    }
    return f;
}

GPUIMAGE::TensorVectorImageType::Pointer GPUIMAGE::CudaImageToITKImage4D()
{
    TensorVectorImageType::SizeType n;
    n[0] = sz.x; n[1] = sz.y; n[2] = sz.z;
    TensorVectorImageType::IndexType start; start.Fill(0);
    TensorVectorImageType::RegionType reg(start, n);

    TensorVectorImageType::Pointer img = TensorVectorImageType::New();
    img->SetRegions(reg);
    TensorVectorImageType::PointType   o;  o[0]  = orig.x; o[1]  = orig.y; o[2]  = orig.z;
    TensorVectorImageType::SpacingType sp; sp[0] = spc.x;  sp[1] = spc.y;  sp[2] = spc.z;
    img->SetOrigin(o);
    img->SetSpacing(sp);
    img->SetDirection(dir);
    img->Allocate();

    const size_t nv = NumVoxels();
    const int    nc = components_per_voxel;
    std::vector<float> host(nv * nc);
    mtlctx::Download(data.buf, host.data(), host.size() * sizeof(float));

    TensorVectorImageType::PixelType *dst = img->GetBufferPointer();
    // `c < 6` is NOT defence the reference lacks - it is required by the differing
    // implementation strategy. The reference reinterprets the flat host buffer as
    // itk::Vector<float,6>* (cuda_image.cxx:258-261), so it never indexes a
    // component; this port copies component-wise into a fixed 6-vector, and
    // without the bound nc > 6 would write past dst[i]. At nc == 6 - the only
    // value this is ever called with - the two are identical.
    for(size_t i = 0; i < nv; i++)
        for(int c = 0; c < nc && c < 6; c++)
            dst[i][c] = host[nc * i + c];
    return img;
}

#endif
