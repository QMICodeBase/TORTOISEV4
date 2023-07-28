#ifndef _CUDAIMAGE_CXX
#define _CUDAIMAGE_CXX

#include "cuda_image.h"
#include "itkImportImageFilter.h"
#include "itkCastImageFilter.h"



CUDAIMAGE::~CUDAIMAGE()
{
    if(this->CudaArraydata !=nullptr)
    {
        cudaFreeArray(this->CudaArraydata);
        this->CudaArraydata=nullptr;

    }
    if(this->PitchedFloatData.ptr !=nullptr)
    {
        cudaFree(this->PitchedFloatData.ptr);
        this->PitchedFloatData.ptr=nullptr;


    }
}


CUDAIMAGE::Pointer CUDAIMAGE::New()
{
    Pointer p = std::make_shared<CUDAIMAGE>();
    return p;
}



void CUDAIMAGE::SetImageFromITK(ImageType3D::Pointer itk_image, bool create_texture)
{
    this->dir = itk_image->GetDirection();

    this->orig.x= itk_image->GetOrigin()[0];
    this->orig.y= itk_image->GetOrigin()[1];
    this->orig.z= itk_image->GetOrigin()[2];

    this->spc.x=  itk_image->GetSpacing()[0];
    this->spc.y=  itk_image->GetSpacing()[1];
    this->spc.z=  itk_image->GetSpacing()[2];

    ImageType3D::SizeType itk_size = itk_image->GetLargestPossibleRegion().GetSize();
    sz.x=itk_size[0];
    sz.y=itk_size[1];
    sz.z=itk_size[2];


    this->components_per_voxel=1;
    this->Allocate();

    copy3DHostToPitchedPtr((float*)itk_image->GetBufferPointer(),PitchedFloatData,sz.x,sz.y,sz.z);

    if(create_texture)
    {
        cudaExtent msz =  make_cudaExtent(itk_size[0],itk_size[1],itk_size[2]);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        gpuErrchk(cudaMalloc3DArray(&CudaArraydata, &channelDesc, msz));

        // --- Copy data to 3D array (host to device)
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = PitchedFloatData;
        copyParams.dstArray = CudaArraydata;
        copyParams.extent   = msz;
        copyParams.kind     = cudaMemcpyDeviceToDevice;
        gpuErrchk(cudaMemcpy3D(&copyParams));
        gpuErrchk(cudaFree(PitchedFloatData.ptr));
        PitchedFloatData.ptr=nullptr;

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));

        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = CudaArraydata;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));        
        texDesc.normalizedCoords=false;
        texDesc.filterMode=cudaFilterModeLinear;
        texDesc.addressMode[0]=cudaAddressModeBorder;
        texDesc.addressMode[1]=cudaAddressModeBorder;
        texDesc.addressMode[2]=cudaAddressModeBorder;
        texDesc.readMode=cudaReadModeElementType;

        gpuErrchk(cudaCreateTextureObject(&texobj, &resDesc, &texDesc, NULL));
    }
}


void CUDAIMAGE::DuplicateFromCUDAImage(CUDAIMAGE::Pointer cp_img)
{
    this->dir = cp_img->dir;
    this->orig= cp_img->orig;
    this->spc= cp_img->spc;
    this->sz= cp_img->sz;
    this->components_per_voxel=cp_img->components_per_voxel;
    this->Allocate();

    cudaExtent copy_extent = make_cudaExtent(cp_img->components_per_voxel*this->sz.x*sizeof(float),this->sz.y,this->sz.z);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = cp_img->getFloatdata();
    copyParams.dstPtr = this->getFloatdata();
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.extent = copy_extent;
    gpuErrchk(cudaMemcpy3D(&copyParams));

}


void CUDAIMAGE::Allocate()
{
    cudaExtent extent =  make_cudaExtent(this->components_per_voxel*sizeof(float)*this->sz.x,this->sz.y,this->sz.z);
    gpuErrchk(cudaMalloc3D(&PitchedFloatData, extent));
    cudaMemset3D(PitchedFloatData,0,extent);
}

void CUDAIMAGE::CreateTexture()
{
    cudaExtent msz =  make_cudaExtent(sz.x,sz.y,sz.z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    gpuErrchk(cudaMalloc3DArray(&CudaArraydata, &channelDesc, msz));

    // --- Copy data to 3D array (host to device)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = PitchedFloatData;
    copyParams.dstArray = CudaArraydata;
    copyParams.extent   = msz;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    gpuErrchk(cudaMemcpy3D(&copyParams));
    gpuErrchk(cudaFree(PitchedFloatData.ptr));
    PitchedFloatData.ptr=nullptr;

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = CudaArraydata;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords=false;
    texDesc.filterMode=cudaFilterModeLinear;
    texDesc.addressMode[0]=cudaAddressModeBorder;
    texDesc.addressMode[1]=cudaAddressModeBorder;
    texDesc.addressMode[2]=cudaAddressModeBorder;
    texDesc.readMode=cudaReadModeElementType;

    gpuErrchk(cudaCreateTextureObject(&texobj, &resDesc, &texDesc, NULL));

}



void CUDAIMAGE::SetImageFromITK(DisplacementFieldType::Pointer itk_field)
{
    this->dir = itk_field->GetDirection();

    this->orig.x= itk_field->GetOrigin()[0];
    this->orig.y= itk_field->GetOrigin()[1];
    this->orig.z= itk_field->GetOrigin()[2];

    this->spc.x=  itk_field->GetSpacing()[0];
    this->spc.y=  itk_field->GetSpacing()[1];
    this->spc.z=  itk_field->GetSpacing()[2];

    ImageType3D::SizeType itk_size = itk_field->GetLargestPossibleRegion().GetSize();
    sz.x=itk_size[0];
    sz.y=itk_size[1];
    sz.z=itk_size[2];        

    this->components_per_voxel=3;
    this->Allocate();

    using DisplacementFieldTypeFloat= itk::Image<itk::Vector<float,3>,3>;
    using FilterType = itk::CastImageFilter<DisplacementFieldType, DisplacementFieldTypeFloat>;
    auto filter = FilterType::New();
    filter->SetInput(itk_field);
    filter->Update();
    auto field2= filter->GetOutput();

    copy3DHostToPitchedPtr((float*)field2->GetBufferPointer(),PitchedFloatData,this->components_per_voxel*sz.x,sz.y,sz.z);
}

void CUDAIMAGE::SetTImageFromITK(DTMatrixImageType::Pointer tensor_img)
{

    this->dir = tensor_img->GetDirection();

    this->orig.x= tensor_img->GetOrigin()[0];
    this->orig.y= tensor_img->GetOrigin()[1];
    this->orig.z= tensor_img->GetOrigin()[2];

    this->spc.x=  tensor_img->GetSpacing()[0];
    this->spc.y=  tensor_img->GetSpacing()[1];
    this->spc.z=  tensor_img->GetSpacing()[2];

    ImageType3D::SizeType itk_size = tensor_img->GetLargestPossibleRegion().GetSize();
    sz.x=itk_size[0];
    sz.y=itk_size[1];
    sz.z=itk_size[2];

    this->components_per_voxel=6;
    this->Allocate();


    float *temp_data= new float[sz.x*sz.y*sz.z*6];
    itk::ImageRegionIteratorWithIndex<DTMatrixImageType> it(tensor_img,tensor_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        DTMatrixImageType::IndexType ind3= it.GetIndex();
        DTMatrixImageType::PixelType mat =it.Get();

        long lin_ind = ind3[2]*sz.y*sz.x*6;
        lin_ind+=ind3[1]*sz.x*6;
        lin_ind+=ind3[0]*6;

        temp_data[lin_ind+0] = mat(0,0);
        temp_data[lin_ind+1] = mat(0,1);
        temp_data[lin_ind+2] = mat(0,2);
        temp_data[lin_ind+3] = mat(1,1);
        temp_data[lin_ind+4] = mat(1,2);
        temp_data[lin_ind+5] = mat(2,2);
    }


    copy3DHostToPitchedPtr((float*)temp_data,PitchedFloatData,this->components_per_voxel*sz.x,sz.y,sz.z);

    delete[] temp_data;

}


CUDAIMAGE::TensorVectorImageType::Pointer CUDAIMAGE::CudaImageToITKImage4D()
{
    ImageType3D::SizeType sz2;
    sz2[0]= this->sz.x;
    sz2[1]= this->sz.y;
    sz2[2]= this->sz.z;
    ImageType3D::IndexType start;
    start.Fill(0);
    ImageType3D::RegionType reg(start,sz2);

    ImageType3D::PointType orig;
    orig[0]= this->orig.x;
    orig[1]= this->orig.y;
    orig[2]= this->orig.z;

    ImageType3D::SpacingType spc;
    spc[0]= this->spc.x;
    spc[1]= this->spc.y;
    spc[2]= this->spc.z;

    float * itk_image_data2 = new float[(long)sz2[0]*sz2[1]*sz2[2]*this->components_per_voxel];
    copy3DPitchedPtrToHost(PitchedFloatData,itk_image_data2,this->components_per_voxel*sz.x,sz.y,sz.z);

    TensorVectorImageType::PixelType* itk_image_data = (TensorVectorImageType::PixelType*)itk_image_data2 ;

    typedef itk::ImportImageFilter< TensorVectorImageType::PixelType , 3 >   ImportFilterType;
    ImportFilterType::Pointer importFilter = ImportFilterType::New();
    importFilter->SetRegion( reg );
    importFilter->SetOrigin( orig );
    importFilter->SetSpacing( spc );
    importFilter->SetDirection(dir);

    const bool importImageFilterWillOwnTheBuffer = true;
    importFilter->SetImportPointer( itk_image_data, (long)sz2[0]*sz2[1]*sz2[2]*this->components_per_voxel,    importImageFilterWillOwnTheBuffer );
    importFilter->Update();
    TensorVectorImageType::Pointer itk_image_float= importFilter->GetOutput();

    return itk_image_float;

}



CUDAIMAGE::ImageType3D::Pointer CUDAIMAGE::CudaImageToITKImage()
{    

    ImageType3D::SizeType sz2;
    sz2[0]= this->sz.x;
    sz2[1]= this->sz.y;
    sz2[2]= this->sz.z;
    ImageType3D::IndexType start;
    start.Fill(0);
    ImageType3D::RegionType reg(start,sz2);

    ImageType3D::PointType orig;
    orig[0]= this->orig.x;
    orig[1]= this->orig.y;
    orig[2]= this->orig.z;

    ImageType3D::SpacingType spc;
    spc[0]= this->spc.x;
    spc[1]= this->spc.y;
    spc[2]= this->spc.z;


    float * itk_image_data = new float[(long)sz2[0]*sz2[1]*sz2[2]];

    copy3DPitchedPtrToHost(PitchedFloatData,itk_image_data,sz.x,sz.y,sz.z);

    typedef itk::ImportImageFilter< float, 3 >   ImportFilterType;
    ImportFilterType::Pointer importFilter = ImportFilterType::New();
    importFilter->SetRegion( reg );
    importFilter->SetOrigin( orig );
    importFilter->SetSpacing( spc );
    importFilter->SetDirection(dir);

    const bool importImageFilterWillOwnTheBuffer = true;
    importFilter->SetImportPointer( itk_image_data, (long)sz2[0]*sz2[1]*sz2[2],    importImageFilterWillOwnTheBuffer );
    importFilter->Update();
    ImageType3D::Pointer itk_image= importFilter->GetOutput();

    return itk_image;

}




CUDAIMAGE::DisplacementFieldType::Pointer CUDAIMAGE::CudaImageToITKField()
{
    if(this->components_per_voxel!=3)
        return nullptr;

    ImageType3D::SizeType sz2;
    sz2[0]= this->sz.x;
    sz2[1]= this->sz.y;
    sz2[2]= this->sz.z;
    ImageType3D::IndexType start;
    start.Fill(0);
    ImageType3D::RegionType reg(start,sz2);

    ImageType3D::PointType orig;
    orig[0]= this->orig.x;
    orig[1]= this->orig.y;
    orig[2]= this->orig.z;

    ImageType3D::SpacingType spc;
    spc[0]= this->spc.x;
    spc[1]= this->spc.y;
    spc[2]= this->spc.z;



    float * itk_image_data2 = new float[(long)sz2[0]*sz2[1]*sz2[2]*3];
    copy3DPitchedPtrToHost(PitchedFloatData,itk_image_data2,3*sz.x,sz.y,sz.z);



    itk::Vector<float,3>* itk_image_data = (itk::Vector<float,3>*)itk_image_data2 ;

    typedef itk::ImportImageFilter< DisplacementFieldTypeFloat::PixelType , 3 >   ImportFilterType;
    ImportFilterType::Pointer importFilter = ImportFilterType::New();
    importFilter->SetRegion( reg );
    importFilter->SetOrigin( orig );
    importFilter->SetSpacing( spc );
    importFilter->SetDirection(dir);

    const bool importImageFilterWillOwnTheBuffer = true;
    importFilter->SetImportPointer( itk_image_data, (long)sz2[0]*sz2[1]*sz2[2]*3,    importImageFilterWillOwnTheBuffer );
    importFilter->Update();
    DisplacementFieldTypeFloat::Pointer itk_image_float= importFilter->GetOutput();
    
    using FilterType = itk::CastImageFilter<DisplacementFieldTypeFloat, DisplacementFieldType>;
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput(itk_image_float);
    filter->Update();
    DisplacementFieldType::Pointer itk_image=filter->GetOutput();
    return itk_image;

}



#endif
