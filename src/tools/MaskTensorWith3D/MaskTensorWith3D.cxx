

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "itkDiffusionTensor3D.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"


    typedef float RealType;
    typedef itk::Image<RealType,4> ImageType4D;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::DiffusionTensor3D<RealType> TensorPixelType;
    typedef itk::Vector<RealType,6> VectorPixelType;
    
    typedef itk::Image<itk::DiffusionTensor3D<RealType>,3>         TensorImageType;
    typedef itk::Image<VectorPixelType,3>         VectorImageType;
    
    typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;
    
    
           

int main( int argc , char * argv[] )
{
    if(argc<3)
    {
        std::cout<<"Usage:  MaskTensorWith3D full_path_to_tensor_image full_path_to_3d_mask"<<std::endl;
        return 0;
    }
    
    
    std::string currdir;
    std::string nm(argv[1]);
    
   
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    

    typedef itk::ImageFileReader<ImageType4D> ImageType4DReaderType;
    ImageType4DReaderType::Pointer imager= ImageType4DReaderType::New();
    imager->SetFileName(nm);
    imager->Update();
    ImageType4D::Pointer tensor_img= imager->GetOutput();

    ImageType4D::SizeType sz4= tensor_img->GetLargestPossibleRegion().GetSize();


    typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
    ImageType3DReaderType::Pointer imager2= ImageType3DReaderType::New();
    imager2->SetFileName(argv[2]);
    imager2->Update();
    ImageType3D::Pointer mask_img= imager2->GetOutput();


    ImageType3D::SizeType sz= mask_img->GetLargestPossibleRegion().GetSize();

    ImageType4D::IndexType ind4;
    ImageType3D::IndexType ind3;


    for(int v=0;v<sz4[3];v++)
    {
        ind4[3]=v;
        for(int k=0;k<sz[2];k++)
        {
            ind4[2]=k;
            ind3[2]=k;
            for(int j=0;j<sz[1];j++)
            {
                ind4[1]=j;
                ind3[1]=j;
                for(int i=0;i<sz[0];i++)
                {
                    ind4[0]=i;
                    ind3[0]=i;

                    float mask_val = mask_img->GetPixel(ind3);
                    float tens_val = tensor_img->GetPixel(ind4)* mask_val;
                    tensor_img->SetPixel(ind4,tens_val);
                }
            }
        }
    }


    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_masked.nii");
       
    
    
    typedef itk::ImageFileWriter<ImageType4D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(tensor_img);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
