

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
        std::cout<<"Usage:  Pad4DImage  input_image  pad_#_voxels_on_left pad_#_voxels_on_right pad_#_voxels_on_top pad_#_voxels_on_bottom pad_#_voxels_on_low pad_#_voxels_on_high  chnage_origin_accordingly? (0 or 1) "<<std::endl;
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
    ImageType4D::Pointer image4D= imager->GetOutput();



    int pad_left= atoi(argv[2]);
    int pad_right= atoi(argv[3]);
    int pad_top= atoi(argv[4]);
    int pad_bottom= atoi(argv[5]);
    int pad_low= atoi(argv[6]);
    int pad_high= atoi(argv[7]);

    int change_header= atoi(argv[8]);


    ImageType4D::SizeType orig_size= image4D->GetLargestPossibleRegion().GetSize();

    ImageType4D::SizeType new_size;
    new_size[0]=orig_size[0]+pad_left+pad_right;
    new_size[1]=orig_size[1]+pad_top+pad_bottom;
    new_size[2]=orig_size[2]+pad_low+pad_high;
    new_size[3]=orig_size[3];



    ImageType4D::Pointer new_image=ImageType4D::New();
    ImageType4D::IndexType start; start.Fill(0);
    ImageType4D::RegionType reg(start,new_size);
    new_image->SetRegions(reg);
    new_image->Allocate();
    new_image->SetOrigin(image4D->GetOrigin());
    new_image->SetSpacing(image4D->GetSpacing());
    new_image->SetDirection(image4D->GetDirection());
    new_image->FillBuffer(0.);

    ImageType4D::IndexType index_old,index_new;

    for(int k=0;k<orig_size[2];k++)
    {
        index_old[2]=k;
        index_new[2]=k+pad_low;
        for(int j=0;j<orig_size[1];j++)
        {
            index_old[1]=j;
            index_new[1]=j+pad_top;
            for(int i=0;i<orig_size[0];i++)
            {
                index_old[0]=i;
                index_new[0]=i+pad_left;

                for(int v=0;v<orig_size[3];v++)
                {
                    index_old[3]=v;
                    index_new[3]=v;
                    new_image->SetPixel(index_new,image4D->GetPixel(index_old));
                }
            }
        }
    }


    if(change_header)
    {
        itk::ContinuousIndex<double,4> cint;
        cint[0]=-pad_left;
        cint[1]=-pad_top;
        cint[2]=-pad_low;
        cint[3]=0;

        ImageType4D::PointType pt;
        image4D->TransformContinuousIndexToPhysicalPoint(cint,pt);
        new_image->SetOrigin(pt);
    }





    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_padded.nii");
       
    
    
    typedef itk::ImageFileWriter<ImageType4D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(new_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
