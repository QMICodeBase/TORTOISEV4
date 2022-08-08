

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
    typedef itk::Image<RealType,3> ImageType3D;

           

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
    

    typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
    ImageType3DReaderType::Pointer imager= ImageType3DReaderType::New();
    imager->SetFileName(nm);
    imager->Update();
    ImageType3D::Pointer image3D= imager->GetOutput();



    int pad_left= atoi(argv[2]);
    int pad_right= atoi(argv[3]);
    int pad_top= atoi(argv[4]);
    int pad_bottom= atoi(argv[5]);
    int pad_low= atoi(argv[6]);
    int pad_high= atoi(argv[7]);

    int change_header= atoi(argv[8]);


    ImageType3D::SizeType orig_size= image3D->GetLargestPossibleRegion().GetSize();

    ImageType3D::SizeType new_size;
    new_size[0]=orig_size[0]+pad_left+pad_right;
    new_size[1]=orig_size[1]+pad_top+pad_bottom;
    new_size[2]=orig_size[2]+pad_low+pad_high;




    ImageType3D::Pointer new_image=ImageType3D::New();
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,new_size);
    new_image->SetRegions(reg);
    new_image->Allocate();
    new_image->SetOrigin(image3D->GetOrigin());
    new_image->SetSpacing(image3D->GetSpacing());
    new_image->SetDirection(image3D->GetDirection());
    new_image->FillBuffer(0.);

    ImageType3D::IndexType index_old,index_new;

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


                new_image->SetPixel(index_new,image3D->GetPixel(index_old));

            }
        }
    }


    if(change_header)
    {
        itk::ContinuousIndex<double,3> cint;
        cint[0]=-pad_left;
        cint[1]=-pad_top;
        cint[2]=-pad_low;


        ImageType3D::PointType pt;
        image3D->TransformContinuousIndexToPhysicalPoint(cint,pt);
        new_image->SetOrigin(pt);
    }





    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_padded.nii");
       
    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(new_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
