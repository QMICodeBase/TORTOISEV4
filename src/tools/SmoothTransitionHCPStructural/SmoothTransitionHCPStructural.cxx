

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageDuplicator.h"

#include "itkImageRegionIteratorWithIndex.h"


    typedef float RealType;
    typedef itk::Image<RealType,3> ImageType3D;

    
           

int main( int argc , char * argv[] )
{
    if(argc<4)
    {
        std::cout<<"Usage:   SmoothTransitionHCPStructural  unmasked_T2_image  equal_weight_slice_number output_name  mask_img(optional)"<<std::endl;
        return EXIT_FAILURE;
    }
    
    
    std::string currdir;
    std::string nm(argv[1]);

    std::string outputname(argv[3]);
    int slice= atoi(argv[2]);

            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    

    typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
    ImageType3DReaderType::Pointer imager2= ImageType3DReaderType::New();
    imager2->SetFileName(argv[1]);
    imager2->Update();
    ImageType3D::Pointer unmasked_img= imager2->GetOutput();


    std::string masked_img_file;


    if(argc<5)
    {
        masked_img_file= currdir + std::string("/skull_strip_out.nii.gz");

        char cmd[1000];
      //  sprintf(cmd,"rm %s",masked_img_file.c_str());
      //  system(cmd);

      //  sprintf(cmd, "export FSLOUTPUTTYPE=NIFTI_GZ;3dSkullStrip -touchup -input %s -prefix %s -orig_vol -fac 1000",  argv[1],masked_img_file.c_str());


        //sprintf(cmd, "export FSLOUTPUTTYPE=NIFTI_GZ;bet2 %s %s -f 0.3 -g -0.1",  argv[1],masked_img_file.c_str());
        sprintf(cmd, "3dSkullStrip -input %s -prefix %s -orig_vol",  argv[1],masked_img_file.c_str());
        system(cmd);
    }
    else
    {
        masked_img_file=argv[4];
    }


    ImageType3DReaderType::Pointer imager3= ImageType3DReaderType::New();
    imager3->SetFileName(masked_img_file);
    imager3->Update();
    ImageType3D::Pointer masked_img= imager3->GetOutput();



   // sprintf(cmd,"rm %s", masked_img_file.c_str());
   // system(cmd);


    typedef itk::ImageDuplicator<ImageType3D> DupType;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(masked_img);
    dup->Update();
    ImageType3D::Pointer output_image= dup->GetOutput();
    output_image->FillBuffer(0);



    ImageType3D::SizeType sz= masked_img->GetLargestPossibleRegion().GetSize();
    ImageType3D::IndexType index;
    for(int k=0;k<sz[2];k++)
    {

        float x2= (k-slice+80)*3./40. -6;
        float unmasked_weight= 1./ (1+ exp(-1.5*x2));
        float masked_weight= 1- unmasked_weight;


        index[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            index[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                index[0]=i;

                float val = unmasked_weight * unmasked_img->GetPixel(index) + masked_weight * masked_img->GetPixel(index);
                output_image->SetPixel(index,val);
            }
        }
    }

    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(argv[3]);
    writer->SetInput(output_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
