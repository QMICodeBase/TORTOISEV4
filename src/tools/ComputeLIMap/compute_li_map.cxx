

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "compute_li_map.h"


    typedef itk::DiffusionTensor3D<RealType> TensorPixelType;
    typedef itk::Vector<RealType,6> VectorPixelType;
    
    typedef itk::Image<itk::DiffusionTensor3D<RealType>,3>         TensorImageType;
    typedef itk::Image<VectorPixelType,3>         VectorImageType;
    

    
    
           

int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<<"Usage:   ComputeLIMap full_path_to_tensor_image full_path_to_AM_image (optional)"<<std::endl;
        std::cout<<"This executable assumes there is an AM image in the same folder as DT image if one is not provided."<<std::endl;
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


    ImageType3D::Pointer A0_image=nullptr;
    if(argc>2)
    {
        typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
        ImageType3DReaderType::Pointer imager2= ImageType3DReaderType::New();
        imager2->SetFileName(argv[2]);
        imager2->Update();
        A0_image= imager2->GetOutput();

    }
    else
    {
        std::string filename(argv[1]);
        std::string::size_type idx=filename.rfind("DT.");
        std::string basename= filename.substr(mypos+1,idx-mypos-1);
        std::string A0name=currdir + basename + std::string("AM.nii");

        typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
        ImageType3DReaderType::Pointer imager2= ImageType3DReaderType::New();
        imager2->SetFileName(A0name);
        imager2->Update();
        A0_image= imager2->GetOutput();
    }


    ImageType3D::Pointer li_map= compute_li_map(image4D,A0_image);



    
       
        
         std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_LI.nii");
       
    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(li_map);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
