
#include "compute_wp_map.h"
    



int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<< "Computes the Westin-Planar map from the diffusion tensor"<<std::endl;
        std::cout<<"Usage:   ComputeWPMap full_path_to_tensor_image"<<std::endl;
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

    ImageType3D::Pointer scalar_image = compute_wp(image4D);
    
       
        
    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_WP.nii");
       
    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(scalar_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
