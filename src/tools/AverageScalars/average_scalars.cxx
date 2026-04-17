

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


    typedef double RealType;     
    typedef itk::Image<RealType,3> ImageType3D;
           

int main( int argc , char * argv[] )
{
    if(argc<3)    
    {
        std::cout<<"Usage:   AverageScalars   full_path_to_textfile_containing_list_of_scalars  name_of_output_image "<<std::endl;
        return EXIT_FAILURE;
    }
    
    ifstream inFile(argv[1]);
    if (!inFile) 
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return EXIT_FAILURE;
    }
    
    
    int N_images=0;
    ImageType3D::Pointer average_image=nullptr;

    
    std::string currdir;
    std::string nm(argv[1]);
    
   
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);

        
    ImageType3D::SizeType imsize;
    
    string line;
    while (getline(inFile, line)) 
    {
        if (line.empty()) 
            continue;
        
        std::string file_name=line;
        FILE * fp= fopen(file_name.c_str(),"rb");
        
        if(!fp)
        {
            file_name= currdir + file_name;
            
            FILE * fp2= fopen(file_name.c_str(),"rb");
            if(!fp2)
            {            
                std::cout<< "File " << line << " does not exist. Exiting!" << std::endl;
                return EXIT_FAILURE;
            }
            else
               fclose(fp2);
        }
        else
            fclose(fp);
        
        std::cout<<"Reading  " << file_name << "  ..."<<std::endl;
        typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
        ImageType3DReaderType::Pointer imager= ImageType3DReaderType::New();
        imager->SetFileName(file_name);
        imager->Update();
        ImageType3D::Pointer image3D= imager->GetOutput();
        
        ImageType3D::SizeType imsize= image3D->GetLargestPossibleRegion().GetSize();
        
        if(N_images==0)
        {
            
            average_image= ImageType3D::New();
            average_image->SetRegions(image3D->GetLargestPossibleRegion());
            average_image->Allocate();
            average_image->SetSpacing(image3D->GetSpacing());
            average_image->SetOrigin(image3D->GetOrigin());
            average_image->SetDirection(image3D->GetDirection());
            average_image->FillBuffer(0);                    
        }
        
        
        
        for(int k=0;k<imsize[2];k++)
        {
            ImageType3D::IndexType index;
            index[2]=k;

            for(int j=0;j<imsize[1];j++)
            {
                index[1]=j;

                for(int i=0;i<imsize[0];i++)
                {
                    index[0]=i;

                    RealType val= image3D->GetPixel(index);
                    RealType val2= average_image->GetPixel(index);
                    average_image->SetPixel(index,val+val2);                                         
                }
            }
            
        }
        
        N_images++;        
    }
    inFile.close();
    
     typedef itk::ImageRegionIteratorWithIndex<ImageType3D>  ItType;
        ItType it(average_image,average_image->GetLargestPossibleRegion());
        it.GoToBegin();    
        while( !it.IsAtEnd() )
        {      
            
            RealType val = it.Get();
            
            it.Set(1.0*val/N_images);
            ++it;
        }
    
           
    
    
    std::string output_name(argv[2]);    
    mypos= nm.rfind(".nii");
    if(mypos ==-1)
        output_name= output_name + string(".nii");  
    std::string new_name= currdir + output_name;
    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(average_image);
    writer->Update();

    
    return EXIT_SUCCESS;
}
