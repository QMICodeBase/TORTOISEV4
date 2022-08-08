

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
    typedef itk::Image<RealType,4> ImageType4D;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::DiffusionTensor3D<RealType> TensorPixelType;
    typedef itk::Vector<RealType,6> VectorPixelType;
    
    typedef itk::Image<itk::DiffusionTensor3D<RealType>,3>         TensorImageType;
    typedef itk::Image<VectorPixelType,3>         VectorImageType;
    
    typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;
           



int main( int argc , char * argv[] )
{
    if(argc<4)
    {
        std::cout<<"Usage:   StdevImages3D   full_path_to_textfile_containing_list_of_images   mean_image output_image "<<std::endl;
        return 0;
    }
    
    ifstream inFile(argv[1]);
    if (!inFile) 
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return 0;
    }
    
    
    int N_images=0;
    ImageType3D::Pointer average_image=nullptr;
    ImageType3D::Pointer std_image=nullptr;
    
    std::string currdir;
    std::string nm(argv[1]);
    
   
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);

        
    ImageType3D::SizeType imsize;

    typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;

    ImageType3DReaderType::Pointer imager2= ImageType3DReaderType::New();
    imager2->SetFileName(argv[2]);
    imager2->Update();
    average_image= imager2->GetOutput();


    std_image= ImageType3D::New();
    std_image->SetRegions(average_image->GetLargestPossibleRegion());
    std_image->Allocate();
    std_image->SetSpacing(average_image->GetSpacing());
    std_image->SetOrigin(average_image->GetOrigin());
    std_image->SetDirection(average_image->GetDirection());
    std_image->FillBuffer(0);


    imsize=std_image->GetLargestPossibleRegion().GetSize();


    
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
                return 0;
            }
            else
               fclose(fp2);
        }
        else
            fclose(fp);
        


        std::cout<<"Reading  " << file_name << "  ..."<<std::endl;

        ImageType3DReaderType::Pointer imager= ImageType3DReaderType::New();
        imager->SetFileName(file_name);
        imager->Update();
        ImageType3D::Pointer image3D= imager->GetOutput();
        
               
        
        
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

                    double diff = image3D->GetPixel(index) - average_image->GetPixel(index);
                    std_image->SetPixel(index,std_image->GetPixel(index)+diff*diff);


                     
                }
            }            
        }
        
        N_images++;        
    }
    inFile.close();
    




        typedef itk::ImageRegionIteratorWithIndex<ImageType3D>  ItType;
        ItType it(std_image,std_image->GetLargestPossibleRegion());
        it.GoToBegin();    
        while( !it.IsAtEnd() )
        {                  
            RealType val = it.Get();
            
            it.Set( sqrt(val/(N_images-1)));
            ++it;
        }
  
    
    
    
    std::string output_name(argv[3]);
    mypos= nm.rfind(".nii");
    if(mypos ==-1)
        output_name= output_name + string(".nii");  
    std::string new_name= currdir + output_name;
    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(std_image);
    writer->Update();

    
    return EXIT_SUCCESS;
}
