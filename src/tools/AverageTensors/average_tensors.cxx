

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
    if(argc<3)    
    {
        std::cout<<"Usage:   AverageTensors   full_path_to_textfile_containing_list_of_tensors    name_of_output_image "<<std::endl;
        return EXIT_FAILURE;
    }
    
    ifstream inFile(argv[1]);
    if (!inFile) 
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return EXIT_FAILURE;
    }
    
    
    int N_images=0;
    ImageType4D::Pointer average_image=nullptr;
    ImageType3D::Pointer counter_image=nullptr;
    
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
                return 0;
            }
            else
               fclose(fp2);
        }
        else
            fclose(fp);
        
        std::cout<<"Reading  " << file_name << "  ..."<<std::endl;
        typedef itk::ImageFileReader<ImageType4D> ImageType4DReaderType;
        ImageType4DReaderType::Pointer imager= ImageType4DReaderType::New();
        imager->SetFileName(file_name);
        imager->Update();
        ImageType4D::Pointer image4D= imager->GetOutput();
        
        
        
        if(N_images==0)
        {
            
            average_image= ImageType4D::New();
            average_image->SetRegions(image4D->GetLargestPossibleRegion());
            average_image->Allocate();
            average_image->SetSpacing(image4D->GetSpacing());
            average_image->SetOrigin(image4D->GetOrigin());
            average_image->SetDirection(image4D->GetDirection());
            average_image->FillBuffer(0);
            
            typedef itk::ExtractImageFilter< ImageType4D, ImageType3D > FilterType;
            ImageType4D::IndexType desiredStart; desiredStart.Fill(0);
            desiredStart[3]=0;        
            ImageType4D::SizeType desiredSize=image4D->GetLargestPossibleRegion().GetSize();
            desiredSize[3]=0;
            ImageType4D::RegionType desiredRegion(desiredStart, desiredSize);
        
            FilterType::Pointer filter = FilterType::New();
            filter->SetExtractionRegion(desiredRegion);
            filter->SetInput( image4D);    
            filter->SetDirectionCollapseToSubmatrix(); // This is required.
            filter->Update();     
            
            counter_image= ImageType3D::New();
            counter_image->SetRegions(filter->GetOutput()->GetLargestPossibleRegion());
            counter_image->Allocate();
            counter_image->SetSpacing(filter->GetOutput()->GetSpacing());
            counter_image->SetOrigin(filter->GetOutput()->GetOrigin());
            counter_image->SetDirection(filter->GetOutput()->GetDirection());
            counter_image->FillBuffer(0);  
            
            imsize=filter->GetOutput()->GetLargestPossibleRegion().GetSize();
        }
        
        
        
        for(int k=0;k<imsize[2];k++)
        {
            ImageType4D::IndexType index4D;
            ImageType3D::IndexType index;
            index[2]=k;
            index4D[2]=k;
            
            for(int j=0;j<imsize[1];j++)
            {
                index[1]=j;
                index4D[1]=j;
                for(int i=0;i<imsize[0];i++)
                {
                    index[0]=i;
                    index4D[0]=i;
                    
                    index4D[3]=0;
                    RealType Dxx= image4D->GetPixel(index4D);
                    index4D[3]=1;
                    RealType Dyy= image4D->GetPixel(index4D);
                    index4D[3]=2;
                    RealType Dzz= image4D->GetPixel(index4D);
                    
                    RealType trace= Dxx+Dyy+Dzz;
                    if(trace>1)
                    {
                        RealType count=counter_image->GetPixel(index);
                        count++;
                        counter_image->SetPixel(index,count);
                        
                        for(int ma=0;ma<6;ma++)
                        {
                            index4D[3]=ma;
                            RealType val= image4D->GetPixel(index4D);
                            RealType av= average_image->GetPixel(index4D);
                            average_image->SetPixel(index4D,av+val);
                        }
                    }   
                }
            }
            
        }
        
        N_images++;        
    }
    inFile.close();
    
    
         typedef itk::ImageRegionIteratorWithIndex<ImageType4D>  ItType;
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
    
    
    typedef itk::ImageFileWriter<ImageType4D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(average_image);
    writer->Update();

    
    return EXIT_SUCCESS;
}
