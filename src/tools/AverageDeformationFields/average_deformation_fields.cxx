

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
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkAddImageFilter.h"
#include "itkShiftScaleImageFilter.h"

    typedef double RealType;        

    
    typedef itk::Image<itk::Vector<RealType,3>,  3>          DisplacementFieldType;
           

int main( int argc , char * argv[] )
{
    if(argc<3)    
    {
        std::cout<<"Usage:   AverageDeformationFields   full_path_to_textfile_containing_list_of_deformation_fiekds   full_path_name_of_output "<<std::endl;
        return EXIT_FAILURE;
    }
    
    ifstream inFile(argv[1]);
    if (!inFile) 
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return EXIT_FAILURE;
    }
    
    
    int N_images=0;
    
      
    std::string currdir;
    std::string nm(argv[1]);
    
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    

    DisplacementFieldType::Pointer avg_deffield=nullptr;
    
    
        
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
        
        
        typedef itk::ImageFileReader<DisplacementFieldType> ReaderType;
        ReaderType::Pointer reader= ReaderType::New();
        reader->SetFileName(file_name);
        reader->Update();
        DisplacementFieldType::Pointer curr_field= reader->GetOutput();
        
        if(N_images==0)
        {
            avg_deffield= DisplacementFieldType::New();
            avg_deffield->SetRegions(curr_field->GetLargestPossibleRegion());
            avg_deffield->Allocate();
            avg_deffield->SetOrigin(curr_field->GetOrigin());
            avg_deffield->SetSpacing(curr_field->GetSpacing());
            avg_deffield->SetDirection(curr_field->GetDirection());
            
            
            itk::Vector<RealType,3> zero;
            zero.Fill(0);
            avg_deffield->FillBuffer(zero);                        
        }
        

        typedef itk::AddImageFilter <DisplacementFieldType, DisplacementFieldType >    AddImageFilterType; 
        AddImageFilterType::Pointer addFilter   = AddImageFilterType::New ();
        addFilter->SetInput1(avg_deffield);
        addFilter->SetInput2(curr_field);        
        addFilter->Update();
        avg_deffield=addFilter->GetOutput();                
  
        N_images++;        
    }

    inFile.close();
    

   typedef itk::ImageRegionIteratorWithIndex<DisplacementFieldType>  ItType;
   ItType it(avg_deffield,avg_deffield->GetRequestedRegion());
   it.GoToBegin();    
   while( !it.IsAtEnd() )
   {
       it.Set(1.*it.Get()/N_images);
      ++it;
   }
  
   
  
  
   typedef itk::ImageFileWriter<DisplacementFieldType> WriterType;
   WriterType::Pointer writer= WriterType::New();
   writer->SetFileName(argv[2]);
   writer->SetInput(avg_deffield);
   writer->Update();
    
    
    
    
    
    
    
    


    
    return EXIT_SUCCESS;
}
