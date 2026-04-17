// compute the average of a list of affine transform

//#include "antsUtilities.h"
#include "itkImageFileReader.h"

#include "itkImageFileWriter.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkTransformFactory.h"

#include "itkAverageAffineTransformFunction.h"

#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;




typedef double RealType;     
typedef itk::Image<RealType,4> ImageType4D;

typedef itk::ImageFileReader<ImageType4D> ReaderType;

typedef std::string string;


void AverageAffineTransform( std::vector< string> affine_names, std::vector< string>  moving_names, string outputname, ImageType4D::Pointer fixed_image)
{
   
   
    typedef itk::MatrixOffsetTransformBase<double, 3,      3> AffineTransformType;
    typedef itk::AverageAffineTransformFunction<AffineTransformType> WarperType;
    itk::TransformFactory<AffineTransformType>::RegisterTransform();


    WarperType average_func;
    typedef itk::TransformFileReader TranReaderType;
  
  
    ImageType4D::SizeType fixed_size = fixed_image->GetLargestPossibleRegion().GetSize();
    ImageType4D::IndexType fixed_center;
    ImageType4D::PointType fixed_center_point;
    fixed_center[0]= (fixed_size[0]-1)/2;
    fixed_center[1]= (fixed_size[1]-1)/2;
    fixed_center[2]= (fixed_size[2]-1)/2;
    fixed_image->TransformIndexToPhysicalPoint(fixed_center,fixed_center_point);
    
    
  
    for(int i=0;i<affine_names.size();i++)
    {
        typename TranReaderType::Pointer tran_reader =          TranReaderType::New();
        tran_reader->SetFileName(affine_names[i]);
        tran_reader->Update();
        typename AffineTransformType::Pointer aff =          dynamic_cast<AffineTransformType *>( (tran_reader->GetTransformList() )->front().GetPointer() );
        
        ReaderType::Pointer reader= ReaderType::New();
        reader->SetFileName(moving_names[i]);
        reader->Update();    
        ImageType4D::Pointer moving_image= reader->GetOutput();
        
        ImageType4D::SizeType moving_size = moving_image->GetLargestPossibleRegion().GetSize();
        ImageType4D::IndexType moving_center;
        ImageType4D::PointType moving_center_point;
        moving_center[0]= (moving_size[0]-1)/2;
        moving_center[1]= (moving_size[1]-1)/2;
        moving_center[2]= (moving_size[2]-1)/2;
        moving_image->TransformIndexToPhysicalPoint(moving_center,moving_center_point);
        
        AffineTransformType::OutputVectorType orig_diff;
        orig_diff[0]= moving_center_point[0] - fixed_center_point[0];
        orig_diff[1]= moving_center_point[1] - fixed_center_point[1];
        orig_diff[2]= moving_center_point[2] - fixed_center_point[2];
        
        AffineTransformType::OutputVectorType total_trans= aff->GetOffset();
        AffineTransformType::OutputVectorType new_trans= total_trans- orig_diff;
        
        aff->SetOffset(new_trans);
        
        average_func.PushBackAffineTransform(aff, 1);
        
    }
  


  AffineTransformType::Pointer aff_ref_tmp = average_func.GetTransformList().begin()->aff;
  WarperType::PointType aff_center = aff_ref_tmp->GetCenter();
  typename AffineTransformType::Pointer aff_output = AffineTransformType::New();

  
  average_func.AverageMultipleAffineTransform(aff_center, aff_output);

  typedef itk::TransformFileWriter TranWriterType;
  typename TranWriterType::Pointer tran_writer = TranWriterType::New();
  tran_writer->SetFileName(outputname);
  tran_writer->SetInput(aff_output);
  tran_writer->Update();

}




int main(int argc, char *argv[])
{
    
    if(argc<5)    
    {
        std::cout<<"Usage:   AverageAffineTransformations2   affine_trans_list  image_list target_image outputname "<<std::endl;
        return EXIT_FAILURE;
    }
    
    std::string outputname(argv[4]);
      
    std::string currdir;
    std::string nm(argv[1]);                
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    
    
    
    ifstream inFile(argv[1]);
    if (!inFile) 
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return EXIT_FAILURE;
    }
    

    std::vector<std::string> affine_names;

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
        
        affine_names.push_back(file_name);
    }
    inFile.close();
    

    
    fstream inFile2(argv[2]);
    if (!inFile2) 
    {
        cerr << "File " << argv[2] << " not found." << endl;
        return EXIT_FAILURE;
    }
    

    std::vector<std::string> moving_names;
    while (getline(inFile2, line)) 
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
        
        moving_names.push_back(file_name);
    }
    inFile2.close();

    
    
    ReaderType::Pointer fixed_reader= ReaderType::New();
    fixed_reader->SetFileName(argv[3]);
    fixed_reader->Update();    
    ImageType4D::Pointer fixed_image= fixed_reader->GetOutput();
        
    
 
    
    AverageAffineTransform(affine_names,moving_names,outputname, fixed_image);
    return EXIT_SUCCESS;
}
