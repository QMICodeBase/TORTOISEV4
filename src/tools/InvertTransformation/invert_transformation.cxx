

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
#include "itkDisplacementFieldTransform.h"
#include "vnl/vnl_cross.h"
#include "itkInvertDisplacementFieldImageFilter.h"


    typedef double RealType;        

    typedef itk::Transform<RealType,3,3> TransformType;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::AffineTransform<RealType,3> AffineTransformType;
    typedef itk::Image<itk::Vector<RealType,3>,  3>          DisplacementFieldType;
    typedef itk::DisplacementFieldTransform<RealType, 3>    DisplacementFieldTransformType;
    typedef itk::Image<RealType,3> ImageType3D;
    
           

    
    
DisplacementFieldType::Pointer InvertDisplacementField( const DisplacementFieldType * field)
{   
    typedef itk::InvertDisplacementFieldImageFilter<DisplacementFieldType> InverterType;

  InverterType::Pointer inverter = InverterType::New();
  inverter->SetInput( field );
  inverter->SetMaximumNumberOfIterations( 200 );
  inverter->SetMeanErrorToleranceThreshold( 0.0003 );
  inverter->SetMaxErrorToleranceThreshold( 0.03 );
  inverter->Update();

  DisplacementFieldType::Pointer inverseField = inverter->GetOutput();

  return inverseField;
}


int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<<"Usage:   InvertTransformation   full_path_to_transformation_file "<<std::endl;
        return 0;
    }

    std::string currdir;
    std::string nm(argv[1]);
    
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    
    
    
    
    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    if(idx != std::string::npos)
    {
        std::string extension = filename.substr(idx+1);
        std::string basename= filename.substr(mypos+1,idx-mypos-1);
        if(extension == string("nii"))
        {
            typedef itk::ImageFileReader<DisplacementFieldType> FieldReaderType;
            typename FieldReaderType::Pointer mreader=FieldReaderType::New();
            mreader->SetFileName(filename);
            mreader->Update();                 
            DisplacementFieldType::Pointer disp_field=mreader->GetOutput();
            
            
            DisplacementFieldType::Pointer disp_field_inv= InvertDisplacementField(disp_field);
            
            std::string output_name;
            int minv_pos=basename.rfind("_MINV.nii");
            if(minv_pos!=-1)
            {
                std::string base_basename= basename.substr(0,minv_pos);
                output_name=currdir + base_basename + std::string("_FINV.nii");
            }
            else
            {
                minv_pos=basename.rfind("_FINV.nii");
                if(minv_pos!=-1)
                {
                    std::string base_basename= basename.substr(0,minv_pos);
                    output_name=currdir + base_basename + std::string("_MINV.nii");
                }
                else
                {
                   output_name=currdir + basename + std::string("_inv.nii");
                }
            }            
            
            typedef itk::ImageFileWriter<DisplacementFieldType> WriterType;
            WriterType::Pointer writer= WriterType::New();
            writer->SetFileName(output_name);
            writer->SetInput(disp_field_inv);
            writer->Update();
        }
        else
        {
            using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
            TransformReaderType::Pointer reader = TransformReaderType::New();
            reader->SetFileName(filename );
            const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
            itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin(); 
            AffineTransformType::Pointer affine_trans = static_cast<AffineTransformType*>((*it).GetPointer());
            
            AffineTransformType::Pointer affine_trans_inv= AffineTransformType::New();
            affine_trans->GetInverse(affine_trans_inv);
            
            std::string basename= filename.substr(mypos+1,idx-mypos-1);
            std::string output_name=currdir + basename + std::string("_inv.txt");
            
            itk::TransformFileWriter::Pointer trwriter = itk::TransformFileWriter::New();
            trwriter->SetInput(affine_trans_inv);
            trwriter->SetFileName(output_name);
            trwriter->Update();
        }
    }
    
   
    

    
    

    
    return 1;
}
