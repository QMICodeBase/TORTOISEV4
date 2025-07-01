

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
#include "itkCompositeTransform.h"
#include "itkDisplacementFieldTransform.h"



    typedef double RealType;        
    typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;
    typedef itk::AffineTransform<RealType,3> AffineTransformType;
    
    typedef itk::Image<itk::Vector<RealType,3>,  3>          DisplacementFieldType;
    typedef itk::DisplacementFieldTransform<RealType, 3>    DisplacementFieldTransformType;
    
    typedef itk::CompositeTransform<RealType, 3>                CompositeTransformType;
    typedef itk::Transform<RealType,3, 3>             TransformType;
    typedef AffineTransformType::Superclass                                   MatrixOffsetTransformBaseType;
    
           
    
AffineTransformType::Pointer CollapseLinearTransforms( CompositeTransformType * compositeTransform )
{
    typename AffineTransformType::Pointer totalTransform = AffineTransformType::New();
    
  
    for( unsigned int n = 0; n < compositeTransform->GetNumberOfTransforms(); n++ )
    {
        TransformType::Pointer transform = compositeTransform->GetNthTransform( n );
        AffineTransformType::Pointer nthTransform = AffineTransformType::New();

        MatrixOffsetTransformBaseType::ConstPointer matrixOffsetTransform = dynamic_cast<MatrixOffsetTransformBaseType * const>( transform.GetPointer() );
        nthTransform->SetMatrix( matrixOffsetTransform->GetMatrix() );
        nthTransform->SetOffset( matrixOffsetTransform->GetOffset() );

        totalTransform->Compose( nthTransform, true );
    }
    return totalTransform;
}


    

int main( int argc , char * argv[] )
{
    if(argc<3)    
    {
        std::cout<<"Usage:   CombineTransformations   full_path_to_trans1 full_path_to_trans2 ..............   full_path_to_trans_n "<<std::endl;
        std::cout<<"The output name will be combined_displacement.nii in the first transformation's folder"<<std::endl;
        return 0;
    }
    
    
      
    std::string currdir;
    std::string nm(argv[1]);
    
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    
    
    CompositeTransformType::Pointer composite_trans= CompositeTransformType::New();
    DisplacementFieldType::Pointer ref_disp_field=nullptr;
    
    bool all_affine=1;    
    for(int i=1;i<argc;i++) 
    {
        std::string filename(argv[i]);
        std::string::size_type idx=filename.rfind('.');
        if(idx != std::string::npos)
        {
            std::string extension = filename.substr(idx+1);
            if(extension == string("nii"))
            {
                all_affine=0;   
                                      
                typedef itk::ImageFileReader<DisplacementFieldType> FieldReaderType;
                typename FieldReaderType::Pointer mreader=FieldReaderType::New();
                mreader->SetFileName(filename);
                mreader->Update();        
                DisplacementFieldTransformType::Pointer disp_trans= DisplacementFieldTransformType::New();
                disp_trans->SetDisplacementField(mreader->GetOutput());
                ref_disp_field=mreader->GetOutput();
                composite_trans->AddTransform(disp_trans);
            }
            else
            {
                if(extension == string("txt"))
                {
                    itk::TransformFileReader::Pointer reader;
                    reader = itk::TransformFileReader::New();
                    reader->SetFileName(filename);
                    reader->Update();
                    typedef itk::TransformFileReader::TransformListType * TransformListType;
                    TransformListType transforms = reader->GetTransformList();   
                    itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin(); 
                    AffineTransformType::Pointer affine_trans = static_cast<AffineTransformType*>((*it).GetPointer());
                    composite_trans->AddTransform(affine_trans);
                }
                else
                {
                    std::cout<<"Transformation " << i-1 << " does not contain .nii or .txt.. Exiting!!"<<std::endl;
                    exit(0);                                    
                }                
            }
        }
        else
        {
            std::cout<<"Transformation " << i-1 << " does not contain .nii or .txt.. Exiting!!"<<std::endl;
            exit(0);
        }        
    }

    composite_trans->FlattenTransformQueue();
    composite_trans->SetOnlyMostRecentTransformToOptimizeOn();
    
    if(all_affine)
    {
        AffineTransformType::Pointer combined_affine= CollapseLinearTransforms(composite_trans );
                
        string tr_name= currdir + string("combined_affine.txt");    
        itk::TransformFileWriter::Pointer trwriter = itk::TransformFileWriter::New();
        trwriter->SetInput(combined_affine);
        trwriter->SetFileName(tr_name);
        trwriter->Update();
    }
    else
    {
        DisplacementFieldType::Pointer combined_displacement_field= DisplacementFieldType::New();
        combined_displacement_field->SetRegions(ref_disp_field->GetLargestPossibleRegion());
        combined_displacement_field->Allocate();
        combined_displacement_field->SetSpacing(ref_disp_field->GetSpacing());
        combined_displacement_field->SetOrigin(ref_disp_field->GetOrigin());
        combined_displacement_field->SetDirection(ref_disp_field->GetDirection());  
                
        DisplacementFieldType::SizeType imsize= ref_disp_field->GetLargestPossibleRegion().GetSize();
        
    
        for( int k=0; k<(int)imsize[2];k++)
        {
            DisplacementFieldType::IndexType index;
            index[2]=k;
        
            for(unsigned int j=0; j<imsize[1];j++)
            {
                index[1]=j;
                for(unsigned int i=0; i<imsize[0];i++)
                {                     
                    index[0]=i;
                                                  
                    DisplacementFieldType::PointType pf;
                    combined_displacement_field->TransformIndexToPhysicalPoint(index,pf);
                    DisplacementFieldType::PointType pmt = composite_trans->TransformPoint(pf);
                    DisplacementFieldType::PixelType vec = pmt-pf;
                    combined_displacement_field->SetPixel(index,vec);
                }
            }
        }
        
        
        typedef itk::ImageFileWriter<DisplacementFieldType> WriterType;
        WriterType::Pointer writer= WriterType::New();
        string name =currdir + string("combined_displacement.nii");    
        writer->SetFileName(name);
        writer->SetInput(combined_displacement_field);
        writer->Update();
    }
}
