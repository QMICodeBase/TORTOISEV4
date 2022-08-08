

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
#include "itkResampleImageFilter.h"

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

    typedef double RealType;        

    typedef itk::Transform<RealType,3,3> TransformType;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::Image<RealType,4> ImageType4D;
    typedef itk::AffineTransform<RealType,3> AffineTransformType;
    typedef itk::Image<itk::Vector<RealType,3>,  3>          DisplacementFieldType;
    typedef itk::DisplacementFieldTransform<RealType, 3>    DisplacementFieldTransformType;
    
           

    

int main( int argc , char * argv[] )
{
    if(argc<4)    
    {
        std::cout<<"Usage:   ApplyTransformationToScalar2   full_path_to_scalar_to_be_transformed  full_path_to_transformation  full_path_to_name_of_output full_path_to_image_with_desired_dimensions InterpolantType (NN, Lin, BSP)"<<std::endl;
        return EXIT_FAILURE;
    }
    
    
    typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
    ImageType3DReaderType::Pointer fixedr= ImageType3DReaderType::New();
    fixedr->SetFileName(argv[1] );
    fixedr->Update();
    ImageType3D::Pointer image3D= fixedr->GetOutput();
    

    ImageType4D::Pointer target_image4D=nullptr;
    ImageType3D::Pointer target_image=nullptr;
    
    if(argc>4)
    {
        typedef itk::ImageFileReader<ImageType4D> ImageType4DReaderType;
        ImageType4DReaderType::Pointer tr= ImageType4DReaderType::New();
        tr->SetFileName(argv[4] );
        tr->Update();
        target_image4D= tr->GetOutput();
        
        typedef itk::ExtractImageFilter< ImageType4D, ImageType3D > FilterType;
        ImageType4D::IndexType desiredStart; desiredStart.Fill(0);
        desiredStart[3]=0;
        
        ImageType4D::SizeType desiredSize=target_image4D->GetLargestPossibleRegion().GetSize();
        desiredSize[3]=0;
        
        ImageType4D::RegionType desiredRegion(desiredStart, desiredSize);
        
        FilterType::Pointer filter = FilterType::New();
        filter->SetExtractionRegion(desiredRegion);
        filter->SetInput(target_image4D);    
        filter->SetDirectionCollapseToSubmatrix(); // This is required.
        filter->Update();
        target_image=filter->GetOutput();
    }
    else
    {
        target_image=image3D;
    }
            
    int interp_type=1;
    if(std::string(argv[5])=="NN")
        interp_type=0;
    if(std::string(argv[5])=="BSP")
        interp_type=2;

    

    
    typedef itk::ResampleImageFilter<ImageType3D, ImageType3D, RealType> ScalarResamplerType;
    typename ScalarResamplerType::Pointer movingResampler = ScalarResamplerType::New();
    
    std::string filename(argv[2]);
    std::string::size_type idx=filename.rfind('.');
    std::string extension = filename.substr(idx+1);

    DisplacementFieldTransformType::Pointer disp_trans=nullptr;
    AffineTransformType::Pointer affine_trans=nullptr;

    if(idx != std::string::npos)
    {

        if(extension == string("nii"))
        {
            typedef itk::ImageFileReader<DisplacementFieldType> FieldReaderType;
            typename FieldReaderType::Pointer mreader=FieldReaderType::New();
            mreader->SetFileName(filename);
            mreader->Update();        
            disp_trans= DisplacementFieldTransformType::New();
            disp_trans->SetDisplacementField(mreader->GetOutput());
            movingResampler->SetTransform(disp_trans); 
        }
        else
        {
            using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
            TransformReaderType::Pointer reader = TransformReaderType::New();
            reader->SetFileName(filename );
            reader->Update();
            const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
            itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin(); 
            affine_trans = static_cast<AffineTransformType*>((*it).GetPointer());
            movingResampler->SetTransform(affine_trans);          
        }
    }
    
    ImageType3D::Pointer transformed_image3D = ImageType3D::New();
    transformed_image3D->SetRegions( target_image->GetLargestPossibleRegion());
    transformed_image3D->Allocate();
    transformed_image3D->SetSpacing( target_image->GetSpacing());
    transformed_image3D->SetOrigin( target_image->GetOrigin());
    transformed_image3D->SetDirection( target_image->GetDirection());
    transformed_image3D->FillBuffer(0);

    typedef itk::NearestNeighborInterpolateImageFunction<ImageType3D> NNInterpolatorType;
    NNInterpolatorType::Pointer interpNN= NNInterpolatorType::New();
    interpNN->SetInputImage(image3D);

    using LinInterpolatorType = itk::LinearInterpolateImageFunction<ImageType3D>;
    LinInterpolatorType::Pointer interpLIN=LinInterpolatorType::New();
    interpLIN->SetInputImage(image3D);

    using BSPInterpolatorType = itk::BSplineInterpolateImageFunction<ImageType3D,double> ;
    BSPInterpolatorType::Pointer interpBSP = BSPInterpolatorType::New();
    interpBSP->SetInputImage(image3D);
    interpBSP->SetSplineOrder(3);




    ImageType3D::SizeType sz = transformed_image3D->GetLargestPossibleRegion().GetSize();

#pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;

                ImageType3D::PointType pt,pt_trans;
                transformed_image3D->TransformIndexToPhysicalPoint(ind3,pt);

                if(extension == string("nii"))
                {
                    pt_trans= disp_trans->TransformPoint(pt);
                }
                else
                {
                    pt_trans= affine_trans->TransformPoint(pt);

                }


                if(interpNN->IsInsideBuffer(pt_trans))
                {
                    float val;
                    if(interp_type==0)
                        val = interpNN->Evaluate(pt_trans);
                    if(interp_type==1)
                        val = interpLIN->Evaluate(pt_trans);
                    if(interp_type==2)
                        val = interpBSP->Evaluate(pt_trans);

                    transformed_image3D->SetPixel(ind3,val);
                }



            }
        }
    }


    
    std::string currdir;
    std::string nm(argv[1]);
    
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    string name = currdir + std::string(argv[3]);
    writer->SetFileName(std::string(argv[3]));
    writer->SetInput(transformed_image3D);
    writer->Update();
    
    return EXIT_SUCCESS;
}

