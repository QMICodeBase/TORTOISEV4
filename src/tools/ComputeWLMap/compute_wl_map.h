#ifndef COMPUTEWLMAP_H
#define COMPUTEWLMAP_H

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
    



ImageType3D::Pointer compute_wl(ImageType4D::Pointer image4D)
{

    
    ImageType4D::SizeType imsize= image4D->GetLargestPossibleRegion().GetSize();
    ImageType4D::IndexType index;
    
    
    ImageType3D::SizeType nsize;
    nsize[0]=imsize[0];
    nsize[1]=imsize[1];
    nsize[2]=imsize[2];
    
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,nsize);
    ImageType3D::PointType norig;
    norig[0]=image4D->GetOrigin()[0];
    norig[1]=image4D->GetOrigin()[1];
    norig[2]=image4D->GetOrigin()[2];
    
    ImageType3D::SpacingType nspc;
    nspc[0]=image4D->GetSpacing()[0];
    nspc[1]=image4D->GetSpacing()[1];
    nspc[2]=image4D->GetSpacing()[2];
    
    ImageType3D::DirectionType ndir;
    ImageType4D::DirectionType dir = image4D->GetDirection();
    ndir(0,0)=dir(0,0);ndir(0,1)=dir(0,1);ndir(0,2)=dir(0,2);
    ndir(1,0)=dir(1,0);ndir(1,1)=dir(1,1);ndir(1,2)=dir(1,2);
    ndir(2,0)=dir(2,0);ndir(2,1)=dir(2,1);ndir(2,2)=dir(2,2);          
    
    
    


    ImageType3D::Pointer scalar_image = ImageType3D::New();
    scalar_image->SetRegions(reg);
    scalar_image->Allocate();
    scalar_image->SetOrigin(norig);
    scalar_image->SetSpacing(nspc);
    scalar_image->SetDirection(ndir);
    scalar_image->FillBuffer(0.);

    ImageType3D::IndexType ind;
    
    for(int k=0;k<imsize[2];k++)
    {
        index[2]=k;        
        ind[2]=k;
        for(int j=0;j<imsize[1];j++)
        {
            index[1]=j;            
            ind[1]=j;
            for(int i=0;i<imsize[0];i++)
            {
                index[0]=i;    
                ind[0]=i;
                
                InternalMatrixType curr_tens;
                
                index[3]=0;
                curr_tens(0,0)=image4D->GetPixel(index);
                index[3]=1;
                curr_tens(1,1)=image4D->GetPixel(index);
                index[3]=2;
                curr_tens(2,2)=image4D->GetPixel(index);
                index[3]=3;
                curr_tens(0,1)=image4D->GetPixel(index);
                curr_tens(1,0)=image4D->GetPixel(index);
                index[3]=4;
                curr_tens(0,2)=image4D->GetPixel(index);
                curr_tens(2,0)=image4D->GetPixel(index);
                index[3]=5;
                curr_tens(1,2)=image4D->GetPixel(index);
                curr_tens(2,1)=image4D->GetPixel(index);
                
                if( (curr_tens(0,0)!=curr_tens(0,0)) ||
                    (curr_tens(0,1)!=curr_tens(0,1)) ||
                    (curr_tens(0,2)!=curr_tens(0,2)) ||
                    (curr_tens(1,1)!=curr_tens(1,1)) ||
                    (curr_tens(1,2)!=curr_tens(1,2)) ||
                    (curr_tens(2,2)!=curr_tens(2,2)))
                    continue;

                if(curr_tens(0,0)+curr_tens(1,1)+curr_tens(2,2)  <1)
                    continue;

                vnl_symmetric_eigensystem<double>  eig(curr_tens);



                double wl = (eig.D(2,2) - eig.D(1,1))/eig.D(2,2);


                scalar_image->SetPixel(ind,wl);
            }   
        }   
    }
    
       
    return scalar_image;


}




#endif
