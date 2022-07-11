#ifndef _COMPUTEFAMAP_H
#define _COMPUTEFAMAP_H

using namespace std;

#include "defines.h"
#include "../utilities/math_utilities.h"
#include "itkImageDuplicator.h"


    
ImageType3D::Pointer compute_fa_map(DTImageType::Pointer dt_img,bool filter=false)
{
    ImageType3D::Pointer FAimage = ImageType3D::New();
    FAimage->SetRegions(dt_img->GetLargestPossibleRegion());
    FAimage->Allocate();
    FAimage->SetOrigin(dt_img->GetOrigin());
    FAimage->SetSpacing(dt_img->GetSpacing());
    FAimage->SetDirection(dt_img->GetDirection());
    FAimage->FillBuffer(0.);

    ImageType3D::Pointer outlier_image = ImageType3D::New();
    outlier_image->SetRegions(dt_img->GetLargestPossibleRegion());
    outlier_image->Allocate();
    outlier_image->SetOrigin(dt_img->GetOrigin());
    outlier_image->SetSpacing(dt_img->GetSpacing());
    outlier_image->SetDirection(dt_img->GetDirection());
    outlier_image->FillBuffer(0.);


    itk::ImageRegionIteratorWithIndex<ImageType3D> it(FAimage, FAimage->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        DTImageType::PixelType vec = dt_img->GetPixel(ind3) *1E6;
        InternalMatrixType curr_tens;

        curr_tens(0,0)= vec[0];
        curr_tens(0,1)= vec[1];
        curr_tens(0,2)= vec[2];
        curr_tens(1,0)= vec[1];
        curr_tens(1,1)= vec[3];
        curr_tens(1,2)= vec[4];
        curr_tens(2,0)= vec[2];
        curr_tens(2,1)= vec[4];
        curr_tens(2,2)= vec[5];


        if( (curr_tens(0,0)!=curr_tens(0,0)) ||
            (curr_tens(0,1)!=curr_tens(0,1)) ||
            (curr_tens(0,2)!=curr_tens(0,2)) ||
            (curr_tens(1,1)!=curr_tens(1,1)) ||
            (curr_tens(1,2)!=curr_tens(1,2)) ||
            (curr_tens(2,2)!=curr_tens(2,2)))
        {
            ++it;
            continue;
        }

        if(curr_tens(0,0)+curr_tens(1,1)+curr_tens(2,2)  <1)
        {
            ++it;
            continue;
        }

        vnl_symmetric_eigensystem<double>  eig(curr_tens);

        if(filter)
        {
            if( (eig.D(0,0)<0) && (eig.D(1,1)>=0) && (eig.D(2,2)>=0))
            {
                float val =(eig.D(0,0) + eig.D(1,1))/2;
                eig.D(0,0)=val;
                eig.D(1,1)=val;
                outlier_image->SetPixel(ind3,1);
            }
            else
            {
                if( (eig.D(0,0)<0) && (eig.D(1,1)<=0) && (eig.D(2,2)<=0))
                {
                    eig.D(0,0)=eig.D(1,1)=eig.D(2,2)=0;
                    outlier_image->SetPixel(ind3,1);
                }
                if( (eig.D(0,0)<0) && (eig.D(1,1)<0) && (eig.D(2,2)>=0))
                {
                    float val =(eig.D(0,0) + eig.D(1,1)+eig.D(2,2))/3;
                    if(val <0)
                    {
                        eig.D(0,0)=eig.D(1,1)=eig.D(2,2)=0;
                        outlier_image->SetPixel(ind3,1);
                    }
                    else
                    {
                        eig.D(0,0)=eig.D(1,1)=eig.D(2,2)=val;
                        outlier_image->SetPixel(ind3,1);
                    }
                }
            }
        }
        double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
        double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
        double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);

        double FA=0;
        if(denom!=0)
            FA= sqrt( 1.5*nom/denom);

        it.Set(FA);

        ++it;
    }

    if(filter)
    {
        typedef itk::ImageDuplicator<ImageType3D> DupType;
        DupType::Pointer dup = DupType::New();
        dup->SetInputImage(FAimage);
        dup->Update();
        ImageType3D::Pointer FAimage2= dup->GetOutput();
        ImageType3D::SizeType imsize = FAimage->GetLargestPossibleRegion().GetSize();

        for(int k=0;k<imsize[2];k++)
        {
            ImageType3D::IndexType ind;
            ind[2]=k;
            for(int j=0;j<imsize[1];j++)
            {
                ind[1]=j;
                for(int i=0;i<imsize[0];i++)
                {
                    ind[0]=i;

                    if(outlier_image->GetPixel(ind)!=0)
                    {
                        std::vector<float> signs;


                        ImageType3D::SizeType size;
                        ImageType3D::IndexType start;

                        start[2]= std::max(0,(int)ind[2]-1);
                        size[2]=   std::min((int)imsize[2]-1,(int)ind[2]+1)-start[2]+1;

                        start[1]= std::max(0,(int)ind[1]-1);
                        size[1]=   std::min((int)imsize[1]-1,(int)ind[1]+1)-start[1]+1;

                        start[0]= std::max(0,(int)ind[0]-1);
                        size[0]=   std::min((int)imsize[0]-1,(int)ind[0]+1)-start[0]+1;

                        ImageType3D::RegionType reg(start,size);

                        itk::ImageRegionIteratorWithIndex<ImageType3D> it(FAimage2,reg);
                        it.GoToBegin();
                        while(!it.IsAtEnd())
                        {
                            signs.push_back(it.Get());
                            ++it;
                        }

                        float val = median(signs);
                        FAimage->SetPixel(ind,val);
                    }
                }
            }
        }
    }

    return FAimage;
}


#endif
