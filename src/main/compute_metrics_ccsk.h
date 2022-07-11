#ifndef _COMPUTEMETRICSCCSK_H
#define _COMPUTEMETRICSCCSK_H

#include "defines.h"

#define LIMCCSK (1E-5)
#define WIN_RAD 4
#define WIN_RAD_Z 2

float  ComputeUpdateCCSK(ImageType3D::IndexType index, ImageType3D::Pointer up_img, ImageType3D::Pointer down_img,
                          ImageType3D::Pointer K_img, ImageType3D::Pointer str_img,
                          DisplacementFieldType::PixelType &updateF,DisplacementFieldType::PixelType &updateM)                          
{
    updateF.Fill(0);
    updateM.Fill(0);
   
    ImageType3D::SizeType d_sz= up_img->GetLargestPossibleRegion().GetSize();
   

    ImageType3D::IndexType start,end;

    start[2]=index[2]-WIN_RAD_Z;
    if(start[2]<0)
        start[2]=0;
    start[1]=index[1]-WIN_RAD;
    if(start[1]<0)
        start[1]=0;
    start[0]=index[0]-WIN_RAD;
    if(start[0]<0)
        start[0]=0;

    end[2]=index[2]+WIN_RAD_Z+1;
    if(end[2]>d_sz[2])
        end[2]=d_sz[2];
    end[1]=index[1]+WIN_RAD+1;
    if(end[1]>d_sz[1])
        end[1]=d_sz[1];
    end[0]=index[0]+WIN_RAD+1;
    if(end[0]>d_sz[0])
        end[0]=d_sz[0];
        
    double suma2 = 0.0;
    double suma = 0.0;
    double  sumac=0;
    double sumc2 = 0.0;
    double sumc = 0.0;
    int N=0;

    float valK_center=K_img->GetPixel(index);
    float valS_center=str_img->GetPixel(index);;
       
    
    ImageType3D::IndexType cind;
    for(int z=start[2];z<end[2];z++)
    {
        cind[2]=z;
        for(int y=start[1];y<end[1];y++)
        {
            cind[1]=y;
            for(int x=start[0];x<end[0];x++)
            {
                cind[0]=x;

                float Kim = K_img->GetPixel(cind);
                float c = str_img->GetPixel(cind);


                suma2 += Kim * Kim;
                suma += Kim;
                sumc2 += c *c;
                sumc += c;
                sumac += Kim*c;

                N++;
            }
        }
    }

    double Kmean = suma/N;
    double Smean= sumc/N;

    float valK = valK_center-Kmean;
    float valS = valS_center -Smean;

    float sKK = suma2 - Kmean*suma;
    float sSS = sumc2 - Smean*sumc;
    float sKS = sumac - Kmean*sumc;


    float sSS_sKK = sSS * sKK;

    float val=0;
    if(fabs(sSS_sKK) > LIMCCSK && fabs(sKK) > LIMCCSK )
    {
        val= -sKS*sKS/ sSS_sKK;

        float first_term= -2*sKS/sSS_sKK *(valS - sKS/sKK*valK);
        float fval = up_img->GetPixel(index);
        float mval = down_img->GetPixel(index);

        float sm_mval_fval=(mval+fval);

        if(sm_mval_fval*sm_mval_fval > LIMCCSK)
        {
            {
                float grad_term =2* mval*mval/sm_mval_fval/sm_mval_fval;
                DisplacementFieldType::PixelType gradI2= ComputeImageGradient(up_img,index);

                updateF[0]= first_term*grad_term * gradI2[0];
                updateF[1]= first_term*grad_term * gradI2[1];
                updateF[2]= first_term*grad_term * gradI2[2];
            }
            {
                float grad_term= 2* fval*fval/sm_mval_fval/sm_mval_fval;
                DisplacementFieldType::PixelType gradJ2= ComputeImageGradient(down_img,index);

                updateM[0]= first_term*grad_term * gradJ2[0];
                updateM[1]= first_term*grad_term * gradJ2[1];
                updateM[2]= first_term*grad_term * gradJ2[2];
            }
        }
    }


     return val;
}



float ComputeMetric_CCSK(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, const ImageType3D::Pointer str_img,
                          DisplacementFieldType::Pointer &updateFieldF, DisplacementFieldType::Pointer &updateFieldM)
{

    updateFieldF = DisplacementFieldType::New();
    updateFieldF->SetRegions(up_img->GetLargestPossibleRegion());
    updateFieldF->SetDirection(up_img->GetDirection());
    updateFieldF->SetOrigin(up_img->GetOrigin());
    updateFieldF->SetSpacing(up_img->GetSpacing());
    updateFieldF->Allocate();


    updateFieldM = DisplacementFieldType::New();
    updateFieldM->SetRegions(up_img->GetLargestPossibleRegion());
    updateFieldM->SetDirection(up_img->GetDirection());
    updateFieldM->SetOrigin(up_img->GetOrigin());
    updateFieldM->SetSpacing(up_img->GetSpacing());
    updateFieldM->Allocate();


    ImageType3D::SizeType imsize=up_img->GetLargestPossibleRegion().GetSize();

    ImageType3D::Pointer m_MetricImage= ImageType3D::New();
    m_MetricImage->SetRegions(up_img->GetLargestPossibleRegion());
    m_MetricImage->Allocate();
    m_MetricImage->SetSpacing(up_img->GetSpacing());
    m_MetricImage->SetOrigin(up_img->GetOrigin());
    m_MetricImage->SetDirection(up_img->GetDirection());


    ImageType3D::Pointer KImage= ImageType3D::New();
    KImage->SetRegions(up_img->GetLargestPossibleRegion());
    KImage->Allocate();
    KImage->SetSpacing(up_img->GetSpacing());
    KImage->SetOrigin(up_img->GetOrigin());
    KImage->SetDirection(up_img->GetDirection());
    KImage->FillBuffer(0);
                   
    
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(KImage,KImage->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        float a= up_img->GetPixel(ind3);
        float b= down_img->GetPixel(ind3);
        float a_b= a+b;
        if(a_b>LIMCCSK)
            it.Set(2*a*b/a_b);

        ++it;
    }

    #pragma omp parallel for //collapse(2)
    for( int k=0; k<(int)imsize[2];k++)
    {
        for(unsigned int j=0; j<imsize[1];j++)
        {
            ImageType3D::IndexType index;
            index[2]=k;
            DisplacementFieldType::PixelType updateF,updateM;

            index[1]=j;
            for(unsigned int i=0; i<imsize[0];i++)
            {
                index[0]=i;
                
                float mv= ComputeUpdateCCSK(index, up_img, down_img,
                                              KImage, str_img,
                                              updateF,updateM);
                m_MetricImage->SetPixel(index,mv);
                updateFieldF->SetPixel(index,updateF);
                updateFieldM->SetPixel(index,updateM);                
            }
        }
    }

  double value = 0;
  typedef itk::ImageRegionIterator<ImageType3D>  ItType;
  ItType it2(m_MetricImage,m_MetricImage->GetRequestedRegion());
  it2.GoToBegin();
  while( !it2.IsAtEnd() )
  {
      value+= it2.Get();
      ++it2;
  }
  value=value/(imsize[0]*imsize[1]*imsize[2]);

  return value;

}

    


    
     




/*
float ComputeMetric_CCSK(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, 
                         const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV)
{


}
                         
                         

float ComputeMetric_CC(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                       const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV  )
{


}
*/



#endif
