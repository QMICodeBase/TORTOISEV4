#ifndef _COMPUTEMETRICSCC_H
#define _COMPUTEMETRICSCC_H

#include "defines.h"
#include "drbuddi_image_utilities.h"

#define WIN_RADCC 4
#define WIN_RADCC_Z 3
#define LIMCC (1E-10)

float  ComputeUpdateCC(ImageType3D::IndexType index,  
                          ImageType3D::Pointer up_img, ImageType3D::Pointer down_img,                         
                          DisplacementFieldType::PixelType &updateF,DisplacementFieldType::PixelType &updateM)                          
{
    updateF.Fill(0);
    updateM.Fill(0);
   
    ImageType3D::SizeType d_sz= up_img->GetLargestPossibleRegion().GetSize();
   

    ImageType3D::IndexType start,end;

    start[2]=index[2]-WIN_RADCC_Z;
    if(start[2]<0)
        start[2]=0;
    start[1]=index[1]-WIN_RADCC;
    if(start[1]<0)
        start[1]=0;
    start[0]=index[0]-WIN_RADCC;
    if(start[0]<0)
        start[0]=0;

    end[2]=index[2]+WIN_RADCC_Z+1;
    if(end[2]>d_sz[2])
        end[2]=d_sz[2];
    end[1]=index[1]+WIN_RADCC+1;
    if(end[1]>d_sz[1])
        end[1]=d_sz[1];
    end[0]=index[0]+WIN_RADCC+1;
    if(end[0]>d_sz[0])
        end[0]=d_sz[0];
        
    double suma2 = 0.0;
    double suma = 0.0;
    double  sumac=0;
    double sumc2 = 0.0;
    double sumc = 0.0;
    int N=0;

    float valF_center=up_img->GetPixel(index);
    float valM_center=down_img->GetPixel(index);;
   
    DisplacementFieldType::PixelType gradI=  ComputeImageGradient( up_img, index);
    DisplacementFieldType::PixelType gradJ=  ComputeImageGradient( down_img, index);
    
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

                float f= up_img->GetPixel(cind);
                float m= down_img->GetPixel(cind);

                suma2 += f * f;
                suma += f;
                sumc2 += m * m;
                sumc += m;
                sumac += f*m;

                N++;
            }
        }
    }

    double Fmean = suma/N;
    double Mmean= sumc/N;

    double valF = valF_center -Fmean;
    double valM = valM_center -Mmean;

    double sFF = suma2 - Fmean*suma;
    double sMM = sumc2 - Mmean*sumc;
    double sFM = sumac - Fmean*sumc;

    double sFF_sMM = sFF * sMM;

    float val=0;
    if(fabs(sFF_sMM) >LIMCC && fabs(sMM) > LIMCC)
    {
        val= -sFM*sFM/ sFF_sMM;

        double first_termF= -2* sFM/sFF_sMM *    (valM - sFM/sFF * valF) ;
        double first_termM= -2* sFM/sFF_sMM *    (valF - sFM/sMM * valM) ;

        updateF[0] = first_termF * gradI[0];
        updateF[1] = first_termF * gradI[1];
        updateF[2] = first_termF * gradI[2];
        updateM[0] = first_termM * gradJ[0];
        updateM[1] = first_termM * gradJ[1];
        updateM[2] = first_termM * gradJ[2];
    }


     return val;
}


float ComputeMetric_CC(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, 
                          DisplacementFieldType::Pointer &updateFieldF, DisplacementFieldType::Pointer &updateFieldM)
{
    updateFieldF = DisplacementFieldType::New();
    updateFieldF->SetRegions(up_img->GetLargestPossibleRegion());
    updateFieldF->SetDirection(up_img->GetDirection());
    updateFieldF->SetOrigin(up_img->GetOrigin());
    updateFieldF->SetSpacing(up_img->GetSpacing());
    updateFieldF->Allocate();
    DisplacementFieldType::PixelType zero; zero.Fill(0);
    updateFieldF->FillBuffer(zero);

    updateFieldM = DisplacementFieldType::New();
    updateFieldM->SetRegions(up_img->GetLargestPossibleRegion());
    updateFieldM->SetDirection(up_img->GetDirection());
    updateFieldM->SetOrigin(up_img->GetOrigin());
    updateFieldM->SetSpacing(up_img->GetSpacing());
    updateFieldM->Allocate();
    updateFieldM->FillBuffer(zero);
          
    ImageType3D::SizeType imsize=up_img->GetLargestPossibleRegion().GetSize();

    ImageType3D::Pointer m_MetricImage= ImageType3D::New();
    m_MetricImage->SetRegions(up_img->GetLargestPossibleRegion());
    m_MetricImage->Allocate();
    m_MetricImage->SetSpacing(up_img->GetSpacing());
    m_MetricImage->SetOrigin(up_img->GetOrigin());
    m_MetricImage->SetDirection(up_img->GetDirection());
    m_MetricImage->FillBuffer(0);
                   
    
    #pragma omp parallel for
    for( int k=0; k<(int)imsize[2];k++)
    {
        ImageType3D::IndexType index;
        index[2]=k;

        DisplacementFieldType::PixelType updateF,updateM;

        for(unsigned int j=0; j<imsize[1];j++)
        {
            index[1]=j;
            for(unsigned int i=0; i<imsize[0];i++)
            {
                index[0]=i;
                
                float mv= ComputeUpdateCC(index,up_img,down_img,
                                             updateF,updateM);
                m_MetricImage->SetPixel(index,mv);
                updateFieldF->SetPixel(index,updateF);
                updateFieldM->SetPixel(index,updateM);                
            }
        }
    }

  double value = 0;
  typedef itk::ImageRegionIterator<ImageType3D>  ItType;
  ItType it(m_MetricImage,m_MetricImage->GetRequestedRegion());
  it.GoToBegin();
  while( !it.IsAtEnd() )
  {
      value+= it.Get();
      ++it;
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
