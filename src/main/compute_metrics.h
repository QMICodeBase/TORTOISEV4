#ifndef _COMPUTEMETRICSMSJAC_H
#define _COMPUTEMETRICSMSJAC_H

#include "defines.h"

    using GaussianSmoothingOperatorType = itk::GaussianOperator<double, 3>;





double mf(double det)
{
    double logd = log(det);
    double ly = logd / (sqrt(1+0.08*logd*logd));
    double y= exp(ly);
    return y;
}


double dmf(double x)
{
    double y= mf(x);
    double lx= log(x);

    double nom = 1./x * sqrt(1+0.08* lx*lx) - lx *  1./sqrt(1+0.08*lx*lx) *0.08* lx *1./x;
    double denom = 1+0.08* lx*lx;

    return y*nom/denom;
}


float  ComputeUpdateMSJac(ImageType3D::IndexType index,  
                          ImageType3D::Pointer up_img, ImageType3D::Pointer down_img,
                          DisplacementFieldType::Pointer gradI_img,  DisplacementFieldType::Pointer gradJ_img,
                          ImageType3D::Pointer detf_img, ImageType3D::Pointer detm_img, 
                          const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV,
                          GaussianSmoothingOperatorType current_Gaussian_operator,DisplacementFieldType::PixelType updateF,DisplacementFieldType::PixelType updateM,
                          int phase,int phase_xyz)

{
   updateF.Fill(0);
   updateM.Fill(0);
   
   ImageType3D::SizeType current_image_size= def_FINB->GetLargestPossibleRegion().GetSize();
   
   if(oindex[0]<1 || oindex[1]<1 || oindex[2]<1 || oindex[0]>current_image_size[0]-2 || oindex[1]>current_image_size[1]-2 || oindex[2]>current_image_size[2]-2)
       return 0;
       
   DisplacementFieldType::PixelType gradI= gradI_img->GetPixel(index);
   DisplacementFieldType::PixelType gradJ= gradJ_img->GetPixel(index);
   DisplacementFieldType::SpacingType spc= gradI_img->GetSpacing();
   
   ImageType3D::SizeType kernel_size = this->current_Gaussian_operator.GetSize();
   int kernel_length= kernel_size[phase];
   int mid_ind = (kernel_length-1)/2;
   float b= this->current_Gaussian_operator.GetElement(mid_ind);
   float a=0;
   if(mid_ind>0)
       a=this->current_Gaussian_operator.GetElement(mid_ind-1);
     
       
    float val=0;
     //////////////////////////////// at x ////////////////////////////////////
     {
         double detf = detf_img->GetPixel(index);
         double detm = detm_img->GetPixel(index);

         if(detf <=0)
             detf=1E-5;
         if(detm <=0)
             detm=1E-5;

         detf= mf(detf);
         detm= mf(detm);



         double valf= up_img->GetPixel( index )  * detf;
         double valm= down_img->GetPixel( index ) * detm;

         double K= (valf-valm);
         val=K;

         updateF[0]=  2 *K*gradI[0]*detf*b;
         updateF[1]=  2 *K*gradI[1]*detf*b;
         updateF[2]=  2 *K*gradI[2]*detf*b;
         updateM[0]=  -2 *K*gradJ[0]*detm*b;
         updateM[1]=  -2 *K*gradJ[1]*detm*b;
         updateM[2]=  -2 *K*gradJ[2]*detm*b;
     }       
   
    //////////////////////////////// at x+1 ////////////////////////////////////
     for(int h=1;h<2;h++)
     {                  
         ImageType3D::IndexType nindex=index;
         nindex[phase]=index[phase]+h;

         if(nindex[0]<h || nindex[1]<h || nindex[2]<h || nindex[0]>current_image_size[0]-h-1 || nindex[1]>current_image_size[1]-h-1 || nindex[2]>current_image_size[2]-h-1)
             continue;

         if(mid_ind-h>=0)
             a=this->current_Gaussian_operator.GetElement(mid_ind-h);
         else
             a=0;

         double fval = up_img->GetPixel(nindex);
         double mval = down_img->GetPixel(nindex);

         double detf2 = detf_img->GetPixel(nindex);
         double detm2 = detm_img->GetPixel(nindex);

         if(detf2 <=0)
             detf2=1E-5;
         if(detm2 <=0)
             detm2=1E-5;

         double detf= mf(detf2);
         double detm= mf(detm2);

         gradJ = gradJ_img->GetPixel(nindex);
         gradI = gradI_img->GetPixel(nindex);

         double K =(fval*detf -mval*detm);
         updateF[phase_xyz]+= 2 * K *  ( gradI[phase_xyz]*detf*a  + dmf(detf2)* fval* -b/2/spc[phase]/h) ;
         updateM[phase_xyz]-= 2*  K *  ( gradJ[phase_xyz]*detm*a  + dmf(detm2)* mval* -b/2/spc[phase]/h);
     }
     
         //////////////////////////////// at x-1 ////////////////////////////////////
     for(int h=1;h<2;h++)
     {                  
         ImageType3D::IndexType nindex=index;
         nindex[phase]=index[phase]-h;

         if(nindex[0]<h || nindex[1]<h || nindex[2]<h || nindex[0]>current_image_size[0]-h-1 || nindex[1]>current_image_size[1]-h-1 || nindex[2]>current_image_size[2]-h-1)
             continue;

         if(mid_ind-h>=0)
             a=this->current_Gaussian_operator.GetElement(mid_ind-h);
         else
             a=0;

         double fval = up_img->GetPixel(nindex);
         double mval = down_img->GetPixel(nindex);

         double detf2 = detf_img->GetPixel(nindex);
         double detm2 = detm_img->GetPixel(nindex);

         if(detf2 <=0)
             detf2=1E-5;
         if(detm2 <=0)
             detm2=1E-5;

         double detf= mf(detf2);
         double detm= mf(detm2);

         gradJ = gradJ_img->GetPixel(nindex);
         gradI = gradI_img->GetPixel(nindex);

         double K =(fval*detf -mval*detm);
         updateF[phase_xyz]+= 2 * K *  ( gradI[phase_xyz]*detf*a  + dmf(detf2)* fval* -b/2/spc[phase]/h) ;
         updateM[phase_xyz]-= 2*  K *  ( gradJ[phase_xyz]*detm*a  + dmf(detm2)* mval* -b/2/spc[phase]/h);
     }     
     return val;
}


float ComputeMetric_MSJac(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, 
                          const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV   ,
                          DisplacementFieldType::Pointer &updateFieldF, DisplacementFieldType::Pointer &updateFieldM,
                          vnl_vector<double> phase_vector,float update_std)
{
    updateFieldF->SetRegions(def_FINV->GetLargestPossibleRegion());
    updateFieldF->SetDirection(def_FINV->GetDirection());
    updateFieldF->SetOrigin(def_FINV->GetOrigin());
    updateFieldF->SetSpacing(def_FINV->GetSpacing());
    updateFieldF->Allocate();
    DisplacementFieldType::PixelType zero; zero.Fill(0);
    updateFieldF->FillBuffer(zero);

    updateFieldM->SetRegions(def_FINV->GetLargestPossibleRegion());
    updateFieldM->SetDirection(def_FINV->GetDirection());
    updateFieldM->SetOrigin(def_FINV->GetOrigin());
    updateFieldM->SetSpacing(def_FINV->GetSpacing());
    updateFieldM->Allocate();
    updateFieldM->FillBuffer(zero);
    
    DisplacementFieldType::Pointer gradI =    DisplacementFieldType::New();
    gradI->SetDirection(def_FINV->GetDirection());
    gradI->SetOrigin(def_FINV->GetOrigin());
    gradI->SetSpacing(def_FINV->GetSpacing());
    gradI->Allocate();
    gradI->FillBuffer(zero);
    
    DisplacementFieldType::Pointer gradJ =    DisplacementFieldType::New();
    gradJ->SetDirection(def_FINV->GetDirection());
    gradJ->SetOrigin(def_FINV->GetOrigin());
    gradJ->SetSpacing(def_FINV->GetSpacing());
    gradJ->Allocate();
    gradJ->FillBuffer(zero);
    

    int phase=0;
    if( (fabs(phase_vector[1]) > fabs(phase_vector[0])) && (fabs(phase_vector[1]) > fabs(phase_vector[2])))
        phase=1;
    if( (fabs(phase_vector[2]) > fabs(phase_vector[0])) && (fabs(phase_vector[2]) > fabs(phase_vector[1])))
        phase=2;
            
    vnl_matrix<double> dir = up_img->GetDirection().GetVnlMatrix();  
    vnl_vector<double> temp = dir*phase_vector;
    int phase_xyz;
    if( (fabs(temp[0])>fabs(temp[1])) && (fabs(temp[0])>fabs(temp[2])))
        phase_xyz=0;
    else if( (fabs(temp[1])>fabs(temp[0])) && (fabs(temp[1])>fabs(temp[2])))
        phase_xyz=1;
    else phase_xyz=2;
        

    GaussianSmoothingOperatorType gaussianSmoothingOperator;
    current_Gaussian_operator.SetDirection(phase);
    current_Gaussian_operator.SetVariance(update_std);
    current_Gaussian_operator.SetMaximumError(0.001);
    current_Gaussian_operator.SetMaximumKernelWidth(31);
    current_Gaussian_operator.CreateDirectional();

    const int h=1;
    
    ImageType3D::SizeType imsize=up_img->GetLargestPossibleRegion().GetSize();

    ImageType3D::Pointer m_MetricImage= ImageType3D::New();
    m_MetricImage->SetRegions(up_img->GetLargestPossibleRegion());
    m_MetricImage->Allocate();
    m_MetricImage->SetSpacing(up_img->GetSpacing());
    m_MetricImage->SetOrigin(up_img->GetOrigin());
    m_MetricImage->SetDirection(up_img->GetDirection());
    m_MetricImage->FillBuffer(0);
    
    ImageType3D::Pointer detf_img= ImageType3D::New();
    detf_img->SetRegions(up_img->GetLargestPossibleRegion());
    detf_img->Allocate();
    detf_img->SetSpacing(up_img->GetSpacing());
    detf_img->SetOrigin(up_img->GetOrigin());
    detf_img->SetDirection(up_img->GetDirection());
    detf_img->FillBuffer(0);
    
    ImageType3D::Pointer detm_img= ImageType3D::New();
    detm_img->SetRegions(up_img->GetLargestPossibleRegion());
    detm_img->Allocate();
    detm_img->SetSpacing(up_img->GetSpacing());
    detm_img->SetOrigin(up_img->GetOrigin());
    detm_img->SetDirection(up_img->GetDirection());
    detm_img->FillBuffer(0);
            
    #pragma omp parallel for
    for( int k=0; k<(int)imsize[2];k++)
    {
        ImageType3D::IndexType index;
        index[2]=k;

        for(unsigned int j=0; j<imsize[1];j++)
        {
            index[1]=j;
            for(unsigned int i=0; i<imsize[0];i++)
            {
                index[0]=i;
                
                DisplacementFieldType::PixelType gI=  ComputeImageGradient( up_img, index);
                DisplacementFieldType::PixelType gJ=  ComputeImageGradient( up_img, index);               
                gradI->SetPixel(index,gI);
                gradJ->SetPixel(index,gJ);                
                
                double detf= ComputeSingleJacobianMatrixAtIndex(def_FINV,index,h);
                double detm= ComputeSingleJacobianMatrixAtIndex(def_FINV,index,h);
                detf_img->SetPixel(index,detf);
                detm_img->SetPixel(index,detm);                
            }
        }
    }
    
    
    
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
                
                float mv= ComputeUpdateMSJac(index,up_img,down_img,
                                   gradI,gradJ, detf_img,detm_img, def_FINV,def_MINV,current_Gaussian_operator,updateF,updateM,phase,phase_xyz);
                m_MetricImage->SetPixel(index,mv);
                updateFieldF->SetPixel(index,updateF);
                updateFieldM->SetPixel(index,updateM);                
            }
        }
    }

  value = 0;
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

    


    
     





float ComputeMetric_CCSK(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, 
                         const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV)
{


}
                         
                         

float ComputeMetric_CC(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                       const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV  )
{


}




#endif
