#ifndef _COMPUTEMETRICSMSJAC_H
#define _COMPUTEMETRICSMSJAC_H

#include "defines.h"
#include "itkGaussianOperator.h"

    using GaussianSmoothingOperatorType = itk::GaussianOperator<float, 3>;





double mf(double det)
{
    double logd = log(det);
    double ly = logd / (sqrt(1+0.2*logd*logd));
    double y= exp(ly);
    return y;
}


double dmf(double x)
{
    double y= mf(x);
    double lx= log(x);

    double nom = 1./x * sqrt(1+0.2* lx*lx) - lx *  1./sqrt(1+0.2*lx*lx) *0.2* lx *1./x;
    double denom = 1+0.2* lx*lx;

    return y*nom/denom;
}


double ComputeSingleJacobianMatrixAtIndex(const DisplacementFieldType::Pointer disp_field,ImageType3D::IndexType &index, int h,int phase, int phase_xyz)
{
    if(index[phase]<h || index[phase]> disp_field->GetLargestPossibleRegion().GetSize()[phase]-h-1)
        return 1.;


       DisplacementFieldType::IndexType Nind=index;
       Nind[phase]+=h;
       double grad=disp_field->GetPixel(Nind)[phase];
       Nind[phase]-=2*h;
       grad-=disp_field->GetPixel(Nind)[phase];
       grad*= 0.5/disp_field->GetSpacing()[phase]/h;


       vnl_vector<double> temp(3,0);
       temp[phase]=grad;
       temp = disp_field->GetDirection().GetVnlMatrix()* temp;

       return 1 + temp[phase_xyz];
}


float  ComputeUpdateMSJac(ImageType3D::IndexType index,
                          const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img,
                          const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV,
                          GaussianSmoothingOperatorType &current_Gaussian_operator,
                          DisplacementFieldType::PixelType &updateF,DisplacementFieldType::PixelType &updateM,
                          const int phase, const int phase_xyz,const vnl_vector<double> &new_phase_vector)
{
   updateF.Fill(0);
   updateM.Fill(0);

   ImageType3D::SizeType current_image_size= def_FINV->GetLargestPossibleRegion().GetSize();

   if(index[0]<1 || index[1]<1 || index[2]<1 || index[0]>current_image_size[0]-2 || index[1]>current_image_size[1]-2 || index[2]>current_image_size[2]-2)
       return 0;


   DisplacementFieldType::PixelType gradI = ComputeImageGradient( up_img, index);
   DisplacementFieldType::PixelType gradJ = ComputeImageGradient( down_img, index);

   DisplacementFieldType::SpacingType spc= up_img->GetSpacing();


   ImageType3D::SizeType kernel_size = current_Gaussian_operator.GetSize();
   int kernel_length= kernel_size[phase];
   int mid_ind = (kernel_length-1)/2;
   float b= current_Gaussian_operator.GetElement(mid_ind);
   float a=0;
   if(mid_ind>0)
       a=current_Gaussian_operator.GetElement(mid_ind-1);
       
   float a_b= a/b;


    float val=0;
     //////////////////////////////// at x ////////////////////////////////////
     {
         double detf = ComputeSingleJacobianMatrixAtIndex(def_FINV,index,1,phase,phase_xyz);
         double detm = ComputeSingleJacobianMatrixAtIndex(def_MINV,index,1,phase,phase_xyz);

         if(detf <=0)
             detf=1E-5;
         if(detm <=0)
             detm=1E-5;

         detf= mf(detf);
         detm= mf(detm);


         double valf= up_img->GetPixel( index )  * detf;
         double valm= down_img->GetPixel( index ) * detm;

         double K= (valf-valm);
         val=K*K;

         updateF[0]=  2 *K*gradI[0]*detf;
         updateF[1]=  2 *K*gradI[1]*detf;
         updateF[2]=  2 *K*gradI[2]*detf;
         updateM[0]=  -2 *K*gradJ[0]*detm;
         updateM[1]=  -2 *K*gradJ[1]*detm;
         updateM[2]=  -2 *K*gradJ[2]*detm;
     }


    //////////////////////////////// at x+1 ////////////////////////////////////
     for(int h=1;h<2;h++)
     {
         ImageType3D::IndexType nindex=index;         
         nindex[phase]+=h;

         if(nindex[0]<h || nindex[1]<h || nindex[2]<h || nindex[0]>current_image_size[0]-h-1 || nindex[1]>current_image_size[1]-h-1 || nindex[2]>current_image_size[2]-h-1)
             continue;

         if(mid_ind-h>=0)
             a=current_Gaussian_operator.GetElement(mid_ind-h);
         else
             a=0;
         a_b=a/b;

         double fval = up_img->GetPixel(nindex);
         double mval = down_img->GetPixel(nindex);

         double detf2 = ComputeSingleJacobianMatrixAtIndex(def_FINV,nindex,h,phase,phase_xyz);
         double detm2 = ComputeSingleJacobianMatrixAtIndex(def_MINV,nindex,h,phase,phase_xyz);

         if(detf2 <=0)
             detf2=1E-5;
         if(detm2 <=0)
             detm2=1E-5;

         double detf= mf(detf2);
         double detm= mf(detm2);

         gradI = ComputeImageGradient( up_img, nindex);
         gradJ = ComputeImageGradient( down_img, nindex);


         double K =(fval*detf -mval*detm);         

         updateF[phase_xyz]+= 2 * K *  ( gradI[phase_xyz]*detf*a_b  -new_phase_vector[phase_xyz]* dmf(detf2)* fval* 0.5/spc[phase]/h) ;
         updateM[phase_xyz]-= 2*  K *  ( gradJ[phase_xyz]*detm*a_b  -new_phase_vector[phase_xyz]* dmf(detm2)* mval* 0.5/spc[phase]/h);
     }

         //////////////////////////////// at x-1 ////////////////////////////////////
     for(int h=1;h<2;h++)
     {
         ImageType3D::IndexType nindex=index;
         nindex[phase]-=h;

         if(nindex[0]<h || nindex[1]<h || nindex[2]<h || nindex[0]>current_image_size[0]-h-1 || nindex[1]>current_image_size[1]-h-1 || nindex[2]>current_image_size[2]-h-1)
             continue;

         if(mid_ind-h>=0)
             a=current_Gaussian_operator.GetElement(mid_ind-h);
         else
             a=0;
         a_b=a/b;

         double fval = up_img->GetPixel(nindex);
         double mval = down_img->GetPixel(nindex);
         double detf2 = ComputeSingleJacobianMatrixAtIndex(def_FINV,nindex,h,phase,phase_xyz);
         double detm2 = ComputeSingleJacobianMatrixAtIndex(def_MINV,nindex,h,phase,phase_xyz);

         if(detf2 <=0)
             detf2=1E-5;
         if(detm2 <=0)
             detm2=1E-5;

         double detf= mf(detf2);
         double detm= mf(detm2);

         gradI = ComputeImageGradient( up_img, nindex);
         gradJ = ComputeImageGradient( down_img, nindex);

         double K =(fval*detf -mval*detm);
         updateF[phase_xyz]+= 2 * K *  ( gradI[phase_xyz]*detf*a_b  + new_phase_vector[phase_xyz]*dmf(detf2)* fval* 0.5/spc[phase]/h) ;
         updateM[phase_xyz]-= 2*  K *  ( gradJ[phase_xyz]*detm*a_b  + new_phase_vector[phase_xyz]*dmf(detm2)* mval* 0.5/spc[phase]/h);
     }
     return val;
}


float ComputeMetric_MSJac(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img,
                          const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV   ,
                          DisplacementFieldType::Pointer &updateFieldF, DisplacementFieldType::Pointer &updateFieldM,
                          vnl_vector<double> phase_vector,itk::GaussianOperator<float,3> &current_Gaussian_operator)
{
    updateFieldF = DisplacementFieldType::New();
    updateFieldF->SetRegions(def_FINV->GetLargestPossibleRegion());
    updateFieldF->SetDirection(def_FINV->GetDirection());
    updateFieldF->SetOrigin(def_FINV->GetOrigin());
    updateFieldF->SetSpacing(def_FINV->GetSpacing());
    updateFieldF->Allocate();

    updateFieldM = DisplacementFieldType::New();
    updateFieldM->SetRegions(def_FINV->GetLargestPossibleRegion());
    updateFieldM->SetDirection(def_FINV->GetDirection());
    updateFieldM->SetOrigin(def_FINV->GetOrigin());
    updateFieldM->SetSpacing(def_FINV->GetSpacing());
    updateFieldM->Allocate();


    int phase=0;
    if( (fabs(phase_vector[1]) > fabs(phase_vector[0])) && (fabs(phase_vector[1]) > fabs(phase_vector[2])))
        phase=1;
    if( (fabs(phase_vector[2]) > fabs(phase_vector[0])) && (fabs(phase_vector[2]) > fabs(phase_vector[1])))
        phase=2;

    vnl_matrix_fixed<double,3,3> dir = up_img->GetDirection().GetVnlMatrix();
    vnl_vector_fixed<double,3> new_phase_vector = dir*phase_vector;
    int phase_xyz;
    if( (fabs(new_phase_vector[0])>fabs(new_phase_vector[1])) && (fabs(new_phase_vector[0])>fabs(new_phase_vector[2])))
        phase_xyz=0;
    else if( (fabs(new_phase_vector[1])>fabs(new_phase_vector[0])) && (fabs(new_phase_vector[1])>fabs(new_phase_vector[2])))
        phase_xyz=1;
    else phase_xyz=2;


    const int h=1;
    ImageType3D::SizeType imsize=up_img->GetLargestPossibleRegion().GetSize();

    ImageType3D::Pointer m_MetricImage= ImageType3D::New();
    m_MetricImage->SetRegions(up_img->GetLargestPossibleRegion());
    m_MetricImage->Allocate();
    m_MetricImage->SetSpacing(up_img->GetSpacing());
    m_MetricImage->SetOrigin(up_img->GetOrigin());
    m_MetricImage->SetDirection(up_img->GetDirection());



    #pragma omp parallel for //collapse(2)
    for( int k=0; k<(int)imsize[2];k++)
    {
        for(unsigned int j=0; j<imsize[1];j++)
        {
            ImageType3D::IndexType index;
            index[2]=k;
            index[1]=j;

            DisplacementFieldType::PixelType updateF, updateM;

            for(unsigned int i=0; i<imsize[0];i++)
            {
                index[0]=i;

                float mv= ComputeUpdateMSJac(index,
                                             up_img,down_img,
                                             def_FINV,def_MINV,
                                             current_Gaussian_operator,
                                             updateF,updateM,
                                             phase,phase_xyz, new_phase_vector);

                updateFieldF->SetPixel(index,updateF);
                updateFieldM->SetPixel(index,updateM);
                m_MetricImage->SetPixel(index,mv);
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





#endif
