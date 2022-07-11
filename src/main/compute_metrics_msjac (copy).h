#ifndef _COMPUTEMETRICSMSJAC_H
#define _COMPUTEMETRICSMSJAC_H

#include "defines.h"
#include "itkGaussianOperator.h"

    using GaussianSmoothingOperatorType = itk::GaussianOperator<float, 3>;





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


double ComputeSingleJacobianMatrixAtIndex(const DisplacementFieldType::Pointer disp_field,ImageType3D::IndexType &index, int h,int phase, int phase_xyz)
{
    if(index[phase]<h || index[phase]> disp_field->GetLargestPossibleRegion().GetSize()[phase]-h-1)
        return 1.;

    bool do_second_order=false;
      // bool do_second_order=true;
       //if(index[phase]<=h || index[phase]>= disp_field->GetLargestPossibleRegion().GetSize()[0]-1-h)
       //    do_second_order=false;

       double grad;

       if(do_second_order)
       {

               DisplacementFieldType::IndexType rind=index;
               DisplacementFieldType::IndexType lind=index;
               DisplacementFieldType::IndexType rrind=index;
               DisplacementFieldType::IndexType llind=index;
               rind[phase]+=h;
               lind[phase]-=h;
               rrind[phase]+=2*h;
               llind[phase]-=2*h;

               double lval = disp_field->GetPixel(lind)[phase];
               double rval = disp_field->GetPixel(rind)[phase];
               double llval = disp_field->GetPixel(llind)[phase];
               double rrval = disp_field->GetPixel(rrind)[phase];

               grad= (-rrval+8.*rval-8.*lval+llval)/12./disp_field->GetSpacing()[phase]/h;


       }
       else
       {

               DisplacementFieldType::IndexType rind=index;
               DisplacementFieldType::IndexType lind=index;
               rind[phase]+=h;
               lind[phase]-=h;

               double lval = disp_field->GetPixel(lind)[phase];
               double rval = disp_field->GetPixel(rind)[phase];

               grad= 0.5*(rval-lval)/disp_field->GetSpacing()[phase]/h;
       }

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
                          int phase,int phase_xyz)
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
             a=current_Gaussian_operator.GetElement(mid_ind-h);
         else
             a=0;

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
             a=current_Gaussian_operator.GetElement(mid_ind-h);
         else
             a=0;

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
         updateF[phase_xyz]+= 2 * K *  ( gradI[phase_xyz]*detf*a  + dmf(detf2)* fval* b/2/spc[phase]/h) ;
         updateM[phase_xyz]-= 2*  K *  ( gradJ[phase_xyz]*detm*a  + dmf(detm2)* mval* b/2/spc[phase]/h);
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
    vnl_matrix<double> dir = up_img->GetDirection().GetVnlMatrix();  
    vnl_vector<double> temp = dir*phase_vector;
    int phase_xyz;
    if( (fabs(temp[0])>fabs(temp[1])) && (fabs(temp[0])>fabs(temp[2])))
        phase_xyz=0;
    else if( (fabs(temp[1])>fabs(temp[0])) && (fabs(temp[1])>fabs(temp[2])))
        phase_xyz=1;
    else phase_xyz=2;

        
    const int h=1;   
    ImageType3D::SizeType imsize=up_img->GetLargestPossibleRegion().GetSize();

    
    std::vector<double> slice_sums;
    slice_sums.resize(imsize[2]);


    #pragma omp parallel for
    for( int k=0; k<(int)imsize[2];k++)
    {
        ImageType3D::IndexType index;
        index[2]=k;
        slice_sums[k]=0;

        DisplacementFieldType::PixelType updateF,updateM;

        for(unsigned int j=0; j<imsize[1];j++)
        {
            index[1]=j;
            for(unsigned int i=0; i<imsize[0];i++)
            {
                index[0]=i;
                
                slice_sums[k]+= ComputeUpdateMSJac(index,
                                             up_img,down_img,
                                             def_FINV,def_MINV,
                                             current_Gaussian_operator,
                                             updateF,updateM,
                                             phase,phase_xyz);                
                updateFieldF->SetPixel(index,updateF);
                updateFieldM->SetPixel(index,updateM);                
            }
        }
    }

  double value = 0;
  for( int k=0; k<(int)imsize[2];k++)
      value+=slice_sums[k];
  value=value/(imsize[0]*imsize[1]*imsize[2]);

  return value;
}

    



#endif
