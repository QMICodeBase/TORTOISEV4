#ifndef _COMPUTEMETRICSCCJACS_H
#define _COMPUTEMETRICSCCJACS_H

#include "defines.h"
#include "itkGaussianOperator.h"
#include "compute_metrics_msjac.h"
#include "itkLinearInterpolateImageFunction.h"

    using GaussianSmoothingOperatorType = itk::GaussianOperator<float, 3>;

#define WIN_RAD_JAC 4
#define WIN_RAD_JAC_Z 2
#define LIMCCJAC (1E-5)


struct FinDiff
{
    float sSS,sSK,sKK, valS, valK;
};
using StructImageType = itk::Image<FinDiff,3>;


float  ComputeUpdateCCJacS(ImageType3D::IndexType index,
                          const ImageType3D::Pointer b0_img, const DisplacementFieldType::Pointer grad_img,
                          const StructImageType::Pointer findiff_img,
                          const DisplacementFieldType::Pointer field,
                          GaussianSmoothingOperatorType &current_Gaussian_operator,
                          DisplacementFieldType::PixelType &update,
                          int phase,int phase_xyz,const vnl_vector<double> &new_phase_vector)
{
   update.Fill(0);

   ImageType3D::SizeType d_sz= field->GetLargestPossibleRegion().GetSize();

   if(index[0]<1 || index[1]<1 || index[2]<1 || index[0]>d_sz[0]-2 || index[1]>d_sz[1]-2 || index[2]>d_sz[2]-2)
       return 0;



   DisplacementFieldType::SpacingType spc= b0_img->GetSpacing();


   ImageType3D::SizeType kernel_size = current_Gaussian_operator.GetSize();
   int kernel_length= kernel_size[phase];
   int mid_ind = (kernel_length-1)/2;   
   float b= current_Gaussian_operator.GetElement(mid_ind);
   float a=0;
   if(mid_ind>0)
       a=current_Gaussian_operator.GetElement(mid_ind-1);
       
   float a_b= a/b;
   a_b=0;



    float val=0;
     //////////////////////////////// at x ////////////////////////////////////
     {

        FinDiff mfindiff = findiff_img->GetPixel(index);

        float sSS_sKK= mfindiff.sSS * mfindiff.sKK;
        if(fabs(sSS_sKK) > LIMCCJAC && fabs(mfindiff.sKK) > LIMCCJAC )
        {
            val= -mfindiff.sSK*mfindiff.sSK/ sSS_sKK;
            float first_term = -2* mfindiff.sSK / sSS_sKK;

            float detF2 = ComputeSingleJacobianMatrixAtIndex(field,index,1,phase,phase_xyz)+1;
            if(detF2 <=0)
                detF2=1E-5;
            float detF= mf(detF2);

            DisplacementFieldType::PixelType M1= grad_img->GetPixel(index)*detF;

            float M2= 0;
            M1[phase_xyz]+=M2;

            float second_term= (mfindiff.valS - mfindiff.sSS/ mfindiff.sKK *mfindiff.valK);
            update[0] = first_term * second_term *M1[0] ;
            update[1] = first_term * second_term *M1[1] ;
            update[2] = first_term * second_term *M1[2] ;
        }

     }


    const int h=1;
    //////////////////////////////// at x+1 ////////////////////////////////////
    {        
        ImageType3D::IndexType nindex=index;
        nindex[phase]+=h;

        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
        {
            FinDiff mfindiff = findiff_img->GetPixel(nindex);
            float sSS_sKK= mfindiff.sSS * mfindiff.sKK;
            if(fabs(sSS_sKK) > LIMCCJAC && fabs(mfindiff.sKK) > LIMCCJAC )
            {
                float first_term= -2*mfindiff.sSK/ sSS_sKK;

                float detF2 = ComputeSingleJacobianMatrixAtIndex(field,nindex,1,phase,phase_xyz)+1;
                if(detF2 <=0)
                    detF2=1E-5;
                float detF= mf(detF2);


              DisplacementFieldType::PixelType M1 = grad_img->GetPixel(nindex);
              M1[0]*= detF*a_b;
              M1[1]*= detF*a_b;
              M1[2]*= detF*a_b;

              float M2= new_phase_vector[phase_xyz]*dmf(detF2)* b0_img->GetPixel(nindex)* -0.5/spc[phase]/h ;
              M1[phase_xyz]+=M2;

              float second_term= (mfindiff.valS - mfindiff.sSK/ mfindiff.sKK *mfindiff.valK);
              //update[phase_xyz] += first_term * second_term *M1[phase_xyz] ;
              update[0] += first_term * second_term *M1[0] ;
              update[1] += first_term * second_term *M1[1] ;
              update[2] += first_term * second_term *M1[2] ;
            }
        }
    }


         //////////////////////////////// at x-1 ////////////////////////////////////
    {        
        ImageType3D::IndexType nindex=index;
        nindex[phase]-=h;


        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
        {
            FinDiff mfindiff = findiff_img->GetPixel(nindex);
            float sSS_sKK= mfindiff.sSS * mfindiff.sKK;
            if(fabs(sSS_sKK) > LIMCCJAC && fabs(mfindiff.sKK) > LIMCCJAC )
            {
                float first_term= -2*mfindiff.sSK/ sSS_sKK;

                float detF2 = ComputeSingleJacobianMatrixAtIndex(field,nindex,1,phase,phase_xyz)+1;
                if(detF2 <=0)
                    detF2=1E-5;
                float detF= mf(detF2);


              DisplacementFieldType::PixelType M1 = grad_img->GetPixel(nindex);
              M1[0]*= detF*a_b;
              M1[1]*= detF*a_b;
              M1[2]*= detF*a_b;

              float M2= new_phase_vector[phase_xyz]*dmf(detF2)* b0_img->GetPixel(nindex)* 0.5/spc[phase]/h ;
              M1[phase_xyz]+=M2;

              float second_term= (mfindiff.valS - mfindiff.sSK/ mfindiff.sKK *mfindiff.valK);
              //update[phase_xyz] += first_term * second_term *M1[phase_xyz] ;
              update[0] += first_term * second_term *M1[0] ;
              update[1] += first_term * second_term *M1[1] ;
              update[2] += first_term * second_term *M1[2] ;
            }
        }

    }

    return val;
}


FinDiff  ComputeFinDiff(ImageType3D::IndexType index,ImageType3D::Pointer b0_img,ImageType3D::Pointer str_img)
{
    ImageType3D::SizeType d_sz= b0_img->GetLargestPossibleRegion().GetSize();

    ImageType3D::IndexType start,end;

    start[2]=index[2]-WIN_RAD_JAC_Z;
    if(start[2]<0)
        start[2]=0;
    start[1]=index[1]-WIN_RAD_JAC;
    if(start[1]<0)
        start[1]=0;
    start[0]=index[0]-WIN_RAD_JAC;
    if(start[0]<0)
        start[0]=0;

    end[2]=index[2]+WIN_RAD_JAC_Z+1;
    if(end[2]>d_sz[2])
        end[2]=d_sz[2];
    end[1]=index[1]+WIN_RAD_JAC+1;
    if(end[1]>d_sz[1])
        end[1]=d_sz[1];
    end[0]=index[0]+WIN_RAD_JAC+1;
    if(end[0]>d_sz[0])
        end[0]=d_sz[0];

    double suma2 = 0.0;
    double suma = 0.0;
    double  sumac=0;
    double sumc2 = 0.0;
    double sumc = 0.0;
    int N=0;

    float valK_center=b0_img->GetPixel(index);
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

                float Kim = b0_img->GetPixel(cind);
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
    FinDiff mfindif;

    mfindif.valK = valK_center-Kmean;
    mfindif.valS = valS_center -Smean;
    mfindif.sKK = suma2 - Kmean*suma;
    mfindif.sSS = sumc2 - Smean*sumc;
    mfindif.sSK = sumac - Kmean*sumc;

    return mfindif;
}

float ComputeMetric_CCJacS(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, const ImageType3D::Pointer str_img,
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
    m_MetricImage->FillBuffer(0);




    {
        ImageType3D::Pointer updet_img= ImageType3D::New();
        updet_img->SetRegions(up_img->GetLargestPossibleRegion());
        updet_img->Allocate();
        updet_img->SetSpacing(up_img->GetSpacing());
        updet_img->SetOrigin(up_img->GetOrigin());
        updet_img->SetDirection(up_img->GetDirection());

        DisplacementFieldType::Pointer upgrad_img= DisplacementFieldType::New();
        upgrad_img->SetRegions(up_img->GetLargestPossibleRegion());
        upgrad_img->Allocate();
        upgrad_img->SetSpacing(up_img->GetSpacing());
        upgrad_img->SetOrigin(up_img->GetOrigin());
        upgrad_img->SetDirection(up_img->GetDirection());

        ImageType3D::Pointer downdet_img= ImageType3D::New();
        downdet_img->SetRegions(down_img->GetLargestPossibleRegion());
        downdet_img->Allocate();
        downdet_img->SetSpacing(up_img->GetSpacing());
        downdet_img->SetOrigin(up_img->GetOrigin());
        downdet_img->SetDirection(up_img->GetDirection());

        DisplacementFieldType::Pointer downgrad_img= DisplacementFieldType::New();
        downgrad_img->SetRegions(up_img->GetLargestPossibleRegion());
        downgrad_img->Allocate();
        downgrad_img->SetSpacing(up_img->GetSpacing());
        downgrad_img->SetOrigin(up_img->GetOrigin());
        downgrad_img->SetDirection(up_img->GetDirection());


        StructImageType::Pointer upfindiff_img= StructImageType::New();
        upfindiff_img->SetRegions(up_img->GetLargestPossibleRegion());
        upfindiff_img->Allocate();
        upfindiff_img->SetSpacing(up_img->GetSpacing());
        upfindiff_img->SetOrigin(up_img->GetOrigin());
        upfindiff_img->SetDirection(up_img->GetDirection());

        StructImageType::Pointer downfindiff_img= StructImageType::New();
        downfindiff_img->SetRegions(up_img->GetLargestPossibleRegion());
        downfindiff_img->Allocate();
        downfindiff_img->SetSpacing(up_img->GetSpacing());
        downfindiff_img->SetOrigin(up_img->GetOrigin());
        downfindiff_img->SetDirection(up_img->GetDirection());



        #pragma omp parallel for collapse(2)
        for( int k=0; k<(int)imsize[2];k++)
        {
            for(unsigned int j=0; j<imsize[1];j++)
            {
                ImageType3D::IndexType index;
                index[2]=k;
                index[1]=j;

                for(unsigned int i=0; i<imsize[0];i++)
                {
                    index[0]=i;

                    upgrad_img->SetPixel(index,ComputeImageGradient( up_img, index));
                    downgrad_img->SetPixel(index,ComputeImageGradient( down_img, index));

                    {
                        double det2= ComputeSingleJacobianMatrixAtIndex(def_FINV,index,1,phase,phase_xyz)+1;
                        if(det2 <=0)
                            det2=1E-5;
                        float det= mf(det2);
                        updet_img->SetPixel(index,det*up_img->GetPixel(index));
                    }
                    {
                        double det2= ComputeSingleJacobianMatrixAtIndex(def_MINV,index,1,phase,phase_xyz)+1;
                        if(det2 <=0)
                            det2=1E-5;
                        float det= mf(det2);
                        downdet_img->SetPixel(index,det*down_img->GetPixel(index));
                    }
                }
            }
        }


        #pragma omp parallel for collapse(2)
        for( int k=0; k<(int)imsize[2];k++)
        {
            for(unsigned int j=0; j<imsize[1];j++)
            {
                ImageType3D::IndexType index;
                index[2]=k;
                index[1]=j;

                for(unsigned int i=0; i<imsize[0];i++)
                {
                    index[0]=i;

                    FinDiff upvals= ComputeFinDiff(index,updet_img,str_img);
                    FinDiff downvals= ComputeFinDiff(index,downdet_img,str_img);
                    upfindiff_img->SetPixel(index,upvals);
                    downfindiff_img->SetPixel(index,downvals);
                }
            }
        }




        #pragma omp parallel for
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

                    float mv1= ComputeUpdateCCJacS(index,
                                                   up_img,  upgrad_img,
                                                   upfindiff_img,
                                                   def_FINV,
                                                   current_Gaussian_operator,
                                                   updateF,
                                                   phase,phase_xyz,new_phase_vector);

                    float mv2= ComputeUpdateCCJacS(index,
                                                   down_img,  downgrad_img,
                                                   downfindiff_img,
                                                   def_MINV,
                                                   current_Gaussian_operator,
                                                   updateM,
                                                   phase,phase_xyz,new_phase_vector);

                    updateFieldF->SetPixel(index,updateF);
                    updateFieldM->SetPixel(index,updateM);
                    m_MetricImage->SetPixel(index,mv1+mv2);
                }
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







float  ComputeUpdateCCJacSSingle(ImageType3D::IndexType index,
                          const ImageType3D::Pointer b0_img, const std::vector<ImageType3D::Pointer> img_gradient,
                          const StructImageType::Pointer findiff_img,
                          const DisplacementFieldType::Pointer field,
                          GaussianSmoothingOperatorType &current_Gaussian_operator,
                          DisplacementFieldType::PixelType &update,
                          int phase,int phase_xyz,const vnl_vector<double> &new_phase_vector)
{
   update.Fill(0);

   ImageType3D::SizeType d_sz= field->GetLargestPossibleRegion().GetSize();

   if(index[0]<1 || index[1]<1 || index[2]<1 || index[0]>d_sz[0]-2 || index[1]>d_sz[1]-2 || index[2]>d_sz[2]-2)
       return 0;


   DisplacementFieldType::SpacingType spc= b0_img->GetSpacing();


//   ImageType3D::SizeType kernel_size = current_Gaussian_operator.GetSize();
//   int kernel_length= kernel_size[phase];
//   int mid_ind = (kernel_length-1)/2;
//   float b= current_Gaussian_operator.GetElement(mid_ind);
//   float a=0;
//   if(mid_ind>0)
//       a=current_Gaussian_operator.GetElement(mid_ind-1);

//   float a_b= a/b;
     float a_b=0;


     typedef itk::LinearInterpolateImageFunction<ImageType3D,double> InterpolatorType;
     InterpolatorType::Pointer interp_x = InterpolatorType::New();
     interp_x->SetInputImage(img_gradient[0]);
     InterpolatorType::Pointer interp_y = InterpolatorType::New();
     interp_y->SetInputImage(img_gradient[1]);
     InterpolatorType::Pointer interp_z = InterpolatorType::New();
     interp_z->SetInputImage(img_gradient[2]);



    float val=0;
     //////////////////////////////// at x ////////////////////////////////////
     {

        FinDiff mfindiff = findiff_img->GetPixel(index);

        float sSS_sKK= mfindiff.sSS * mfindiff.sKK;
        if(fabs(sSS_sKK) > LIMCCJAC && fabs(mfindiff.sKK) > LIMCCJAC )
        {
            val= -mfindiff.sSK*mfindiff.sSK/ sSS_sKK;

            ImageType3D::PointType xyz;
            b0_img->TransformIndexToPhysicalPoint(index,xyz);
            vnl_vector<double> xyz_p = xyz.GetVnlVector() + field->GetPixel(index).GetVnlVector();
            ImageType3D::PointType xyz_pp; xyz_pp[0]=xyz_p[0]; xyz_pp[1]=xyz_p[1]; xyz_pp[2]=xyz_p[2];
            itk::ContinuousIndex<double,3> ijk_pp;
            b0_img->TransformPhysicalPointToContinuousIndex(xyz_pp,ijk_pp);
            double grad_x=0,grad_y=0,grad_z=0;
            if(interp_x->IsInsideBuffer(ijk_pp))
            {
                grad_x=interp_x->EvaluateAtContinuousIndex(ijk_pp);
                grad_y=interp_y->EvaluateAtContinuousIndex(ijk_pp);
                grad_z=interp_z->EvaluateAtContinuousIndex(ijk_pp);
            }

            double first_term= -2* mfindiff.sSK / sSS_sKK;

            double det = ComputeSingleJacobianMatrixAtIndex(field,index,1,phase,phase_xyz);
            if(det <=-1)
               det=-1+1E-5;



            double M1[3]= {grad_x *(1+det),grad_y *(1+det),grad_z*(1+det)};
            float M2= 0;
            M1[phase_xyz]+=M2;

            double second_term= (mfindiff.valS - mfindiff.sSS/ mfindiff.sKK *mfindiff.valK);
            update[0] = first_term * second_term *M1[0] ;
            update[1] = first_term * second_term *M1[1] ;
            update[2] = first_term * second_term *M1[2] ;
        }
     }


    const int h=1;
    //////////////////////////////// at x+1 ////////////////////////////////////
    {
        ImageType3D::IndexType nindex=index;
        nindex[phase]+=h;

        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
        {
            FinDiff mfindiff = findiff_img->GetPixel(nindex);
            float sSS_sKK= mfindiff.sSS * mfindiff.sKK;
            if(fabs(sSS_sKK) > LIMCCJAC && fabs(mfindiff.sKK) > LIMCCJAC )
            {
                ImageType3D::PointType xyz;
                b0_img->TransformIndexToPhysicalPoint(nindex,xyz);
                vnl_vector<double> xyz_p = xyz.GetVnlVector() + field->GetPixel(nindex).GetVnlVector();
                ImageType3D::PointType xyz_pp; xyz_pp[0]=xyz_p[0]; xyz_pp[1]=xyz_p[1]; xyz_pp[2]=xyz_p[2];
                itk::ContinuousIndex<double,3> ijk_pp;
                b0_img->TransformPhysicalPointToContinuousIndex(xyz_pp,ijk_pp);
                double grad_x=0,grad_y=0,grad_z=0;
                if(interp_x->IsInsideBuffer(ijk_pp))
                {
                    grad_x=interp_x->EvaluateAtContinuousIndex(ijk_pp);
                    grad_y=interp_y->EvaluateAtContinuousIndex(ijk_pp);
                    grad_z=interp_z->EvaluateAtContinuousIndex(ijk_pp);
                }

                double first_term= -2* mfindiff.sSK / sSS_sKK;
                double det = ComputeSingleJacobianMatrixAtIndex(field,nindex,1,phase,phase_xyz);
                if(det <=-1)
                   det=-1+1E-5;


                double M1[3]= {grad_x *(1+det)*a_b,grad_y *(1+det)*a_b,grad_z*(1+det)*a_b};
                double M2= new_phase_vector[phase_xyz]* b0_img->GetPixel(nindex)* -0.5/spc[phase]/h ;;
                M1[phase_xyz]+=M2;

                double second_term= (mfindiff.valS - mfindiff.sSK/ mfindiff.sKK *mfindiff.valK);
                update[0] += first_term * second_term *M1[0] ;
                update[1] += first_term * second_term *M1[1] ;
                update[2] += first_term * second_term *M1[2] ;
            }
        }
    }


         //////////////////////////////// at x-1 ////////////////////////////////////
    {
        ImageType3D::IndexType nindex=index;
        nindex[phase]-=h;


        if(nindex[0]>=h && nindex[1]>=h && nindex[2]>=h && nindex[0]<=d_sz[0]-h-1 && nindex[1]<=d_sz[1]-h-1 && nindex[2]<=d_sz[2]-h-1)
        {
            FinDiff mfindiff = findiff_img->GetPixel(nindex);
            float sSS_sKK= mfindiff.sSS * mfindiff.sKK;
            if(fabs(sSS_sKK) > LIMCCJAC && fabs(mfindiff.sKK) > LIMCCJAC )
            {
                ImageType3D::PointType xyz;
                b0_img->TransformIndexToPhysicalPoint(nindex,xyz);
                vnl_vector<double> xyz_p = xyz.GetVnlVector() + field->GetPixel(nindex).GetVnlVector();
                ImageType3D::PointType xyz_pp; xyz_pp[0]=xyz_p[0]; xyz_pp[1]=xyz_p[1]; xyz_pp[2]=xyz_p[2];
                itk::ContinuousIndex<double,3> ijk_pp;
                b0_img->TransformPhysicalPointToContinuousIndex(xyz_pp,ijk_pp);
                double grad_x=0,grad_y=0,grad_z=0;
                if(interp_x->IsInsideBuffer(ijk_pp))
                {
                    grad_x=interp_x->EvaluateAtContinuousIndex(ijk_pp);
                    grad_y=interp_y->EvaluateAtContinuousIndex(ijk_pp);
                    grad_z=interp_z->EvaluateAtContinuousIndex(ijk_pp);
                }

                double first_term= -2* mfindiff.sSK / sSS_sKK;
                double det = ComputeSingleJacobianMatrixAtIndex(field,nindex,1,phase,phase_xyz);
                if(det <=-1)
                   det=-1+1E-5;


                double M1[3]= {grad_x *(1+det)*a_b,grad_y *(1+det)*a_b,grad_z*(1+det)*a_b};
                double M2= new_phase_vector[phase_xyz]* b0_img->GetPixel(nindex)* 0.5/spc[phase]/h ;;
                M1[phase_xyz]+=M2;

                double second_term= (mfindiff.valS - mfindiff.sSK/ mfindiff.sKK *mfindiff.valK);
                update[0] += first_term * second_term *M1[0] ;
                update[1] += first_term * second_term *M1[1] ;
                update[2] += first_term * second_term *M1[2] ;
            }
        }

    }
    update=-update;

    return val;
}











float ComputeMetric_CCJacSSingle(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, const ImageType3D::Pointer str_img,
                                 const std::vector<ImageType3D::Pointer> up_img_gradient,const std::vector<ImageType3D::Pointer> down_img_gradient,
                                 const DisplacementFieldType::Pointer def_FINV,
                                 DisplacementFieldType::Pointer &updateFieldINV,
                                 vnl_vector<double> phase_vector,itk::GaussianOperator<float,3> &current_Gaussian_operator)
{
    updateFieldINV = DisplacementFieldType::New();
    updateFieldINV->SetRegions(def_FINV->GetLargestPossibleRegion());
    updateFieldINV->SetDirection(def_FINV->GetDirection());
    updateFieldINV->SetOrigin(def_FINV->GetOrigin());
    updateFieldINV->SetSpacing(def_FINV->GetSpacing());
    updateFieldINV->Allocate();


    DisplacementFieldType::Pointer def_MINV = DisplacementFieldType::New();
    def_MINV->SetRegions(def_FINV->GetLargestPossibleRegion());
    def_MINV->SetDirection(def_FINV->GetDirection());
    def_MINV->SetOrigin(def_FINV->GetOrigin());
    def_MINV->SetSpacing(def_FINV->GetSpacing());
    def_MINV->Allocate();

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(def_MINV,def_MINV->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3 = it.GetIndex();
        it.Set(-1.*def_FINV->GetPixel(ind3));
    }


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
    m_MetricImage->FillBuffer(0);




    {
        ImageType3D::Pointer updet_img= ImageType3D::New();
        updet_img->SetRegions(up_img->GetLargestPossibleRegion());
        updet_img->Allocate();
        updet_img->SetSpacing(up_img->GetSpacing());
        updet_img->SetOrigin(up_img->GetOrigin());
        updet_img->SetDirection(up_img->GetDirection());

        ImageType3D::Pointer downdet_img= ImageType3D::New();
        downdet_img->SetRegions(down_img->GetLargestPossibleRegion());
        downdet_img->Allocate();
        downdet_img->SetSpacing(up_img->GetSpacing());
        downdet_img->SetOrigin(up_img->GetOrigin());
        downdet_img->SetDirection(up_img->GetDirection());


        StructImageType::Pointer upfindiff_img= StructImageType::New();
        upfindiff_img->SetRegions(up_img->GetLargestPossibleRegion());
        upfindiff_img->Allocate();
        upfindiff_img->SetSpacing(up_img->GetSpacing());
        upfindiff_img->SetOrigin(up_img->GetOrigin());
        upfindiff_img->SetDirection(up_img->GetDirection());

        StructImageType::Pointer downfindiff_img= StructImageType::New();
        downfindiff_img->SetRegions(up_img->GetLargestPossibleRegion());
        downfindiff_img->Allocate();
        downfindiff_img->SetSpacing(up_img->GetSpacing());
        downfindiff_img->SetOrigin(up_img->GetOrigin());
        downfindiff_img->SetDirection(up_img->GetDirection());



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
                    {
                        double det= ComputeSingleJacobianMatrixAtIndex(def_FINV,index,1,phase,phase_xyz);
                        if(det <=-1)
                            det=-1+1E-5;
                        updet_img->SetPixel(index,(1+det)*up_img->GetPixel(index));
                        downdet_img->SetPixel(index,(1-det)*down_img->GetPixel(index));
                    }
                }
            }
        }


        #pragma omp parallel for collapse(2)
        for( int k=0; k<(int)imsize[2];k++)
        {
            for(unsigned int j=0; j<imsize[1];j++)
            {
                ImageType3D::IndexType index;
                index[2]=k;
                index[1]=j;

                for(unsigned int i=0; i<imsize[0];i++)
                {
                    index[0]=i;

                    FinDiff upvals= ComputeFinDiff(index,updet_img,str_img);
                    FinDiff downvals= ComputeFinDiff(index,downdet_img,str_img);
                    upfindiff_img->SetPixel(index,upvals);
                    downfindiff_img->SetPixel(index,downvals);
                }
            }
        }



        #pragma omp parallel for
        for( int k=0; k<(int)imsize[2];k++)
        {
            ImageType3D::IndexType index;
            index[2]=k;
            for(unsigned int j=0; j<imsize[1];j++)
            {
                index[1]=j;

                DisplacementFieldType::PixelType updateFINV,updateMINV ;
                for(unsigned int i=0; i<imsize[0];i++)
                {
                    index[0]=i;

                    float mv1= ComputeUpdateCCJacSSingle(index,
                                                   up_img,  up_img_gradient,
                                                   upfindiff_img,
                                                   def_FINV,
                                                   current_Gaussian_operator,
                                                   updateFINV,
                                                   phase,phase_xyz,new_phase_vector);

                    float mv2= ComputeUpdateCCJacSSingle(index,
                                                   down_img,  down_img_gradient,
                                                   downfindiff_img,
                                                   def_MINV,
                                                   current_Gaussian_operator,
                                                   updateMINV,
                                                   phase,phase_xyz,new_phase_vector);

                    updateFieldINV->SetPixel(index,updateFINV-updateMINV);
                    m_MetricImage->SetPixel(index,mv1+mv2);
                }
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









#endif
