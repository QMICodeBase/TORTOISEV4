#ifndef _DRBUDDIBase_CXX
#define _DRBUDDIBase_CXX


#include "DRBUDDIBase.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "registration_settings.h"

#include "../tools/EstimateTensor/estimate_tensor_wlls.h"
#include "create_mask.h"
#include "../tools/ComputeFAMap/compute_fa_map.h"

#include "rigid_register_images.h"

#include "itkResampleImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkKdTreeGenerator.h"
#include "itkKdTree.h"
#include "itkListSample.h"
#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkTransformFileWriter.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "DRBUDDI_Diffeo.h"



void DRBUDDIBase::Process()
{
    Step0_CreateImages();
    Step1_RigidRegistration();
    Step2_DiffeoRegistration();
    Step3_WriteOutput();
}


void DRBUDDIBase::CreateBlipUpQuadImage()
{
    std::vector<std::string> str_names= parser->getStructuralNames();
    ImageType3D::SpacingType new_spacing;

    if(str_names.size())
    {
        ImageType3D::Pointer str_img = readImageD<ImageType3D>(str_names[0]);
        new_spacing = str_img->GetSpacing();
    }
    else
    {
        new_spacing= b0_up->GetSpacing();
    }
    while(new_spacing[0]>=1)
        new_spacing=new_spacing/1.5;

    ImageType3D::SizeType new_size;
    new_size[0] = (int)ceil(1.0*b0_up->GetLargestPossibleRegion().GetSize()[0] * b0_up->GetSpacing()[0]/new_spacing[0]);
    new_size[1] = (int)ceil(1.0*b0_up->GetLargestPossibleRegion().GetSize()[1] * b0_up->GetSpacing()[1]/new_spacing[1]);
    new_size[2] = (int)ceil(1.0*b0_up->GetLargestPossibleRegion().GetSize()[2] * b0_up->GetSpacing()[2]/new_spacing[2]);
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType nreg(start,new_size);

    ImageType3D::SpacingType new_spc;
    new_spc[0]=  1.0*b0_up->GetLargestPossibleRegion().GetSize()[0] * b0_up->GetSpacing()[0] / new_size[0];
    new_spc[1]=  1.0*b0_up->GetLargestPossibleRegion().GetSize()[1] * b0_up->GetSpacing()[1] / new_size[1];
    new_spc[2]=  1.0*b0_up->GetLargestPossibleRegion().GetSize()[2] * b0_up->GetSpacing()[2] / new_size[2];

    itk::ContinuousIndex<double,3> ind;
    ind[0]=-0.5;
    ind[1]=-0.5;
    ind[2]=-0.5;
    ImageType3D::PointType pt;
    this->b0_up->TransformContinuousIndexToPhysicalPoint(ind,pt);

    vnl_vector<double> vec(3);
    vec[0]= ind[0]* new_spc[0];
    vec[1]= ind[1]* new_spc[1];
    vec[2]= ind[2]* new_spc[2];
    vnl_vector<double> nvec= b0_up->GetDirection().GetVnlMatrix() * vec;
    ImageType3D::PointType new_orig;
    new_orig[0]=pt[0]-nvec[0];
    new_orig[1]=pt[1]-nvec[1];
    new_orig[2]=pt[2]-nvec[2];

    ImageType3D::Pointer b0_up_ref_img =  ImageType3D::New();
    b0_up_ref_img->SetDirection(b0_up->GetDirection());
    b0_up_ref_img->SetOrigin(new_orig);
    b0_up_ref_img->SetSpacing(new_spc);
    b0_up_ref_img->SetRegions(nreg);

    int pad=16;
    if(parser->getDisableInitRigid())
        pad=0;

    ind[0]=-pad/2;
    ind[1]=-pad/2;
    ind[2]=0;
    b0_up_ref_img->TransformContinuousIndexToPhysicalPoint(ind,pt);

    new_size[0]=new_size[0]+pad;
    new_size[1]=new_size[1]+pad;
    new_size[2]=new_size[2];
    nreg.SetSize(new_size);
    nreg.SetIndex(start);

    b0_up_ref_img =  ImageType3D::New();
    b0_up_ref_img->SetDirection(b0_up->GetDirection());
    b0_up_ref_img->SetOrigin(pt);
    b0_up_ref_img->SetSpacing(new_spc);
    b0_up_ref_img->SetRegions(nreg);

    itk::IdentityTransform<double,3>::Pointer  id=itk::IdentityTransform<double,3>::New();
    id->SetIdentity();

    using InterpolatorType= itk::BSplineInterpolateImageFunction<ImageType3D,double,double>;
    InterpolatorType::Pointer interp=InterpolatorType::New();
    interp->SetSplineOrder(3);


    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(b0_up_ref_img);
    resampleFilter2->SetInput(b0_up);
    resampleFilter2->SetTransform(id);
    resampleFilter2->SetInterpolator(interp);
    resampleFilter2->Update();
    this->b0_up_quad= resampleFilter2->GetOutput();



    itk::ImageRegionIterator<ImageType3D> it(this->b0_up_quad,this->b0_up_quad->GetLargestPossibleRegion());
    for(it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        if(it.Get()<0)
            it.Set(0);
    }
}

ImageType3D::Pointer DRBUDDIBase::JacobianTransformImage(ImageType3D::Pointer img,DisplacementFieldType::Pointer field,ImageType3D::Pointer ref_img)
{
    DisplacementFieldTransformType::Pointer disp_trans = DisplacementFieldTransformType::New();
    disp_trans->SetDisplacementField(field);

    using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType3D,double>;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(3);

    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(ref_img);
  //  resampleFilter2->SetInterpolator(interpolator);
    resampleFilter2->SetInput(img);
    resampleFilter2->SetTransform(disp_trans);
    resampleFilter2->Update();
    ImageType3D::Pointer trans_img= resampleFilter2->GetOutput();

    vnl_vector<double> phase_vector(3,0);
    if(this->PE_string=="vertical")
        phase_vector[1]=1;
    if(this->PE_string=="horizontal")
        phase_vector[0]=1;
    if(this->PE_string=="slice")
        phase_vector[2]=1;
    phase_vector= this->b0_up->GetDirection().GetVnlMatrix() * phase_vector;
    int phase_id=0;
    if(  (fabs(phase_vector[1])>fabs(phase_vector[0])) &&  (fabs(phase_vector[1])>fabs(phase_vector[0])) )
          phase_id=1;
    if(  (fabs(phase_vector[2])>fabs(phase_vector[0])) &&  (fabs(phase_vector[2])>fabs(phase_vector[1])) )
          phase_id=2;

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(trans_img, trans_img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType index = it.GetIndex();
        InternalMatrixType  A = ComputeJacobianAtIndex(field,index);
        //double det= vnl_det<double>(A);
        double det = A(phase_id,phase_id);
        if(det>0)
        {
          //  double logd = log(det);
          //  double ly = logd / (sqrt(1+0.2*logd*logd));
          //  det=exp(ly);
            it.Set(it.Get()*det);
        }
        else
        {
            ImageType3D::IndexType tind=index;
            int Ntot=0;
            double tot=0;
            for(int k=-1;k<=1;k++)
            {
                tind[2]=index[2]+k;
                for(int j=-2;j<=2;j++)
                {
                    tind[1]=index[1]+j;
                    for(int i=-2;i<=2;i++)
                    {
                        tind[0]=index[0]+i;
                        InternalMatrixType  AA = ComputeJacobianAtIndex(field,tind);
                        double det2 = AA(phase_id,phase_id);
                        if(det2>0)
                        {
                            Ntot++;
                            tot+=det2;
                        }
                    }
                }
            }
            if(Ntot>0)
            {
                det=tot/Ntot;
             //   double logd = log(det);
             //   double ly = logd / (sqrt(1+0.2*logd*logd));
             //   det=exp(ly);
                it.Set(it.Get()*det);
            }
            else
                it.Set(0);
        }
        ++it;
    }
    return trans_img;
}



InternalMatrixType DRBUDDIBase::ComputeJacobianAtIndex(DisplacementFieldType::Pointer disp_field, DisplacementFieldType::IndexType index)
{
    InternalMatrixType A;
    A.set_identity();

    if(index[0]<=0 || index[0]>= disp_field->GetLargestPossibleRegion().GetSize()[0]-1)
        return A;

    if(index[1]<=0 || index[1]>= disp_field->GetLargestPossibleRegion().GetSize()[1]-1)
        return A;

    if(index[2]<=0 || index[2]>= disp_field->GetLargestPossibleRegion().GetSize()[2]-1)
        return A;

    bool do_second_order=false;
    if(index[0]==1 || index[0]== disp_field->GetLargestPossibleRegion().GetSize()[0]-2)
        do_second_order=false;
    if(index[1]==1 || index[1]== disp_field->GetLargestPossibleRegion().GetSize()[1]-2)
        do_second_order=false;
    if(index[2]==1 || index[2]== disp_field->GetLargestPossibleRegion().GetSize()[2]-2)
        do_second_order=false;

    if(do_second_order)
    {
        for(int dim=0;dim<3;dim++)   // derivative w.r.t.
        {
            DisplacementFieldType::IndexType rind=index;
            DisplacementFieldType::IndexType lind=index;
            DisplacementFieldType::IndexType rrind=index;
            DisplacementFieldType::IndexType llind=index;
            rind[dim]++;
            lind[dim]--;
            rrind[dim]+=2;
            llind[dim]-=2;

            DisplacementFieldType::PixelType lval = disp_field->GetPixel(lind);
            DisplacementFieldType::PixelType rval = disp_field->GetPixel(rind);
            DisplacementFieldType::PixelType llval = disp_field->GetPixel(llind);
            DisplacementFieldType::PixelType rrval = disp_field->GetPixel(rrind);

            DisplacementFieldType::PixelType deriv= (-rrval+8.*rval-8.*lval+llval)/12./disp_field->GetSpacing()[dim];

            A.set_column(dim,deriv.GetVnlVector());
        }
    }
    else
    {
        for(int dim=0;dim<3;dim++)   // derivative w.r.t.
        {
            DisplacementFieldType::IndexType rind=index;
            DisplacementFieldType::IndexType lind=index;
            rind[dim]++;
            lind[dim]--;

            DisplacementFieldType::PixelType lval = disp_field->GetPixel(lind);
            DisplacementFieldType::PixelType rval = disp_field->GetPixel(rind);

            DisplacementFieldType::PixelType deriv= 0.5*(rval-lval)/disp_field->GetSpacing()[dim];

            A.set_column(dim,deriv.GetVnlVector());
        }
    }

    vnl_vector<double> phys_vec(3);
    phys_vec= disp_field->GetDirection().GetVnlMatrix()*A.get_row(0);
    A.set_row(0,phys_vec);
    phys_vec= disp_field->GetDirection().GetVnlMatrix()*A.get_row(1);
    A.set_row(1,phys_vec);
    phys_vec= disp_field->GetDirection().GetVnlMatrix()*A.get_row(2);
    A.set_row(2,phys_vec);
    A(0,0)+=1;
    A(1,1)+=1;
    A(2,2)+=1;


    return A;
}

void DRBUDDIBase::CreateCorrectionImage(std::string nii_filename,ImageType3D::Pointer &b0_img, ImageType3D::Pointer &FA_img)
{
    std::string bmtxt_name= nii_filename.substr(0,nii_filename.rfind(".nii"))+std::string(".bmtxt");
    vnl_matrix<double> Bmatrix= read_bmatrix_file(bmtxt_name);
    int Nvols= Bmatrix.rows();

     bool use_tensor=true;
     int ndwi=0;
     vnl_vector<double> bvals = Bmatrix.get_column(0)+Bmatrix.get_column(3)+Bmatrix.get_column(5);
     for(int v=0;v<bvals.size();v++)
         if(bvals[v]>200)
             ndwi++;
     if(ndwi<12)
         use_tensor=false;
     vnl_svd<double> msvd(Bmatrix);
     use_tensor=true;
     vnl_diag_matrix<double> W = msvd.W();
     for(int v=0;v<W.cols();v++)
     {
         if(W(v,v)<1E-50)
         {
             use_tensor=false;
             break;
         }
     }
     if(use_tensor)
     {
         (*stream)<<"TENSOR fitting data for b=0 image generation!!"<<std::endl;
     }
     else
     {
         (*stream)<<"NOT USING TENSOR fitting for b=0 image generation!!"<<std::endl;
     }



     std::string inc_name= nii_filename.substr(0,nii_filename.rfind(".nii"))+std::string("_inc.nii");
     std::vector<ImageType3DBool::Pointer> final_inclusion_imgs;
     std::vector<ImageType3D::Pointer> final_data;
     final_data.resize(Nvols);     
     if(Nvols==1)
     {
         final_data[0]=readImageD<ImageType3D>(nii_filename);
     }
     else
     {
         for(int v=0;v<Nvols;v++)
         {
             final_data[v]=read_3D_volume_from_4D(nii_filename,v);
         }
     }

     if(fs::exists(inc_name))
     {
         final_inclusion_imgs.resize(Nvols);
         for(int v=0;v<Nvols;v++)
         {
             final_inclusion_imgs[v]=read_3D_volume_from_4DBool(inc_name,v);
         }
     }



     if(use_tensor)
     {
         int DWI_bval = RegistrationSettings::get().getValue<int>("DRBUDDI_DWI_bval_tensor_fitting");

         std::vector<int> dummy;
         if(DWI_bval==0)
         {
             for(int i=0;i<bvals.size();i++)
             {
                 dummy.push_back(i);
             }
         }
         else
         {
             for(int i=0;i<bvals.size();i++)
             {
                 if(bvals[i]<= 1.05*DWI_bval)
                     dummy.push_back(i);
             }
         }

         //int b0_vol_id = my_json["B0VolId"];
         ImageType3D::Pointer mask_img= nullptr;
         if(main_mask_img)
             mask_img=main_mask_img;
         else
             mask_img= create_mask(final_data[0]);
         ImageType3D::Pointer A0_image=nullptr;
         DTImageType::Pointer  dt_image=nullptr;
         if(dummy.size()<15)
             dummy.resize(0);
         dt_image= EstimateTensorWLLS_sub_nomm(final_data,Bmatrix,dummy,A0_image,nullptr,final_inclusion_imgs);
         FA_img = compute_fa_map(dt_image);


         itk::ImageRegionIteratorWithIndex<ImageType3D> mit(A0_image,A0_image->GetLargestPossibleRegion());
         mit.GoToBegin();
         while(!mit.IsAtEnd())
         {
             ImageType3D::IndexType ind3= mit.GetIndex();
             float val =mit.Get();
             if(val!=val || val<0)
             {
                 mit.Set(0);
             }
             if(mask_img->GetPixel(ind3)==0)
                 FA_img->SetPixel(ind3,0);

             ++mit;
         }

         b0_img=A0_image;

     }
     else
     {
         FA_img=nullptr;

         float b0bval = bvals.min_value();
         std::vector<int> b0_indices;
         for(int v=0;v<bvals.size();v++)
         {
             if(bvals[v]<=1.05*b0bval)
             {
                 b0_indices.push_back(v);
             }
         }

         b0_img= final_data[b0_indices[0]];
         for(int v=1;v<b0_indices.size();v++)
         {
             ImageType3D::Pointer im2= final_data[b0_indices[v]];

             typedef itk::AddImageFilter<ImageType3D,ImageType3D,ImageType3D> AdderType;
             AdderType::Pointer adder= AdderType::New();
             adder->SetInput1(b0_img);
             adder->SetInput2(im2);
             adder->Update();
             b0_img= adder->GetOutput();
         }
         if(b0_indices.size()>1)
         {
             typedef itk::DivideImageFilter<ImageType3D,ImageType3D,ImageType3D> DividerType;
             DividerType::Pointer divider= DividerType::New();
             divider->SetInput1(b0_img);
             divider->SetConstant(1.*b0_indices.size());
             divider->Update();
             b0_img=divider->GetOutput();
         }

         itk::ImageRegionIterator<ImageType3D> it(b0_img,b0_img->GetLargestPossibleRegion());
         it.GoToBegin();
         while(!it.IsAtEnd())
         {
             if(it.Get()<0)
                 it.Set(0);
             ++it;
         }
     }
}




#endif

