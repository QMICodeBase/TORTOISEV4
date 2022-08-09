#include "defines.h"
#include "CreateGradientNonlinearityBMatrix_parser.h"
#include "TORTOISE.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "rigid_register_images.h"

#include "../tools/gradnonlin/gradcal.h"
#include "../tools/gradnonlin/init_iso_gw.h"
#include "../tools/gradnonlin/mk_displacementMaps.h"


InternalMatrixType  pixel_bmatrix(const GradCoef &E, ImageType3D::PointType point,const vnl_vector<double> &norms)
{
    std::vector <int> xkeys = E.Xkeys;
    std::vector <int> ykeys = E.Ykeys;
    std::vector <int> zkeys = E.Zkeys;

    int nxkey, nykey, nzkey, na;
    nxkey = xkeys.size()/2;
    nykey = ykeys.size()/2;
    nzkey = zkeys.size()/2;
    na = nxkey + nykey + nzkey;

    double x1 = point[0]/250.;
    double y1 = point[1]/250.;
    double z1 = point[2]/250.;

    double rr = std::sqrt(x1*x1 + y1 *y1 + z1*z1);
    double phi = std::atan2(y1,x1);
    double zz = z1/rr;
    double temp = 0;

    double axx = 0, ayy =0, azz = 0;
    double axy =0, azy =0;
    double ayx =0, azx =0;
    double axz =0, ayz =0;

    int ll, mm =0;

    /* X gradients */
    for (int kk =0; kk < nxkey; kk ++){
        if (E.gradX_coef.at(kk) == 0)
            continue;
        temp = E.gradX_coef.at(kk);
        ll = E.Xkeys.at(2*kk);
        mm = E.Xkeys.at(2*kk+1);

        int mma =abs(mm);
        if(E.gradType=="siemens" && mma>0 && ll>1)
        {
            temp/= sqrt(2 * factorial(ll-mma) / factorial(ll+mma)) ;
            temp*=  sqrt(double((2 * ll + 1) * factorial(ll - mma)) / double(2. * factorial(ll + mma)));
        }

        axx+= temp * dshdx('X', ll, mm, rr, phi, zz,0)/norms(0);
        ayx+= temp * dshdx('Y', ll, mm, rr, phi, zz,0)/norms(0);
        azx+= temp * dshdx('Z', ll, mm, rr, phi, zz,0)/norms(0);
    }
    /* Y gradients */
    for (int kk =0; kk < nykey; kk ++){
        temp = E.gradY_coef.at(kk);
        if (temp == 0)
            continue;
        ll = E.Ykeys.at(2*kk);
        mm = E.Ykeys.at(2*kk+1);
        int mma =abs(mm);
        if(E.gradType=="siemens" && mma>0 && ll>1)
        {
            temp/= sqrt(2 * factorial(ll-mma) / factorial(ll+mma)) ;
            temp*=  sqrt(double((2 * ll + 1) * factorial(ll - mma)) / double(2. * factorial(ll + mma)));
        }

        axy+= temp * dshdx('X',ll, mm, rr,phi,zz,0 )/norms(1);
        ayy+= temp * dshdx('Y',ll, mm, rr, phi, zz,0)/norms(1);
        azy+= temp * dshdx('Z', ll, mm, rr, phi,zz,0)/norms(1);
    }
    /* Z gradients */
    for (int kk =0; kk < nzkey; kk ++){
        temp = E.gradZ_coef.at(kk);
        if (temp == 0)
            continue;
        ll = E.Zkeys.at(2*kk);
        mm = E.Zkeys.at(2*kk+1);

        int mma =abs(mm);
        if(E.gradType=="siemens" && mma>0 && ll>1)
        {
            temp/= sqrt(2 * factorial(ll-mma) / factorial(ll+mma)) ;
            temp*=  sqrt(double((2 * ll + 1) * factorial(ll - mma)) / double(2. * factorial(ll + mma)));
        }

        axz+= temp * dshdx('X',ll, mm, rr,phi,zz,0 )/norms(2);
        ayz+= temp * dshdx('Y',ll, mm, rr, phi, zz,0)/norms(2);
        azz+= temp * dshdx('Z', ll, mm, rr, phi,zz,0)/norms(2);
    }


    InternalMatrixType trans_mat;
    trans_mat(0,0)=axx; trans_mat(0,1)=axy; trans_mat(0,2)=axz;
    trans_mat(1,0)=ayx; trans_mat(1,1)=ayy; trans_mat(1,2)=ayz;
    trans_mat(2,0)=azx; trans_mat(2,1)=azy; trans_mat(2,2)=azz;
    trans_mat=trans_mat.transpose();

    return  trans_mat;

}


std::vector<ImageType3D::Pointer>
ComputeLImgFromCoeffs(ImageType3D::Pointer final_b0, ImageType3D::Pointer initial_b0, TORTOISE::OkanQuadraticTransformType ::Pointer rigid_trans, std::string coeffs_file, bool is_GE)
{
    GRADCAL *grads =new GRADCAL(coeffs_file);
    GradCoef E= grads->get_struct();


    ImageType3D::Pointer first_vol_grad=initial_b0;
    ImageType3D::Pointer first_vol=initial_b0;

    if(!initial_b0)
    {
        first_vol_grad= final_b0;
        first_vol= final_b0;
    }

    vnl_matrix<double> dicom_to_it_transformation(3,3);
    dicom_to_it_transformation.set_identity();
    dicom_to_it_transformation(0,0)=-1;
    dicom_to_it_transformation(1,1)=-1;


    if (is_GE)
    {
        using DupType = itk::ImageDuplicator<ImageType3D>;
        DupType::Pointer dup = DupType::New();
        dup->SetInputImage(first_vol);
        dup->Update();
        first_vol_grad=dup->GetOutput();

        vnl_matrix<double> spc_mat(3,3);  spc_mat.fill(0);
        spc_mat(0,0)= first_vol->GetSpacing()[0];
        spc_mat(1,1)= first_vol->GetSpacing()[1];
        spc_mat(2,2)= first_vol->GetSpacing()[2];

        vnl_vector<double> indv(3);
        indv[0]= (first_vol->GetLargestPossibleRegion().GetSize()[0] -1)/2.;
        indv[1]= (first_vol->GetLargestPossibleRegion().GetSize()[1] -1)/2.;
        indv[2]= (first_vol->GetLargestPossibleRegion().GetSize()[2] -1)/2.;

        vnl_vector<double>  new_origv= -first_vol->GetDirection().GetVnlMatrix() *spc_mat * indv;
        ImageType3D::PointType new_orig;
        new_orig[0]=first_vol->GetOrigin()[0];
        new_orig[1]=first_vol->GetOrigin()[1];
        new_orig[2]=new_origv[2];
        first_vol_grad->SetOrigin(new_orig);
    }


    std::vector<ImageType3D::Pointer> Limg;
    Limg.resize(9);
    for(int v=0;v<9;v++)
    {
        Limg[v]=ImageType3D::New();
        Limg[v]->SetRegions(final_b0->GetLargestPossibleRegion());
        Limg[v]->Allocate();
        Limg[v]->SetDirection(final_b0->GetDirection());
        Limg[v]->SetSpacing(final_b0->GetSpacing());
        Limg[v]->SetOrigin(final_b0->GetOrigin());
        Limg[v]->FillBuffer(0);
    }

    ImageType3D::SizeType sz= final_b0->GetLargestPossibleRegion().GetSize();

    vnl_matrix_fixed<double,3,3> RAS_to_LPS; RAS_to_LPS.set_identity();
    RAS_to_LPS(0,0)=-1;
    RAS_to_LPS(1,1)=1;
    vnl_matrix<double> flip_mat= final_b0->GetDirection().GetVnlMatrix() * first_vol_grad->GetDirection().GetVnlMatrix().transpose();

    vnl_vector<double> norms(3,1);
    vnl_matrix_fixed<double,3,3> id_trans; id_trans.set_identity();


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

                // Get the point the b0 physical space
                ImageType3D::PointType pt,pt_trans;
                final_b0->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(rigid_trans)
                    pt_trans=rigid_trans->TransformPoint(pt);

                itk::ContinuousIndex<double,3> cind;
                first_vol->TransformPhysicalPointToContinuousIndex(pt_trans,cind);

                InternalMatrixType RotMat; RotMat.set_identity();
                if(grads)
                {
                    ImageType3D::PointType pt_scanner_space;
                    first_vol_grad->TransformContinuousIndexToPhysicalPoint(cind,pt_scanner_space);
                    vnl_vector<double> temp=  dicom_to_it_transformation * pt_scanner_space.GetVnlVector();
                    pt_scanner_space[0]=temp[0];
                    pt_scanner_space[1]=temp[1];
                    pt_scanner_space[2]=temp[2];
                    // This is the final physical point in NIFTI coordinate system

                    RotMat= pixel_bmatrix(E,pt_scanner_space,norms);
                    RotMat=RAS_to_LPS.transpose() * RotMat  * RAS_to_LPS;
                }

                if(rigid_trans)
                {
                    vnl_matrix<double> dirmat= first_vol->GetDirection().GetVnlMatrix();
                    vnl_matrix_fixed<double,3,3> rotmat2= rigid_trans->GetMatrix().GetVnlMatrix();

                    RotMat= rotmat2 * RotMat * rotmat2.transpose();
                }

                Limg[0]->SetPixel(ind3,RotMat(0,0)-1);
                Limg[1]->SetPixel(ind3,RotMat(0,1));
                Limg[2]->SetPixel(ind3,RotMat(0,2));
                Limg[3]->SetPixel(ind3,RotMat(1,0));
                Limg[4]->SetPixel(ind3,RotMat(1,1)-1);
                Limg[5]->SetPixel(ind3,RotMat(1,2));
                Limg[6]->SetPixel(ind3,RotMat(2,0));
                Limg[7]->SetPixel(ind3,RotMat(2,1));
                Limg[8]->SetPixel(ind3,RotMat(2,2)-1);


            } //for i
        } //for j
    } //for k

    delete grads;
    return Limg;
}



InternalMatrixType ComputeJacobianAtIndex(DisplacementFieldType::Pointer disp_field, DisplacementFieldType::IndexType index)
{
    InternalMatrixType A;
    A.fill(0);

    for(int d=0;d<3;d++)
        if(index[d]<=0 || index[d]>= disp_field->GetLargestPossibleRegion().GetSize()[d]-1)
            return A;

    DisplacementFieldType::IndexType Nind=index;
    for(int dim=0;dim<3;dim++)   // derivative w.r.t.
    {
        DisplacementFieldType::PixelType val,val1,val2;
        Nind[dim]++;
        val1 = disp_field->GetPixel(Nind);
        disp_field->TransformPhysicalVectorToLocalVector(val1,val);
        Nind[dim]-=2;
        val1=disp_field->GetPixel(Nind);
        disp_field->TransformPhysicalVectorToLocalVector(val1,val2);
        val-=val2;
        Nind[dim]++;
        val*=0.5/disp_field->GetSpacing()[dim];

        A.set_column(dim,val.GetVnlVector());
    }
    return A;
}


std::vector<ImageType3D::Pointer> ComputeLImgFromField(ImageType3D::Pointer final_b0, ImageType3D::Pointer initial_b0, TORTOISE::OkanQuadraticTransformType::Pointer rigid_trans, DisplacementFieldType::Pointer gw_field, bool is_GE)
{
    ImageType3D::Pointer first_vol_grad=initial_b0;
    ImageType3D::Pointer first_vol=initial_b0;

    if(!initial_b0)
    {
        first_vol_grad= final_b0;
        first_vol= final_b0;
    }

    vnl_matrix<double> dicom_to_it_transformation(3,3);
    dicom_to_it_transformation.set_identity();
    dicom_to_it_transformation(0,0)=-1;
    dicom_to_it_transformation(1,1)=-1;

    vnl_matrix_fixed<double,3,3> RAS_to_LPS; RAS_to_LPS.set_identity();
    RAS_to_LPS(0,0)=-1;
    RAS_to_LPS(1,1)=1;

    std::vector<ImageType3D::Pointer> Limg;
    Limg.resize(9);
    for(int v=0;v<9;v++)
    {
        Limg[v]=ImageType3D::New();
        Limg[v]->SetRegions(final_b0->GetLargestPossibleRegion());
        Limg[v]->Allocate();
        Limg[v]->SetDirection(final_b0->GetDirection());
        Limg[v]->SetSpacing(final_b0->GetSpacing());
        Limg[v]->SetOrigin(final_b0->GetOrigin());
        Limg[v]->FillBuffer(0);
    }

    ImageType3D::SizeType sz= final_b0->GetLargestPossibleRegion().GetSize();


    vnl_matrix<double> flip_mat= final_b0->GetDirection().GetVnlMatrix() * first_vol->GetDirection().GetVnlMatrix().transpose();
    vnl_vector<double> norms(3,1);
    vnl_matrix_fixed<double,3,3> id_trans; id_trans.set_identity();

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

                // Get the point the b0 physical space
                ImageType3D::PointType pt,pt_trans;
                final_b0->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(rigid_trans)
                    pt_trans=rigid_trans->TransformPoint(pt);


                itk::ContinuousIndex<double,3> cind;
                first_vol->TransformPhysicalPointToContinuousIndex(pt_trans,cind);

                InternalMatrixType A; A.set_identity();
                if(gw_field)
                {
                    gw_field->TransformPhysicalPointToContinuousIndex(pt_trans,cind);
                    ImageType3D::IndexType jac_ind3;

                    for(int yy=0;yy<3;yy++)
                    {
                        if(cind[yy]<0)
                            jac_ind3[yy]=0;
                        else
                            jac_ind3[yy]= (unsigned int)std::round(cind[yy]);
                    }

                    A= ComputeJacobianAtIndex(gw_field,jac_ind3);
                    //A is now in IJK space
                    A= gw_field->GetDirection().GetVnlMatrix() * A * gw_field->GetDirection().GetTranspose();
                    //A is now in ITK XYZ space
                    A= dicom_to_it_transformation * A * dicom_to_it_transformation.transpose();
                    //A is now in NIFTI XYZ space
                    A=RAS_to_LPS.transpose() * A  * RAS_to_LPS;
                }

                if(rigid_trans)
                {
                    vnl_matrix<double> dirmat= first_vol->GetDirection().GetVnlMatrix();
                    vnl_matrix_fixed<double,3,3> rotmat2= rigid_trans->GetMatrix().GetVnlMatrix();

                    A= rotmat2 * A * rotmat2.transpose();
                }

                Limg[0]->SetPixel(ind3,A(0,0));
                Limg[1]->SetPixel(ind3,A(0,1));
                Limg[2]->SetPixel(ind3,A(0,2));
                Limg[3]->SetPixel(ind3,A(1,0));
                Limg[4]->SetPixel(ind3,A(1,1));
                Limg[5]->SetPixel(ind3,A(1,2));
                Limg[6]->SetPixel(ind3,A(2,0));
                Limg[7]->SetPixel(ind3,A(2,1));
                Limg[8]->SetPixel(ind3,A(2,2));

            } //for i
        } //for j
    } //for k

    return Limg;


}





int main(int argc, char*argv[])
{
    CreateGradientNonlinearityBMatrix_PARSER* parser = new     CreateGradientNonlinearityBMatrix_PARSER( argc , argv );
    
    
    std::string final_img_name = parser->getFinalImageName();
    std::string initial_img_name = parser->getInitialImageName();
    bool isGE= parser->getIsGE();
    std::string nonlinearity_file_name = parser->getNonlinearity();


    using OkanQuadraticTransformType=TORTOISE::OkanQuadraticTransformType;
    OkanQuadraticTransformType::Pointer rigid_trans= OkanQuadraticTransformType::New();
    rigid_trans->SetPhase("vertical");
    rigid_trans->SetIdentity();


    ImageType3D::Pointer initial_b0=nullptr;
    ImageType3D::Pointer final_b0 = readImageD<ImageType3D>(final_img_name);
    if(initial_img_name!="")
    {
        initial_b0 = readImageD<ImageType3D>(initial_img_name);
        std::cout<<"Registering the initial b=0 to the final b=0 image."<<std::endl;
        rigid_trans= RigidRegisterImages(final_b0,initial_b0);
    }


    std::vector<ImageType3D::Pointer> graddev_vbmat_img;
    DisplacementFieldType::Pointer gradwarp_field_inv=nullptr;
    if(nonlinearity_file_name.find(".nii")!=std::string::npos)
    {
        gradwarp_field_inv = readImageD<DisplacementFieldType>(nonlinearity_file_name);
        graddev_vbmat_img= ComputeLImgFromField(final_b0,initial_b0,rigid_trans,gradwarp_field_inv,isGE);
    }
    else
    {
        gradwarp_field_inv= mk_displacement(nonlinearity_file_name,initial_b0,isGE);
        graddev_vbmat_img= ComputeLImgFromCoeffs(final_b0,initial_b0,rigid_trans,nonlinearity_file_name,isGE);
    }

    std::string gradwarp_name= final_img_name.substr(0,final_img_name.rfind(".nii"))+"_gradwarp_field.nii";
    writeImageD<DisplacementFieldType>(gradwarp_field_inv,gradwarp_name);




    std::string outname= final_img_name.substr(0,final_img_name.rfind(".nii"))+"_graddev.nii";
    for(int v=0;v<9;v++)
    {
        write_3D_image_to_4D_file<float>(graddev_vbmat_img[v],outname,v,9);
    }
               
}


