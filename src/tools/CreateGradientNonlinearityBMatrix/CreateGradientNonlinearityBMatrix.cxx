#include "defines.h"
#include "CreateGradientNonlinearityBMatrix_parser.h"
#include "TORTOISE.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "rigid_register_images.h"

#include "../tools/gradnonlin/gradcal.h"
#include "../tools/gradnonlin/init_iso_gw.h"
#include "../tools/gradnonlin/mk_displacementMaps.h"
#include "../tools/DRTAMAS/DRTAMAS_utilities_cp.h"


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

    double x1 = point[0]/E.R0;
    double y1 = point[1]/E.R0;
    double z1 = point[2]/E.R0;

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

    /*
    axx= \del_f^x / \del x   ; axy= \del_f^y / \del x  ;  axz= \del_f^z / \del x
    ayx= \del_f^x / \del y   ; ayy= \del_f^y / \del y  ;  ayz= \del_f^z / \del y
    azx= \del_f^x / \del z   ; azy= \del_f^y / \del z  ;  azz= \del_f^z / \del z
    */

    InternalMatrixType trans_mat;
    trans_mat(0,0)=axx; trans_mat(0,1)=axy; trans_mat(0,2)=axz;
    trans_mat(1,0)=ayx; trans_mat(1,1)=ayy; trans_mat(1,2)=ayz;
    trans_mat(2,0)=azx; trans_mat(2,1)=azy; trans_mat(2,2)=azz;
    trans_mat=trans_mat.transpose();

    /*
    vnl_matrix_fixed<double,6,6> trans_mat;
    // bxx                      bxy                               bxz                             byy                         byz                             bzz
    trans_mat(0,0) =axx*axx    ;trans_mat(0,1) =axx*axy          ;trans_mat(0,2) =axx*axz         ;trans_mat(0,3) =axy*axy   ;trans_mat(0,4) =axy*axz         ;trans_mat(0,5) =axz*axz;
    trans_mat(1,0) =2*axx*ayx  ;trans_mat(1,1) =axx*ayy+axy+ayx  ;trans_mat(1,2) =axx*ayz+axz*ayx ;trans_mat(1,3) =2*axy*ayy ;trans_mat(1,4) =axy*ayz+ayy*axz ;trans_mat(1,5) =2*axz*ayz;
    trans_mat(2,0) =2*axx*azx  ;trans_mat(2,1) =axx*azy+axy*azx  ;trans_mat(2,2) =axx*azz+axz*azx ;trans_mat(2,3) =2*axy*azy ;trans_mat(2,4) =axy*azz+axz*azy ;trans_mat(2,5) =2*axz*azz;
    trans_mat(3,0) =ayx*ayx    ;trans_mat(3,1) =ayx*ayy          ;trans_mat(3,2) =ayx*ayz         ;trans_mat(3,3) =ayy*ayy   ;trans_mat(3,4) =ayy*ayz         ;trans_mat(3,5) =ayz*ayz;
    trans_mat(4,0) =2*ayx*azx  ;trans_mat(4,1) =ayx*azy+ayy*azx  ;trans_mat(4,2) =ayx*azz+ayz*azx ;trans_mat(4,3) =2*ayy*azy ;trans_mat(4,4) =ayy*azz+ayz*azy ;trans_mat(4,5) =2*ayz*azz;
    trans_mat(5,0) =azx*azx    ;trans_mat(5,1) =azx*azy          ;trans_mat(5,2) =azx*azz         ;trans_mat(5,3) =azy*azy   ;trans_mat(5,4) =azy*azz         ;trans_mat(5,5) =azz*azz;
    */

    return  trans_mat;
}


DisplacementFieldType::PixelType  GradWarpDispAtPoint(GradCoef &E,ImageType3D::PointType xyz )
{
    xyz[0]+=0.0001;
    double r= sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2]);
    double theta= std::acos(xyz[2]/r);
    double phi= std::atan2(xyz[1]/r,xyz[0]/r);

    DisplacementFieldType::PixelType vec;

    {
        int nx = E.Xkeys.size()/2;
        vnl_vector<double> ls(nx);
        vnl_vector<double> ms(nx);

        for(int i=0;i<nx;i++)
        {
            ls[i]= E.Xkeys[2*i];
            ms[i]= E.Xkeys[2*i+1];
        }

        int lmax= ls.max_value();
        vnl_matrix<double> alpha(lmax+1,lmax+1,0);
        vnl_matrix<double> beta(lmax+1,lmax+1,0);

        for(int i=0;i<nx;i++)
        {
            double l= ls[i];
            double m= ms[i];

            if(m >=0)
            {
                if(l==1 && m==1)
                    alpha(l,m)= E.gradX_coef[i]-1;
                else
                    alpha(l,m)= E.gradX_coef[i];
            }
            else
                beta(l,(int)fabs(m))= E.gradX_coef[i];
        }

        double b=0;
        for(int l=0;l<=lmax;l++)
        {
            double f= pow(r/E.R0,l);
            for(int m=0;m<=l;m++)
            {
                if(alpha(l,m)==0 && beta(l,m)==0)
                    continue;

                double f2=0;
                if(alpha(l,m)!=0)
                    f2= alpha(l,m) * std::cos(m*phi);
                if(beta(l,m)!=0)
                    f2+= beta(l,m) * std::sin(m*phi);

                double ptemp = plgndr(l, m, std::cos(theta));
                double nrmfact=1.;
                if(m>0)
                {
                    nrmfact= pow(-1,m) * sqrt(double((2 * l + 1) * factorial(l - m)) / double(2. * factorial(l + m)));
                }
                double p= nrmfact*ptemp;
                b+=f*p*f2;
            }
        }
        vec[0]=E.R0*b;
    }

    {
        int ny = E.Ykeys.size()/2;
        vnl_vector<double> ls(ny);
        vnl_vector<double> ms(ny);

        for(int i=0;i<ny;i++)
        {
            ls[i]= E.Ykeys[2*i];
            ms[i]= E.Ykeys[2*i+1];
        }

        int lmax= ls.max_value();
        vnl_matrix<double> alpha(lmax+1,lmax+1,0);
        vnl_matrix<double> beta(lmax+1,lmax+1,0);

        for(int i=0;i<ny;i++)
        {
            double l= ls[i];
            double m= ms[i];

            if(m >=0)
            {
                alpha(l,m)= E.gradY_coef[i];
            }
            else
            {
                if(l==1 && m==-1)
                    beta(l,(int)fabs(m))= E.gradY_coef[i]-1;
                else
                    beta(l,(int)fabs(m))= E.gradY_coef[i];
            }
        }

        double b=0;
        for(int l=0;l<=lmax;l++)
        {
            double f= pow(r/E.R0,l);
            for(int m=0;m<=l;m++)
            {
                if(alpha(l,m)==0 && beta(l,m)==0)
                    continue;

                double f2=0;
                if(alpha(l,m)!=0)
                    f2= alpha(l,m) * std::cos(m*phi);
                if(beta(l,m)!=0)
                    f2+= beta(l,m) * std::sin(m*phi);

                double ptemp = plgndr(l, m, std::cos(theta));
                double nrmfact=1.;
                if(m>0)
                {
                    nrmfact= pow(-1,m) * sqrt(double((2 * l + 1) * factorial(l - m)) / double(2. * factorial(l + m)));
                }
                double p= nrmfact*ptemp;
                b+=f*p*f2;
            }
        }
        vec[1]=E.R0*b;

    }

    {
        int nz = E.Zkeys.size()/2;
        vnl_vector<double> ls(nz);
        vnl_vector<double> ms(nz);

        for(int i=0;i<nz;i++)
        {
            ls[i]= E.Zkeys[2*i];
            ms[i]= E.Zkeys[2*i+1];
        }

        int lmax= ls.max_value();
        vnl_matrix<double> alpha(lmax+1,lmax+1,0);
        vnl_matrix<double> beta(lmax+1,lmax+1,0);

        for(int i=0;i<nz;i++)
        {
            double l= ls[i];
            double m= ms[i];

            if(m >=0)
            {
                if(l==1 && m==0)
                    alpha(l,m)= E.gradZ_coef[i]-1;
                else
                    alpha(l,m)= E.gradZ_coef[i];
            }
            else
                beta(l,(int)fabs(m))= E.gradZ_coef[i];
        }

        double b=0;
        for(int l=0;l<=lmax;l++)
        {
            double f= pow(r/E.R0,l);
            for(int m=0;m<=l;m++)
            {
                if(alpha(l,m)==0 && beta(l,m)==0)
                    continue;

                double f2=0;
                if(alpha(l,m)!=0)
                    f2= alpha(l,m) * std::cos(m*phi);
                if(beta(l,m)!=0)
                    f2+= beta(l,m) * std::sin(m*phi);

                double ptemp = plgndr(l, m, std::cos(theta));
                double nrmfact=1.;
                if(m>0)
                {
                    nrmfact= pow(-1,m) * sqrt(double((2 * l + 1) * factorial(l - m)) / double(2. * factorial(l + m)));
                }
                double p= nrmfact*ptemp;
                b+=f*p*f2;
            }
        }
        vec[2]=E.R0*b;
    }

    return vec;

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

    //for siemens
    InternalMatrixType lps2lai;lps2lai.set_identity();
    lps2lai(1,1)=-1;
    lps2lai(2,2)=-1;

    //for ge and philips
    InternalMatrixType lps2ras;lps2ras.set_identity();
    lps2ras(1,1)=-1;
    lps2ras(0,0)=-1;

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

    vnl_vector<double> norms(3,1);
    vnl_matrix_fixed<double,3,3> id_trans; id_trans.set_identity();


    //The next part is for debugging purposes. Numerical and analytical derivations should be identical.
    /*
    InternalMatrixType Lmat; Lmat.set_identity();
    ImageType3D::PointType  pt_scanner_space;
    pt_scanner_space.Fill(-100);

    Lmat= pixel_bmatrix(E,pt_scanner_space,norms);

    InternalMatrixType Lmat2; Lmat2.set_identity();

    double EPS=0.1;
    for(int d=0;d<3;d++)
    {
        pt_scanner_space[d]+=EPS;
        DisplacementFieldType::PixelType vec_p = GradWarpDispAtPoint(E,pt_scanner_space );
        pt_scanner_space[d]-=2*EPS;
        DisplacementFieldType::PixelType vec_m = GradWarpDispAtPoint(E,pt_scanner_space );
        pt_scanner_space[d]+=EPS;


        Lmat2(0,d)= (vec_p[0]-vec_m[0])/(2*EPS);
        Lmat2(1,d)= (vec_p[1]-vec_m[1])/(2*EPS);
        Lmat2(2,d)= (vec_p[2]-vec_m[2])/(2*EPS);
    }
    Lmat2=Lmat2+ id_trans;

    std::cout<<"Lmat: "<<std::endl;
    std::cout<<Lmat<<std::endl<<std::endl;
    std::cout<<"Lmat2: "<<std::endl;
    std::cout<<Lmat2<<std::endl<<std::endl;
    int ma=0;
    */



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

                InternalMatrixType Lmat; Lmat.set_identity();
                if(grads)
                {
                    ImageType3D::PointType pt_itk_space, pt_scanner_space;
                    first_vol_grad->TransformContinuousIndexToPhysicalPoint(cind,pt_itk_space);
                    // pt_itk_space is in ITK xyz coordinate system


                    if(E.gradType=="siemens")
                    {
                        auto pt_scanner_vnl = lps2lai * pt_itk_space.GetVnlVector();
                        pt_scanner_space[0]= pt_scanner_vnl[0];
                        pt_scanner_space[1]= pt_scanner_vnl[1];
                        pt_scanner_space[2]= pt_scanner_vnl[2];

                        //Lmat is in whatever scanner coordinate space
                        Lmat= pixel_bmatrix(E,pt_scanner_space,norms);

                        // to ITK
                        Lmat = lps2lai * Lmat * lps2lai;
                    }
                    else
                    {
                        auto pt_scanner_vnl = lps2ras * pt_itk_space.GetVnlVector();
                        pt_scanner_space[0]= pt_scanner_vnl[0];
                        pt_scanner_space[1]= pt_scanner_vnl[1];
                        pt_scanner_space[2]= pt_scanner_vnl[2];

                        //Lmat is in whatever scanner coordinate space
                        Lmat= pixel_bmatrix(E,pt_scanner_space,norms);

                        // to ITK
                        Lmat = lps2ras * Lmat * lps2ras;
                    }


                    //What pixel_bmatrix outputs is the backward Jacobian
                    //Let's make it forward Jacobian
                    //I know this is wrong, it sohuldnt be transpose but inverse.
                    //everyone does it this way though
                    Lmat=Lmat.transpose();

                    // now let's transform Lmat to ijk space  of the native space image
                    Lmat = first_vol_grad->GetDirection().GetTranspose() *
                           Lmat  *
                           first_vol_grad->GetDirection().GetVnlMatrix();

                }

                if(rigid_trans)
                {                    
                    Lmat= first_vol_grad->GetDirection().GetVnlMatrix() *Lmat * first_vol_grad->GetDirection().GetTranspose();
                    Lmat= rigid_trans->GetMatrix().GetTranspose() *Lmat * rigid_trans->GetMatrix().GetVnlMatrix();
                    Lmat = final_b0->GetDirection().GetTranspose() *Lmat* final_b0->GetDirection().GetVnlMatrix();
                }

                //convert to HCP format ordering of tensor elements
                Lmat=Lmat.transpose();

                Limg[0]->SetPixel(ind3,Lmat(0,0));
                Limg[1]->SetPixel(ind3,Lmat(0,1));
                Limg[2]->SetPixel(ind3,Lmat(0,2));
                Limg[3]->SetPixel(ind3,Lmat(1,0));
                Limg[4]->SetPixel(ind3,Lmat(1,1));
                Limg[5]->SetPixel(ind3,Lmat(1,2));
                Limg[6]->SetPixel(ind3,Lmat(2,0));
                Limg[7]->SetPixel(ind3,Lmat(2,1));
                Limg[8]->SetPixel(ind3,Lmat(2,2));


            } //for i
        } //for j
    } //for k



    delete grads;
    return Limg;
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

                    A=ComputeJacobian(gw_field,jac_ind3);  //A is in ITK xyz space


                    // I really think the following is WRONG..
                    // I think it should be the inverse, NOT transpose
                    // but everyone does it this way so I will do it this way too.
                    //A= vnl_matrix_inverse<double>(A);

                    //A is the backward Jacobian
                    //Let's make it forward Jacobian
                    //I know this is wrong, it sohuldnt be transpose but inverse.
                    //everyone does it this way though
                    A=A.transpose();

                    // now let's transform A to ijk space  of the native space image
                    A= first_vol->GetDirection().GetTranspose() *  A  * first_vol->GetDirection().GetVnlMatrix();
                }

                if(rigid_trans)
                {
                    A= first_vol->GetDirection().GetVnlMatrix() *A;
                    A= rigid_trans->GetMatrix().GetTranspose() *A;
                    A = final_b0->GetDirection().GetTranspose() *A;
                }
                //convert to HCP format ordering of tensor elements
                A=A.transpose();

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



    ImageType3D::Pointer final_b0 = readImageD<ImageType3D>(final_img_name);
    ImageType3D::Pointer initial_b0=final_b0;
    if(initial_img_name!="")
    {
        initial_b0 = readImageD<ImageType3D>(initial_img_name);
        std::cout<<"Registering the initial b=0 to the final b=0 image."<<std::endl;
        rigid_trans= RigidRegisterImages(final_b0,initial_b0);
    }


    std::string outname= final_img_name.substr(0,final_img_name.rfind(".nii"))+"_graddev";
    std::vector<ImageType3D::Pointer> graddev_vbmat_img;
    DisplacementFieldType::Pointer gradwarp_field_inv=nullptr;
    if(nonlinearity_file_name.find(".nii")!=std::string::npos)
    {
        gradwarp_field_inv = readImageD<DisplacementFieldType>(nonlinearity_file_name);
        graddev_vbmat_img= ComputeLImgFromField(final_b0,initial_b0,rigid_trans,gradwarp_field_inv,isGE);
        outname=outname + "_f.nii";
    }
    else
    {
        gradwarp_field_inv= mk_displacement(nonlinearity_file_name,initial_b0,isGE);
        graddev_vbmat_img= ComputeLImgFromCoeffs(final_b0,initial_b0,rigid_trans,nonlinearity_file_name,isGE);
        outname=outname + "_c.nii";
    }

    std::string gradwarp_name= final_img_name.substr(0,final_img_name.rfind(".nii"))+"_gradwarp_field.nii";
    writeImageD<DisplacementFieldType>(gradwarp_field_inv,gradwarp_name);



    for(int v=0;v<9;v++)
    {
        write_3D_image_to_4D_file<float>(graddev_vbmat_img[v],outname,v,9);
    }
               
}


