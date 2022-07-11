#ifndef _MK_DISPLACEMENTMAPS_H
#define _MK_DISPLACEMENTMAPS_H


#include "init_iso_gw.h"
#include "cmath"

#include "defines.h"




//#include "gradcal.h"
//#include "iso_gw.h"


struct basis1{
    vnl_vector<double>  xbasis;
    vnl_vector<double>  ybasis;
    vnl_vector<double>  zbasis;
};


vnl_matrix <double> mkbasis(vnl_matrix<double> smat,
                            ImageType3D::Pointer b0_image,
                            std::vector<int>keyelements,
                            std::vector<double>coefelement,
                            bool laplace = 1,
                            int order =2){
    ImageType3D::SizeType sizes = b0_image->GetLargestPossibleRegion().GetSize();
    int nIndex = sizes[0] * sizes[1] * sizes[2];
    itk::ImageRegionIteratorWithIndex<ImageType3D> imageIterator(b0_image, b0_image->GetLargestPossibleRegion());

    ImageType3D::IndexType index;
    imageIterator.GoToBegin();
    vnl_vector <double> inds4 (4,1);
    vnl_vector <double> physCoor(4);

    int nterms = keyelements.size()/2;
    int ncoefs = coefelement.size();
    bool nocoef = 0;
    vnl_matrix<double> basis;

    basis= vnl_matrix<double>(nIndex,1,0);

    int countID = 0;
    if (laplace){

        int ll,mm=0;

        while(!imageIterator.IsAtEnd()){
            index = imageIterator.GetIndex();

            inds4[0] = index[0];
            inds4[1] = index[1];
            inds4[2] = index[2];
            inds4[3] = 1;

            physCoor = smat * inds4;
            vnl_vector <double>xyz_r = physCoor/250;
            double x1 = xyz_r[0];
            double y1 = xyz_r[1];
            double z1 = xyz_r[2];

           /*mkbasis*/
            double rr = std::sqrt(x1*x1 + y1 *y1 + z1*z1);
            double phi = std::atan2(y1,x1);
            double zz = z1;
            double czz = 0.0;
            double temp = 0;

            double tempbasis = 0;
            if (zz != 0){
                czz = zz/rr;
            }
            for (int nn = 0; nn < nterms ; nn++){
                ll = keyelements.at(2*nn);
                mm = keyelements.at(2*nn+1);
                if (coefelement.at(nn) ==0){
                    continue;
                }
                if (ll == 0)
                    tempbasis +=coefelement.at(nn);
                else{
                    temp = std::pow(rr,ll);
                    tempbasis += coefelement.at(nn)*temp * sphericalz(ll,mm,phi,czz,0);
                }
            }
            basis(countID,0) = tempbasis;
            ++imageIterator;
            countID ++;
    }

    }
    return basis;
}



DisplacementFieldType::Pointer mk_displacement_siemens(GradCoef &E,ImageType3D::Pointer b0_image )
{

    vnl_matrix_fixed <double,4,4> smat;


    /* Read inputs */
    ImageType3D::SizeType sizes= b0_image->GetLargestPossibleRegion().GetSize();
    long npixels = (long)sizes[0] * sizes[1] * sizes[2];


    /* Getting Transformation matrix from header to transform to Phys Coor.*/

    ImageType3D::DirectionType dir = b0_image->GetDirection();
    ImageType3D::SpacingType spc = b0_image->GetSpacing();
    ImageType3D::PointType origin = b0_image->GetOrigin();


    vnl_matrix_fixed<double,3,3> spc_mat; spc_mat.fill(0);
    spc_mat(0,0)=spc[0];spc_mat(1,1)=spc[1]; spc_mat(2,2)=spc[2];

    vnl_matrix_fixed<double,4,4> itk_ijk_to_xyz_matrix;
    itk_ijk_to_xyz_matrix.set_identity();
    itk_ijk_to_xyz_matrix.update(dir.GetVnlMatrix()*spc_mat,0,0);
    itk_ijk_to_xyz_matrix(0,3)=origin[0];
    itk_ijk_to_xyz_matrix(1,3)=origin[1];
    itk_ijk_to_xyz_matrix(2,3)=origin[2];

    vnl_matrix_fixed<double,4,4> itk_to_RAS_transformation;
    itk_to_RAS_transformation.set_identity();
    itk_to_RAS_transformation(0,0)=-1;
    itk_to_RAS_transformation(1,1)=-1;

    vnl_matrix_fixed<double,4,4> RAS_to_LAI_transformation;
    RAS_to_LAI_transformation.set_identity();
    RAS_to_LAI_transformation(0,0)=-1;
    RAS_to_LAI_transformation(2,2)=-1;

    vnl_matrix_fixed<double,3,3> xyz_to_LPS_trans;
    xyz_to_LPS_trans.set_identity();
    xyz_to_LPS_trans(0,0)=-1;
    xyz_to_LPS_trans(1,1)=-1;
    //xyz_to_LPS_trans(2,2)=-1;


    vnl_matrix_fixed<double,4,4> ijk_to_xyz_transformation=   itk_to_RAS_transformation* itk_ijk_to_xyz_matrix;

    ImageType3D::Pointer R_img= ImageType3D::New();
    R_img->SetRegions(b0_image->GetLargestPossibleRegion());
    R_img->Allocate();
    R_img->SetSpacing(spc);
    R_img->SetOrigin(origin);
    R_img->SetDirection(dir);
    R_img->FillBuffer(0);

    ImageType3D::Pointer theta_img= ImageType3D::New();
    theta_img->SetRegions(b0_image->GetLargestPossibleRegion());
    theta_img->Allocate();
    theta_img->SetSpacing(spc);
    theta_img->SetOrigin(origin);
    theta_img->SetDirection(dir);
    theta_img->FillBuffer(0);

    ImageType3D::Pointer phi_img= ImageType3D::New();
    phi_img->SetRegions(b0_image->GetLargestPossibleRegion());
    phi_img->Allocate();
    phi_img->SetSpacing(spc);
    phi_img->SetOrigin(origin);
    phi_img->SetDirection(dir);
    phi_img->FillBuffer(0);


    for(int k=0;k<sizes[2];k++)
    {
        vnl_vector<double> ijk(4);
        DisplacementFieldType::IndexType ind3;
        ind3[2]=k;
        ijk[3]=1;
        ijk[2]=k;
        for(int j=0;j<sizes[1];j++)
        {
            ijk[1]=j;
            ind3[1]=j;
            for(int i=0;i<sizes[0];i++)
            {
                ijk[0]=i;
                ind3[0]=i;

                vnl_vector<double> xyz= ijk_to_xyz_transformation * ijk;
                xyz[0]+=0.0001;

                double r= sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2]);
                double theta= std::acos(xyz[2]/r);
                double phi= std::atan2(xyz[1]/r,xyz[0]/r);

                R_img->SetPixel(ind3,r);
                theta_img->SetPixel(ind3,theta);
                phi_img->SetPixel(ind3,phi);
            }
        }
    }

    DisplacementFieldType::Pointer fieldImage= DisplacementFieldType::New();
    fieldImage->SetRegions(b0_image->GetLargestPossibleRegion());
    fieldImage->Allocate();
    fieldImage->SetDirection(b0_image->GetDirection());
    fieldImage->SetSpacing(b0_image->GetSpacing());
    fieldImage->SetOrigin(b0_image->GetOrigin());



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

        #pragma omp parallel for
        for(int k=0;k<sizes[2];k++)
        {
            DisplacementFieldType::IndexType ind3;
            ind3[2]=k;
            for(int j=0;j<sizes[1];j++)
            {
                ind3[1]=j;
                for(int i=0;i<sizes[0];i++)
                {
                    ind3[0]=i;


                    double b=0;
                    for(int l=0;l<=lmax;l++)
                    {
                        double f= pow(R_img->GetPixel(ind3)/E.R0,l);
                        for(int m=0;m<=l;m++)
                        {
                            if(alpha(l,m)==0 && beta(l,m)==0)
                                continue;

                            double f2=0;
                            if(alpha(l,m)!=0)
                                f2= alpha(l,m) * std::cos(m*phi_img->GetPixel(ind3));
                            if(beta(l,m)!=0)
                                f2+= beta(l,m) * std::sin(m*phi_img->GetPixel(ind3));

                            double ptemp = plgndr(l, m, std::cos(theta_img->GetPixel(ind3)));
                            double nrmfact=1.;
                            if(m>0)
                            {
                                nrmfact= pow(-1,m) * sqrt(double((2 * l + 1) * factorial(l - m)) / double(2. * factorial(l + m)));
                            }
                            double p= nrmfact*ptemp;
                            b+=f*p*f2;
                        }
                    }
                    DisplacementFieldType::PixelType vec = fieldImage->GetPixel(ind3);
                    vec[0]=E.R0*b;
                    fieldImage->SetPixel(ind3,vec);
                }
            }
        }
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


        #pragma omp parallel for
        for(int k=0;k<sizes[2];k++)
        {
            DisplacementFieldType::IndexType ind3;
            ind3[2]=k;
            for(int j=0;j<sizes[1];j++)
            {
                ind3[1]=j;
                for(int i=0;i<sizes[0];i++)
                {
                    ind3[0]=i;


                    double b=0;
                    for(int l=0;l<=lmax;l++)
                    {
                        double f= pow(R_img->GetPixel(ind3)/E.R0,l);
                        for(int m=0;m<=l;m++)
                        {
                            if(alpha(l,m)==0 && beta(l,m)==0)
                                continue;

                            double f2=0;
                            if(alpha(l,m)!=0)
                                f2= alpha(l,m) * std::cos(m*phi_img->GetPixel(ind3));
                            if(beta(l,m)!=0)
                                f2+= beta(l,m) * std::sin(m*phi_img->GetPixel(ind3));

                            double ptemp = plgndr(l, m, std::cos(theta_img->GetPixel(ind3)));
                            double nrmfact=1.;
                            if(m>0)
                            {
                                nrmfact= pow(-1,m) * sqrt(double((2 * l + 1) * factorial(l - m)) / double(2. * factorial(l + m)));
                            }
                            double p= nrmfact*ptemp;
                            b+=f*p*f2;
                        }
                    }
                    DisplacementFieldType::PixelType vec = fieldImage->GetPixel(ind3);
                    vec[1]=E.R0*b;
                    fieldImage->SetPixel(ind3,vec);
                }
            }
        }
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

        #pragma omp parallel for
        for(int k=0;k<sizes[2];k++)
        {
            DisplacementFieldType::IndexType ind3;
            ind3[2]=k;
            for(int j=0;j<sizes[1];j++)
            {
                ind3[1]=j;
                for(int i=0;i<sizes[0];i++)
                {
                    ind3[0]=i;


                    double b=0;
                    for(int l=0;l<=lmax;l++)
                    {
                        double f= pow(R_img->GetPixel(ind3)/E.R0,l);
                        for(int m=0;m<=l;m++)
                        {
                            if(alpha(l,m)==0 && beta(l,m)==0)
                                continue;

                            double f2=0;
                            if(alpha(l,m)!=0)
                                f2= alpha(l,m) * std::cos(m*phi_img->GetPixel(ind3));
                            if(beta(l,m)!=0)
                                f2+= beta(l,m) * std::sin(m*phi_img->GetPixel(ind3));

                            double ptemp = plgndr(l, m, std::cos(theta_img->GetPixel(ind3)));
                            double nrmfact=1.;
                            if(m>0)
                            {
                                nrmfact= pow(-1,m) * sqrt(double((2 * l + 1) * factorial(l - m)) / double(2. * factorial(l + m)));
                            }
                            double p= nrmfact*ptemp;
                            b+=f*p*f2;
                        }
                    }
                    DisplacementFieldType::PixelType vec = fieldImage->GetPixel(ind3);
                    vec[2]=E.R0*b;


                    vnl_vector<double> vec2= xyz_to_LPS_trans * vec.GetVnlVector();
                  //  vec2=-vec2;
                    vec.SetVnlVector(vec2);

                    fieldImage->SetPixel(ind3,vec);
                }
            }
        }
    }

    return fieldImage;

}

DisplacementFieldType::Pointer mk_displacement_nonsiemens(GradCoef &E,ImageType3D::Pointer b0_image, bool is_GE )
{
       ImageType3D::DirectionType dir = b0_image->GetDirection();
       ImageType3D::SpacingType spc = b0_image->GetSpacing();
       ImageType3D::PointType origin = b0_image->GetOrigin();

       vnl_vector<double> orig_vec(3,0);
       orig_vec[0]=origin[0];
       orig_vec[1]=origin[1];
       orig_vec[2]=origin[2];

       vnl_matrix<double> dicom_to_it_transformation(3,3);
       dicom_to_it_transformation.set_identity();
       dicom_to_it_transformation(0,0)=-1;
       dicom_to_it_transformation(1,1)=-1;

       vnl_matrix<double> new_direction = dicom_to_it_transformation * dir.GetVnlMatrix();
       vnl_vector<double> new_orig_vec= dicom_to_it_transformation * orig_vec;
       vnl_matrix<double> spc_mat(3,3);
       spc_mat.fill(0);
       spc_mat(0,0)= spc[0];
       spc_mat(1,1)= spc[1];
       spc_mat(2,2)= spc[2];
       vnl_matrix<double> nmat= new_direction *spc_mat;

       vnl_matrix_fixed<double,4,4> smat;
       smat.set_identity();
       smat(0,0) = nmat(0,0);    smat(0,1) = nmat(0,1);     smat(0,2) = nmat(0,2);     smat(0,3) = new_orig_vec[0];
       smat(1,0) = nmat(1,0);    smat(1,1) = nmat(1,1);     smat(1,2) = nmat(1,2);     smat(1,3) = new_orig_vec[1];
       smat(2,0) = nmat(2,0);    smat(2,1) = nmat(2,1);     smat(2,2) = nmat(2,2);     smat(2,3) = new_orig_vec[2];
       if (is_GE){
           smat(2,3) = -((b0_image->GetLargestPossibleRegion().GetSize()[2]-1)*spc[2])/2.;
       }

       std::vector<int> keyElements;
       std::vector<double> coefElements;
       int nx,ny,nz = 0;
       int ncoefx, ncoefy,ncoefz = 0;
       double normtemp = 1;
       nx = E.Xkeys.size()/2;
       ny = E.Ykeys.size()/2;
       nz = E.Zkeys.size()/2;
       ncoefx = E.gradX_coef.size();
       ncoefy = E.gradY_coef.size();
       ncoefz = E.gradZ_coef.size();
       std::vector<double>xcoef = E.gradX_coef;
       std::vector<double>ycoef = E.gradY_coef;
       std::vector<double>zcoef = E.gradZ_coef;
       /* Computing the mkbasis part */
            /* Fine non-linear terms */
       for (int nn = 0; nn < nx ; nn++){
           if ((E.Xkeys.at(2*nn) == 1) & (E.Xkeys.at(2*nn+1) == 1)){
               normtemp = E.gradX_coef.at(nn);
               continue;
           }
           /*norm coefs */
           xcoef.at(nn) = xcoef.at(nn)/normtemp;
           coefElements.push_back(xcoef.at(nn)/normtemp);
           keyElements.push_back(E.Xkeys.at(2*nn));
           keyElements.push_back(E.Xkeys.at(2*nn+1));
       }

       vnl_matrix<double> xbasis = mkbasis(smat,b0_image,  keyElements,coefElements,1,2 );

       keyElements.clear();
       coefElements.clear();
       for (int nn = 0; nn < ny ; nn++){
           if ((E.Ykeys.at(2*nn) == 1) & (E.Ykeys.at(2*nn+1) == -1)){
               normtemp = E.gradY_coef.at(nn);
               continue;
           }
           /*norm coefs */
           coefElements.push_back(ycoef.at(nn)/normtemp);

           keyElements.push_back(E.Ykeys.at(2*nn));
           keyElements.push_back(E.Ykeys.at(2*nn+1));
       }
       vnl_matrix<double> ybasis = mkbasis(smat,b0_image,  keyElements,coefElements,1,2 );


       keyElements.clear();
       coefElements.clear();
       for (int nn = 0; nn < nz ; nn++){
           if ((E.Zkeys.at(2*nn) == 1) & (E.Zkeys.at(2*nn+1) == 0)){
               normtemp = E.gradZ_coef.at(nn);
               continue;
           }
           /*norm coefs */
           coefElements.push_back(zcoef.at(nn)/normtemp);
           keyElements.push_back(E.Zkeys.at(2*nn));
           keyElements.push_back(E.Zkeys.at(2*nn+1));
       }

       vnl_matrix<double>zbasis = mkbasis(smat,b0_image,  keyElements,coefElements,1,2 );
       basisFuntions Basis;
       Basis.xbasis.push_back(xbasis);
       Basis.ybasis.push_back(ybasis);
       Basis.zbasis.push_back(zbasis);


       DisplacementFieldType::Pointer fieldImage= DisplacementFieldType::New();
       fieldImage->SetRegions(b0_image->GetLargestPossibleRegion());
       fieldImage->Allocate();
       fieldImage->SetDirection(b0_image->GetDirection());
       fieldImage->SetSpacing(b0_image->GetSpacing());
       fieldImage->SetOrigin(b0_image->GetOrigin());


       itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(fieldImage,fieldImage->GetLargestPossibleRegion());
       it.GoToBegin();
       int countID =0;
       while(!it.IsAtEnd())
       {
           DisplacementFieldType::IndexType index= it.GetIndex();
           DisplacementFieldType::PixelType vec;
           vec[0]= E.R0* xbasis(countID,0);
           vec[1]= E.R0* ybasis(countID,0);
           vec[2]= E.R0* zbasis(countID,0);

           it.Set(vec);
           ++it;
           countID++;
       }

       return fieldImage;

}





DisplacementFieldType::Pointer mk_displacement(std::string gradFn,ImageType3D::Pointer b0_image, bool is_GE=0)
{

    GRADCAL *grads = new GRADCAL(gradFn);
    GradCoef E = grads->get_struct();

    if(E.gradType=="siemens")
        return mk_displacement_siemens(E,b0_image);
    else
        return mk_displacement_nonsiemens(E,b0_image,is_GE);

}


#endif
