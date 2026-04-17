#include <iostream>
#include <fstream>
using namespace std;

#include "defines.h"
#include "DRTAMAS_utilities_cp.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

typedef itk::MatrixOffsetTransformBase<double, 3,      3> AffineTransformType;
using JacobianImageType = itk::Image< vnl_matrix_fixed<double,3,3> , 3>;



ImageType3D::Pointer  ExtractJacobianComponent(JacobianImageType::Pointer JAC_img,int row,int col)
{
    ImageType3D::Pointer img=ImageType3D::New();
    img->SetRegions(JAC_img->GetLargestPossibleRegion());
    img->Allocate();
    img->SetDirection(JAC_img->GetDirection());
    img->SetOrigin(JAC_img->GetOrigin());
    img->SetSpacing(JAC_img->GetSpacing());

    ImageType3D::SpacingType spc = JAC_img->GetSpacing();
    ImageType3D::DirectionType dir = JAC_img->GetDirection();

    vnl_matrix_fixed<double,3,3> SD;
    SD(0,0)=dir(0,0)/spc[0]; SD(0,1)=dir(1,0)/spc[0]; SD(0,2)=dir(2,0)/spc[0];
    SD(1,0)=dir(0,1)/spc[1]; SD(1,1)=dir(1,1)/spc[1]; SD(1,2)=dir(2,1)/spc[1];
    SD(2,0)=dir(0,2)/spc[2]; SD(2,1)=dir(1,2)/spc[2]; SD(2,2)=dir(2,2)/spc[2];

    vnl_svd_fixed<double, 3, 3> svd_SD(SD);
    vnl_matrix_fixed<double,3,3> SD_inv = svd_SD.inverse();


    vnl_matrix_fixed<double,3,3> I; I.set_identity();


    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img,img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        JacobianImageType::PixelType jac= JAC_img->GetPixel(ind3);

        JacobianImageType::PixelType A = (jac-I) * SD_inv;
        //JacobianImageType::PixelType A = (jac-I) ;

        it.Set(A(row,col));
    }

    return img;
}


ImageType3D::Pointer  SpatialIntegrateImg(ImageType3D::Pointer der_img, int dim)
{
    ImageType3D::Pointer img=ImageType3D::New();
    img->SetRegions(der_img->GetLargestPossibleRegion());
    img->Allocate();
    img->SetDirection(der_img->GetDirection());
    img->SetOrigin(der_img->GetOrigin());
    img->SetSpacing(der_img->GetSpacing());
    img->FillBuffer(0);

    ImageType3D::IndexType ind3;
    ImageType3D::SizeType sz = img->GetLargestPossibleRegion().GetSize();

    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::SizeType des_sz;
    des_sz[0]=sz[0];
    des_sz[1]=sz[1];
    des_sz[2]=sz[2];

    start[dim]=1;
    des_sz[dim]=1;
    ImageType3D::RegionType des_reg(start,des_sz);

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img,des_reg);
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType curr_start_ind3= it.GetIndex();

        for(int i=0;i<sz[dim]-2;i++)
        {
            ImageType3D::IndexType curr_jac_ind= curr_start_ind3;
            ImageType3D::IndexType curr_ip_ind= curr_start_ind3;
            ImageType3D::IndexType curr_im_ind= curr_start_ind3;

            curr_jac_ind[dim]= curr_start_ind3[dim]+i;
            curr_ip_ind[dim]= curr_start_ind3[dim]+i+1;
            curr_im_ind[dim]= curr_start_ind3[dim]+i-1;

            double new_val= 2.* der_img->GetPixel(curr_jac_ind) + img->GetPixel(curr_im_ind);
            img->SetPixel(curr_ip_ind,new_val);
        }
    }

    return img;
}


ImageType3D::Pointer SpatialDerivative(ImageType3D::Pointer Aimg,int dim)
{
    ImageType3D::Pointer img=ImageType3D::New();
    img->SetRegions(Aimg->GetLargestPossibleRegion());
    img->Allocate();
    img->SetDirection(Aimg->GetDirection());
    img->SetOrigin(Aimg->GetOrigin());
    img->SetSpacing(Aimg->GetSpacing());
    img->FillBuffer(0);

    ImageType3D::SizeType sz = Aimg->GetLargestPossibleRegion().GetSize();

    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::SizeType des_sz;
    des_sz[0]=sz[0];
    des_sz[1]=sz[1];
    des_sz[2]=sz[2];

    start[dim]=1;
    des_sz[dim]=sz[dim]-2;
    ImageType3D::RegionType des_reg(start,des_sz);

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img,des_reg);
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        ImageType3D::IndexType ind3_p=ind3;
        ImageType3D::IndexType ind3_m=ind3;

        ind3_p[dim]= ind3[dim]+1;
        ind3_m[dim]= ind3[dim]-1;

        float val = 0.5*(Aimg->GetPixel(ind3_p)-Aimg->GetPixel(ind3_m));
        img->SetPixel(ind3,val);
    }

    return img;
}

template <class ImageType>
typename ImageType::Pointer  AddImages(typename  ImageType::Pointer im1, typename ImageType::Pointer im2 , float mult=1)
{
    typename ImageType::Pointer img=ImageType::New();
    img->SetRegions(im1->GetLargestPossibleRegion());
    img->Allocate();
    img->SetDirection(im1->GetDirection());
    img->SetOrigin(im1->GetOrigin());
    img->SetSpacing(im1->GetSpacing());


    itk::ImageRegionIteratorWithIndex<ImageType> it(img,img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        typename  ImageType::IndexType ind3= it.GetIndex();
        typename ImageType::PixelType val = im1->GetPixel(ind3) + im2->GetPixel(ind3)*mult;
        it.Set(val);
    }

    return img;
}


vnl_matrix_fixed<double,3,3>  AverageAffinesReturnInv(std::vector<vnl_matrix_fixed<double,3,3> >& all_affines)
{
    int N= all_affines.size();

    Eigen::Matrix3d avg_eigen_mat = Eigen::Matrix3d::Zero();
    for(int i=0;i<N;i++)
    {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> curr_eigen_mat(
            all_affines[i].data_block(),
            all_affines[i].rows(),
            all_affines[i].cols()
            );


        Eigen::Matrix3d L =curr_eigen_mat.log();
        avg_eigen_mat= avg_eigen_mat +L;
    }
    avg_eigen_mat=avg_eigen_mat/N;
    avg_eigen_mat= avg_eigen_mat.exp();

    Eigen::Matrix3d avg_eigen_mat_inv = avg_eigen_mat.inverse();

    vnl_matrix_fixed<double,3,3> mat;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat(i, j) = avg_eigen_mat_inv(i, j);
        }
    }
    return mat;
}

ImageType3D::Pointer  MaskDilate(ImageType3D::Pointer template_img)
{
    ImageType3D::Pointer mask_img=ImageType3D::New();
    mask_img->SetRegions(template_img->GetLargestPossibleRegion());
    mask_img->Allocate();
    mask_img->SetDirection(template_img->GetDirection());
    mask_img->SetSpacing(template_img->GetSpacing());
    mask_img->SetOrigin(template_img->GetOrigin());
    mask_img->FillBuffer(0);

    int rad=1;
    ImageType3D::SizeType sz = mask_img->GetLargestPossibleRegion().GetSize();

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(mask_img,mask_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        if(template_img->GetPixel(ind3)>1)
        {
            ImageType3D::IndexType tind;
            for(int k=ind3[2]-rad;k<=ind3[2]+rad;k++)
            {
                tind[2]=k;
                if(tind[2]<0)
                    tind[2]=0;
                if(tind[2]>sz[2]-1)
                    tind[2]=sz[2]-1;

                for(int j=ind3[1]-rad;j<=ind3[1]+rad;j++)
                {
                    tind[1]=j;
                    if(tind[1]<0)
                        tind[1]=0;
                    if(tind[1]>sz[1]-1)
                        tind[1]=sz[1]-1;
                    for(int i=ind3[0]-rad;i<=ind3[0]+rad;i++)
                    {
                        tind[0]=i;
                        if(tind[0]<0)
                            tind[0]=0;
                        if(tind[0]>sz[0]-1)
                            tind[0]=sz[0]-1;
                        mask_img->SetPixel(tind,1);
                    }
                }
            }
        }
    }
    return mask_img;
}

int main( int argc , char * argv[] )
{
    if(argc<2)
    {
        std::cout<<"Usage:   ConstrainDefFields   full_path_to_textfile_containing_list_of_deformation_fields full_path_to_tensor_template "<<std::endl;
        return EXIT_FAILURE;
    }

    ifstream inFile(argv[1]);
    if (!inFile)
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return EXIT_FAILURE;
    }

    ifstream inFile2(argv[2]);
    if (!inFile2)
    {
        cerr << "Template file " << argv[2] << " not found." << endl;
        return EXIT_FAILURE;
    }

    ImageType3D::Pointer template_img= readImageD<ImageType3D>(argv[2]);
    ImageType3D::Pointer mask_img= MaskDilate(template_img);    

    std::vector<DisplacementFieldType::Pointer> fields;

    std::string currdir;
    std::string nm(argv[1]);
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else
        currdir= nm.substr(0,mypos+1);

    std::vector<std::string> filenames;


    std::cout<<"Reading fields..."<<std::endl;
    string line;
    while (getline(inFile, line))
    {
        if (line.empty())
            continue;

        std::string file_name=line;
        FILE * fp= fopen(file_name.c_str(),"rb");

        if(!fp)
        {
            file_name= currdir + file_name;

            FILE * fp2= fopen(file_name.c_str(),"rb");
            if(!fp2)
            {
                std::cout<< "File " << line << " does not exist. Exiting!" << std::endl;
                return 0;
            }
            else
                fclose(fp2);
        }
        else
            fclose(fp);


        DisplacementFieldType::Pointer curr_field= readImageD<DisplacementFieldType>(file_name);
        fields.push_back(curr_field);
        filenames.push_back(file_name);
    }
    inFile.close();
    std::cout<<"Done reading fields!"<<std::endl;

    JacobianImageType::Pointer avg_affine_inv_img=JacobianImageType::New();
    avg_affine_inv_img->SetRegions(fields[0]->GetLargestPossibleRegion());
    avg_affine_inv_img->Allocate();
    avg_affine_inv_img->SetDirection(fields[0]->GetDirection());
    avg_affine_inv_img->SetOrigin(fields[0]->GetOrigin());
    avg_affine_inv_img->SetSpacing(fields[0]->GetSpacing());
    JacobianImageType::PixelType id;
    id.set_identity();
    avg_affine_inv_img->FillBuffer(id);

    std::cout<<"Computing voxelwise average Jacobian image..."<<std::endl;

    {
        itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(fields[0],fields[0]->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            DisplacementFieldType::IndexType ind3= it.GetIndex();
            std::vector< vnl_matrix_fixed<double,3,3> > all_affines;
            for(int v=0;v<fields.size();v++)
            {
                vnl_matrix_fixed<double,3,3> curr_JAC= ComputeJacobian(fields[v],ind3);
                all_affines.push_back(curr_JAC);
            }

            vnl_matrix_fixed<double,3,3> avg_affine_inv = AverageAffinesReturnInv(all_affines);
            avg_affine_inv_img->SetPixel(ind3,avg_affine_inv);
        }
    }


    std::cout<<"Done computing voxelwise average Jacobian image!"<<std::endl;




    for(int v=0;v<fields.size();v++)
    {
        DisplacementFieldType::Pointer curr_field = fields[v];

        JacobianImageType::Pointer curr_new_JAC_img=JacobianImageType::New();
        curr_new_JAC_img->SetRegions(curr_field->GetLargestPossibleRegion());
        curr_new_JAC_img->Allocate();
        curr_new_JAC_img->SetDirection(curr_field->GetDirection());
        curr_new_JAC_img->SetOrigin(curr_field->GetOrigin());
        curr_new_JAC_img->SetSpacing(curr_field->GetSpacing());

        itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(curr_field,curr_field->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            DisplacementFieldType::IndexType ind3= it.GetIndex();

            vnl_matrix_fixed<double,3,3> curr_JAC= ComputeJacobian(curr_field,ind3);
            auto avg_inv_voxel= avg_affine_inv_img->GetPixel(ind3);

            vnl_svd_fixed<double,3,3> mysvd(curr_JAC);
            vnl_matrix_fixed<double,3,3> new_JAC = mysvd.U() * mysvd.W() * avg_inv_voxel * mysvd.V().transpose();

//            vnl_matrix_fixed<double,3,3> new_JAC = curr_JAC *avg_inv_voxel ;
            curr_new_JAC_img->SetPixel(ind3, new_JAC);
        }


        DisplacementFieldType::Pointer new_field= DisplacementFieldType::New();
        new_field->SetRegions(curr_field->GetLargestPossibleRegion());
        new_field->Allocate();
        new_field->SetDirection(curr_field->GetDirection());
        new_field->SetOrigin(curr_field->GetOrigin());
        new_field->SetSpacing(curr_field->GetSpacing());
        DisplacementFieldType::PixelType zero; zero.Fill(0);
        new_field->FillBuffer(zero);

        for(int vcomp=0;vcomp<3;vcomp++)
        {
           // ImageType3D::Pointer jac_comp_img_i = ExtractJacobianComponent(curr_new_JAC_img,vcomp,vcomp);
          //  ImageType3D::Pointer new_field_comp = SpatialIntegrateImg(jac_comp_img_i,vcomp);

           //int int_dim_0=vcomp;
           //int int_dim_1= (vcomp+1)%3;
           //int int_dim_2= (vcomp+2)%3;

           int int_dim_0=0;
           int int_dim_1= 1;
           int int_dim_2= 2;


           ImageType3D::Pointer jac_comp_img_i = ExtractJacobianComponent(curr_new_JAC_img,vcomp,int_dim_0);
           ImageType3D::Pointer fxyz = SpatialIntegrateImg(jac_comp_img_i,int_dim_0);

           ImageType3D::Pointer del_f1xyz_dely = SpatialDerivative(fxyz,int_dim_1);
           ImageType3D::Pointer jac_comp_img_j = ExtractJacobianComponent(curr_new_JAC_img,vcomp,int_dim_1);
           ImageType3D::Pointer del_gyz_dely = AddImages<ImageType3D>(jac_comp_img_j,del_f1xyz_dely,-1);
           ImageType3D::Pointer gyz = SpatialIntegrateImg(del_gyz_dely,int_dim_1);


           ImageType3D::Pointer jac_comp_img_k = ExtractJacobianComponent(curr_new_JAC_img,vcomp,int_dim_2);
           ImageType3D::Pointer del_fxyz_delz= SpatialDerivative(fxyz,int_dim_2);
           ImageType3D::Pointer del_gyz_delz= SpatialDerivative(gyz,int_dim_2);

           ImageType3D::Pointer temp = AddImages<ImageType3D>(jac_comp_img_k,del_fxyz_delz,-1);
           ImageType3D::Pointer del_hz_delz=AddImages<ImageType3D>(temp,del_gyz_delz,-1);
           ImageType3D::Pointer hz= SpatialIntegrateImg(del_hz_delz,int_dim_2);


           //writeImageD<ImageType3D> (fxyz,"/qmi08_raid/rakibul/DOWNS_data/DTI/T4/population_template_cont29pat15_T4/okan_test/smaller/0fxyz.nii");
           //writeImageD<ImageType3D> (gyz,"/qmi08_raid/rakibul/DOWNS_data/DTI/T4/population_template_cont29pat15_T4/okan_test/smaller/0gyz.nii");
           //writeImageD<ImageType3D> (hz,"/qmi08_raid/rakibul/DOWNS_data/DTI/T4/population_template_cont29pat15_T4/okan_test/smaller/0hz.nii");
           //exit(1);

           gyz=AddImages<ImageType3D>(gyz,hz);
           fxyz=AddImages<ImageType3D>(fxyz,gyz);

            itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it2(curr_field,curr_field->GetLargestPossibleRegion());
            for(it2.GoToBegin();!it2.IsAtEnd();++it2)
            {
                DisplacementFieldType::IndexType ind3= it2.GetIndex();
                DisplacementFieldType::PixelType vec= new_field->GetPixel(ind3);
                vec[vcomp]= fxyz->GetPixel(ind3);
                new_field->SetPixel(ind3,vec);
            }
        }
/*

        {
            itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it3(curr_field,curr_field->GetLargestPossibleRegion());
            for(it3.GoToBegin();!it3.IsAtEnd();++it3)
            {
                DisplacementFieldType::IndexType ind3= it3.GetIndex();
                if(!mask_img->GetPixel(ind3))
                {
                    new_field->SetPixel(ind3,it3.Get());
                }
            }
        }
*/

        std::string oname = filenames[v].substr(0,filenames[v].rfind(".nii"))+"_cnstr.nii";
        writeImageD<DisplacementFieldType>(new_field,oname);
    }

    return EXIT_SUCCESS;
}
