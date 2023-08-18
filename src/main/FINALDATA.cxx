#ifndef _FINALDATA_CXX
#define _FINALDATA_CXX



#include "FINALDATA.h"

#include "registration_settings.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "rigid_register_images.h"
#include "../tools/ReorientImage/reorient_image.h"
#include "../utilities/read_bmatrix_file.h"
#include "../tools/RotateBMatrix/rotate_bmatrix.h"


#include "create_mask.h"
#include "../tools/EstimateMAPMRI/MAPMRIModel.h"
#include "../tools/EstimateTensor/DTIModel.h"
#include "../tools/TORTOISEBmatrixToFSLBVecs/tortoise_bmatrix_to_fsl_bvecs.h"
#include "../tools/gradnonlin/init_iso_gw.h"


#include "itkIdentityTransform.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkTransformFileReader.h"
#include "itkImageDuplicator.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkListSample.h"
#include "itkKdTreeGenerator.h"
#include "itkMultiplyImageFilter.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"


#include "../tools/DRTAMAS/DRTAMAS_utilities_cp.h"

#include "vnl/vnl_cross.h"

#include "../utilities/math_utilities.h"

#include "itkInvertDisplacementFieldImageFilterOkan.h"

#include <algorithm>

#include <Eigen/Dense>
using namespace Eigen;


ImageType3D::Pointer FINALDATA::UnObliqueImage(ImageType3D::Pointer img)
{
    ImageType3D::SizeType sz= img->GetLargestPossibleRegion().GetSize();
    ImageType3D::DirectionType orig_dir = img->GetDirection();
    ImageType3D::DirectionType new_dir;

    for(int r=0;r<3;r++)
    {
        for(int c=0;c<3;c++)
        {
            if(fabs(orig_dir(r,c))>0.5)
                new_dir(r,c)  =  sgn<double>(orig_dir(r,c));
            else
                new_dir(r,c)  =0;
        }
    }

    double mn[3],mx[3];
    int is[2]={0,(int)sz[0]-1};
    int js[2]={0,(int)sz[1]-1};
    int ks[2]={0,(int)sz[2]-1};

    for(int d=0;d<3;d++)
    {
        mx[d]=-1E10;
        mn[d]=1E10;
        for(int kk=0;kk<2;kk++)
        {
            ImageType3D::IndexType ind3;
            ind3[2]= ks[kk];
            for(int jj=0;jj<2;jj++)
            {
                ind3[1]= js[jj];
                for(int ii=0;ii<2;ii++)
                {
                    ind3[0]= is[ii];
                    ImageType3D::PointType pt;
                    img->TransformIndexToPhysicalPoint(ind3,pt);

                    if(pt[d]<mn[d])
                        mn[d]=pt[d];
                    if(pt[d]>mx[d])
                        mx[d]=pt[d];
                }
            }
        }
    }
    vnl_vector<double> FOV(3,0);
    FOV[0]= fabs(mx[0]-mn[0]);
    FOV[1]= fabs(mx[1]-mn[1]);
    FOV[2]= fabs(mx[2]-mn[2]);
    vnl_matrix_fixed<double,3,3> Sinv; Sinv.fill(0);
    Sinv(0,0)= 1./img->GetSpacing()[0];
    Sinv(1,1)= 1./img->GetSpacing()[1];
    Sinv(2,2)= 1./img->GetSpacing()[2];
    vnl_matrix_fixed<double,3,3> S; S.fill(0);
    S(0,0)= img->GetSpacing()[0];
    S(1,1)= img->GetSpacing()[1];
    S(2,2)= img->GetSpacing()[2];

    vnl_vector< double> new_sizes_v= Sinv * new_dir.GetTranspose()* FOV;
    ImageType3D::SizeType new_size;
    new_size[0] = (unsigned int)(std::ceil(fabs(new_sizes_v[0])));
    new_size[1] = (unsigned int)(std::ceil(fabs(new_sizes_v[1])));
    new_size[2] = (unsigned int)(std::ceil(fabs(new_sizes_v[2])));


    itk::ContinuousIndex<double,3> mid_ind;
    mid_ind[0]= ((double)(sz[0])-1)/2.;
    mid_ind[1]= ((double)(sz[1])-1)/2.;
    mid_ind[2]= ((double)(sz[2])-1)/2.;
    ImageType3D::PointType mid_pt;
    img->TransformContinuousIndexToPhysicalPoint(mid_ind,mid_pt);
    mid_ind[0]= ((double)(new_size[0])-1)/2.;
    mid_ind[1]= ((double)(new_size[1])-1)/2.;
    mid_ind[2]= ((double)(new_size[2])-1)/2.;

    vnl_vector<double> new_orig_v = mid_pt.GetVnlVector() - new_dir.GetVnlMatrix() * S * mid_ind.GetVnlVector();
    ImageType3D::PointType new_orig;
    new_orig[0]=new_orig_v[0];
    new_orig[1]=new_orig_v[1];
    new_orig[2]=new_orig_v[2];





    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,new_size);

    ImageType3D::Pointer ref_img= ImageType3D::New();
    ref_img->SetRegions(reg);
    ref_img->SetSpacing(img->GetSpacing());
    ref_img->SetDirection(new_dir);
    ref_img->SetOrigin(new_orig);

    using BSPInterPolatorType = itk::BSplineInterpolateImageFunction<ImageType3D,double>;
    BSPInterPolatorType::Pointer interp = BSPInterPolatorType::New();
    interp->SetSplineOrder(3);
    interp->SetInputImage(img);

    using IdTransformType = itk::IdentityTransform<double,3>;
    IdTransformType::Pointer id_trans= IdTransformType::New();
    id_trans->SetIdentity();

    using ResamplerType = itk::ResampleImageFilter<ImageType3D,ImageType3D>;
    ResamplerType::Pointer resampler= ResamplerType::New();
    resampler->SetDefaultPixelValue(0);
    resampler->SetInput(img);
    resampler->SetInterpolator(interp);
    resampler->SetOutputParametersFromImage(ref_img);
    resampler->SetTransform(id_trans);
    resampler->Update();
    ImageType3D::Pointer new_img= resampler->GetOutput();

    return new_img;
}



ImageType3D::Pointer FINALDATA::GenerateFirstStructural()
{
    std::vector<std::string>  structural_names = RegistrationSettings::get().getVectorValue<std::string>("structural");
    std::string reorientation_name= RegistrationSettings::get().getValue<std::string>("reorientation");

    std::string output_orientation= RegistrationSettings::get().getValue<std::string>("output_orientation");
    std::vector<float> final_res= RegistrationSettings::get().getVectorValue<float>("output_res");
    std::vector<int>  final_Nvoxels= RegistrationSettings::get().getVectorValue<int>("output_voxels");


    ImageType3D::Pointer target_img=nullptr;
    if(reorientation_name!="")
        target_img = readImageD<ImageType3D>(reorientation_name);
    else
    {
        if(structural_names.size())
            target_img = readImageD<ImageType3D>(structural_names[0]);
    }


    if(!target_img)
    {
        //if(fs::exists(this->temp_folder + std::string("/b0_corrected_final.nii")))
        //    target_img = readImageD<ImageType3D>(this->temp_folder + std::string("/b0_corrected_final.nii"));
        //else if(fs::exists(this->temp_folder + std::string("/blip_up_b0_corrected_JAC.nii")))
        //    target_img= readImageD<ImageType3D>(this->temp_folder + std::string("/blip_up_b0_corrected_JAC.nii"));
        //else
            target_img= read_3D_volume_from_4D(this->data_names[0],0);
    }

    if(output_orientation!="")
    {
        target_img= ReorientImage3D(target_img, "", output_orientation);
    }



/*
    vnl_matrix_fixed<double,3,3> str_dir= target_img->GetDirection().GetVnlMatrix();
    bool oblique=false;
    for(int r=0;r<3;r++)
        for(int c=0;c<3;c++)
            if( fabs(str_dir(r,c)) >0.0001 &&  fabs(fabs(str_dir(r,c))-1)>0.0001     )
            {
                oblique=true;
                break;
            }

    if(oblique)
    {
        target_img= UnObliqueImage(target_img);
        str_dir= target_img->GetDirection().GetVnlMatrix();
    }
*/

    /*
    if(output_orientation=="")
    {
        for(int d=0;d<3;d++)
        {
            if( (fabs(str_dir(d,0)) > fabs(str_dir(d,1))) &&  (fabs(str_dir(d,0)) > fabs(str_dir(d,2))) )
            {
                if(str_dir(d,0)>0)
                    output_orientation=output_orientation +"L";
                else
                    output_orientation=output_orientation +"R";
            }
            if( (fabs(str_dir(d,1)) > fabs(str_dir(d,0))) &&  (fabs(str_dir(d,1)) > fabs(str_dir(d,2))) )
            {
                if(str_dir(d,1)>0)
                    output_orientation=output_orientation +"P";
                else
                    output_orientation=output_orientation +"A";
            }
            if( (fabs(str_dir(d,2)) > fabs(str_dir(d,0))) &&  (fabs(str_dir(d,2)) > fabs(str_dir(d,1))) )
            {
                if(str_dir(d,2)>0)
                    output_orientation=output_orientation +"S";
                else
                    output_orientation=output_orientation +"I";
            }
        }
    }
*/



    ImageType3D::SizeType new_sz;
    ImageType3D::PointType new_orig;
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::SpacingType new_spc;


    if(final_res.size()!=3)
    {
        ImageType3D::Pointer first_vol= read_3D_volume_from_4D(this->data_names[0],0);
        ImageType3D::SpacingType spc= first_vol->GetSpacing();
        new_spc[0]=spc[0];
        new_spc[1]=spc[1];
        new_spc[2]=spc[2];
    }
    else
    {
        new_spc[0]=final_res[0];
        new_spc[1]=final_res[1];
        new_spc[2]=final_res[2];
    }


    if(final_Nvoxels.size()!=0)
    {
        itk::ContinuousIndex<double,3> mid_ind;
        ImageType3D::PointType mid_pt;
        mid_ind[0] = (1.*target_img->GetLargestPossibleRegion().GetSize()[0] -1)/2.;
        mid_ind[1] = (1.*target_img->GetLargestPossibleRegion().GetSize()[1] -1)/2.;
        mid_ind[2] = (1.*target_img->GetLargestPossibleRegion().GetSize()[2] -1)/2.;

        new_sz[0]= final_Nvoxels[0];
        new_sz[1]= final_Nvoxels[1];
        new_sz[2]= final_Nvoxels[2];

        target_img->TransformContinuousIndexToPhysicalPoint(mid_ind,mid_pt);

        vnl_matrix_fixed<double,3,3> spc_mat;
        spc_mat.fill(0);
        spc_mat(0,0)=new_spc[0];
        spc_mat(1,1)=new_spc[1];
        spc_mat(2,2)=new_spc[2];

        itk::ContinuousIndex<double,3> new_mid_ind;
        new_mid_ind[0]= (1.*new_sz[0] -1)/2.;
        new_mid_ind[1]= (1.*new_sz[1] -1)/2.;
        new_mid_ind[2]= (1.*new_sz[2] -1)/2.;

        vnl_vector<double> new_orig_vec= mid_pt.GetVnlVector() - target_img->GetDirection().GetVnlMatrix() * spc_mat * new_mid_ind.GetVnlVector();
        new_orig[0]=new_orig_vec[0];
        new_orig[1]=new_orig_vec[1];
        new_orig[2]=new_orig_vec[2];
    }
    else
    {
        // gotta find updated FOV, updated size, updated orig

        float FOV[3];
        FOV[0]=target_img->GetSpacing()[0]* target_img->GetLargestPossibleRegion().GetSize()[0];
        FOV[1]=target_img->GetSpacing()[1]* target_img->GetLargestPossibleRegion().GetSize()[1];
        FOV[2]=target_img->GetSpacing()[2]* target_img->GetLargestPossibleRegion().GetSize()[2];

        new_sz[0]= (int)std::round(FOV[0]/new_spc[0]);
        new_sz[1]= (int)std::round(FOV[1]/new_spc[1]);
        new_sz[2]= (int)std::round(FOV[2]/new_spc[2]);

        itk::ContinuousIndex<double,3> corner_ind;
        corner_ind.Fill(-0.5);
        ImageType3D::PointType corner_pt;
        target_img->TransformContinuousIndexToPhysicalPoint(corner_ind,corner_pt);
        vnl_matrix_fixed<double,3,3> spc_mat;
        spc_mat.fill(0);
        spc_mat(0,0)=new_spc[0];
        spc_mat(1,1)=new_spc[1];
        spc_mat(2,2)=new_spc[2];

        vnl_vector<double> new_orig_vec= corner_pt.GetVnlVector()- target_img->GetDirection().GetVnlMatrix()* spc_mat * corner_ind.GetVnlVector();
        new_orig[0]=new_orig_vec[0];
        new_orig[1]=new_orig_vec[1];
        new_orig[2]=new_orig_vec[2];
    }


    ImageType3D::RegionType reg(start,new_sz);
    ImageType3D::Pointer ref_img= ImageType3D::New();
    ref_img->SetRegions(reg);
    ref_img->SetDirection(target_img->GetDirection());
    ref_img->SetSpacing(new_spc);
    ref_img->SetOrigin(new_orig);


    typedef itk::IdentityTransform<double, 3> TransformType;
    TransformType::Pointer id_trans= TransformType::New();
    id_trans->SetIdentity();

    typedef itk::BSplineInterpolateImageFunction<ImageType3D, double, double> InterpolatorType;
    InterpolatorType::Pointer interp = InterpolatorType::New();
    interp->SetSplineOrder(3);

    typedef itk::ResampleImageFilter<ImageType3D, ImageType3D> ResamplerType;
    ResamplerType::Pointer resampler= ResamplerType::New();
    resampler->SetTransform(id_trans);
    resampler->SetInput(target_img);
    resampler->SetInterpolator(interp);
    resampler->SetOutputParametersFromImage(ref_img);
    resampler->Update();
    ImageType3D::Pointer structural_image= resampler->GetOutput();

    itk::ImageRegionIterator<ImageType3D> it(structural_image, structural_image->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::PixelType val= it.Get();
        if(val<0)
            it.Set(0);
        ++it;
    }

    return structural_image;
}


ImageType3D::Pointer FINALDATA::GenerateStructurals()
{
    std::string reorientation_name= RegistrationSettings::get().getValue<std::string>("reorientation");
    std::vector<std::string>  structural_names = RegistrationSettings::get().getVectorValue<std::string>("structural");

    std::vector<ImageType3D::Pointer> structurals;
    ImageType3D::Pointer first_structural_img= GenerateFirstStructural();
    structurals.push_back(first_structural_img);

    int str_start_id;
    if(reorientation_name=="")
        str_start_id=1;
    else
        str_start_id=0;


    for(int s=str_start_id;s<structural_names.size();s++)
    {
        ImageType3D::Pointer curr_str_img= readImageD<ImageType3D>(structural_names[s]);
        RigidTransformType::Pointer trans = RigidRegisterImagesEuler( first_structural_img,  curr_str_img,"MI",parser->getRigidLR());


        typedef itk::BSplineInterpolateImageFunction<ImageType3D, double, double> InterpolatorType;
        InterpolatorType::Pointer interp = InterpolatorType::New();
        interp->SetSplineOrder(3);

        typedef itk::ResampleImageFilter<ImageType3D, ImageType3D> ResamplerType;
        ResamplerType::Pointer resampler= ResamplerType::New();
        resampler->SetTransform(trans);
        resampler->SetInput(curr_str_img);
        resampler->SetInterpolator(interp);
        resampler->SetOutputParametersFromImage(first_structural_img);
        resampler->Update();
        ImageType3D::Pointer structural_image= resampler->GetOutput();
        itk::ImageRegionIterator<ImageType3D> it(structural_image, structural_image->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            ImageType3D::PixelType val= it.Get();
            if(val<0)
                it.Set(0);
            ++it;
        }

        structurals.push_back(structural_image);
    }

    std::string output_folder= fs::path(output_name).parent_path().string();
    if(output_folder=="")
        output_folder="./";

    for(int s=0;s<structurals.size();s++)
    {
        char nm[1000]={0};
        sprintf(nm,"%s/structural_%d.nii",output_folder.c_str(),s);
        writeImageD<ImageType3D>(structurals[s],nm);
    }
    return structurals[0];
}


void FINALDATA::ReadOrigTransforms()
{
    std::string output_gradnonlin_type = RegistrationSettings::get().getValue<std::string>("output_gradnonlin_Bmtxt_type");
    std::string epi_type = RegistrationSettings::get().getValue<std::string>("epi");
    std::string moteddy_type = RegistrationSettings::get().getValue<std::string>("correction_mode");
    bool s2v_type = RegistrationSettings::get().getValue<bool>("s2v");
    bool repol_type = RegistrationSettings::get().getValue<bool>("repol");
    std::string drift_type = RegistrationSettings::get().getValue<std::string>("drift");
    float THR=RegistrationSettings::get().getValue<float>("outlier_prob");


    if(fs::exists(this->temp_folder +"/b0_to_str_rigidtrans.hdf5") && parser->getStructuralNames().size()!=0)
    {
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(this->temp_folder +"/b0_to_str_rigidtrans.hdf5" );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();
        this->b0_t0_str_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
    }

    if(fs::exists(this->temp_folder +"/bdown_to_bup_rigidtrans.hdf5"))
    {
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(this->temp_folder +"/bdown_to_bup_rigidtrans.hdf5" );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();
        this->b0down_t0_b0up_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
    }

    if(fs::exists(this->temp_folder +"/deformation_FINV.nii.gz") && epi_type!="off")
    {
        this->epi_trans[0]=readImageD<DisplacementFieldType>(this->temp_folder +"/deformation_FINV.nii.gz");
    }
    if(fs::exists(this->temp_folder +"/deformation_MINV.nii.gz") && epi_type!="off")
    {
        this->epi_trans[1]=readImageD<DisplacementFieldType>(this->temp_folder +"/deformation_MINV.nii.gz");
    }

    std::string gradnonlin_field_name= RegistrationSettings::get().getValue<std::string>("grad_nonlin");
    std::string gradnonlin_name_inv = gradnonlin_field_name.substr(0,gradnonlin_field_name.rfind(".nii"))+ "_inv.nii";
    if(fs::exists(gradnonlin_name_inv))
    {
        this->gradwarp_field= readImageD<DisplacementFieldType>(gradnonlin_name_inv);
        this->gradwarp_field_forward= readImageD<DisplacementFieldType>(gradnonlin_field_name);
    }


    for(int d=0;d<2;d++)
    {
        if(data_names[d]=="")
            continue;
        std::string up_name = data_names[d];
        std::string bmtxt_name = up_name.substr(0,up_name.rfind(".nii"))+".bmtxt";
        vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);
        Nvols[d] = Bmatrix.rows();


        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string moteddy_trans_name= this->temp_folder + "/" + basename + "_moteddy_transformations.txt";

        this->dwi_transforms[d].resize(Nvols[d]);
        if(fs::exists(moteddy_trans_name) && moteddy_type!="off")
        {
            std::ifstream moteddy_text_file(moteddy_trans_name);
            for( int vol=0; vol<Nvols[d];vol++)
            {
                std::string line;
                std::getline(moteddy_text_file,line);

                OkanQuadraticTransformType::Pointer quad_trans= OkanQuadraticTransformType::New();
                quad_trans->SetPhase(this->PE_strings[d]);
                quad_trans->SetIdentity();

                OkanQuadraticTransformType::ParametersType params=quad_trans->GetParameters();
                line=line.substr(1);
                for(int p=0;p<OkanQuadraticTransformType::NQUADPARAMS;p++)
                {
                    int npos = line.find(", ");
                    std::string curr_p_string = line.substr(0,npos);

                    double val = atof(curr_p_string.c_str());
                    params[p]=val;
                    line=line.substr(npos+2);
                }
                quad_trans->SetParameters(params);
                OkanQuadraticTransformType::ParametersType flags;
                flags.SetSize(OkanQuadraticTransformType::NQUADPARAMS);
                flags.Fill(0);
                flags[0]=flags[1]=flags[2]=flags[3]=flags[4]=flags[5]=1;
                quad_trans->SetParametersForOptimizationFlags(flags);
                this->dwi_transforms[d][vol]    = quad_trans;
            }
            moteddy_text_file.close();
        }
        else
        {
            OkanQuadraticTransformType::Pointer id_trans= OkanQuadraticTransformType::New();
            id_trans->SetPhase(this->PE_strings[d]);
            id_trans->SetIdentity();
            for(int v=0;v<Nvols[d];v++)
                this->dwi_transforms[d][v]=id_trans;
        }

        std::string s2v_trans_name= this->temp_folder + "/" + basename + "_s2v_transformations.txt";
        if(fs::exists(s2v_trans_name) && s2v_type)
        {
            this->s2v_transformations[d].resize(Nvols[d]);
            ImageType3D::Pointer first_vol= read_3D_volume_from_4D(up_name,0);
            ImageType3D::SizeType sz= first_vol->GetLargestPossibleRegion().GetSize();

            std::ifstream s2v_text_file(s2v_trans_name);
            for( int vol=0; vol<Nvols[d];vol++)
            {
                this->s2v_transformations[d][vol].resize(sz[2]);

                for(int k=0;k<sz[2];k++)
                {
                    std::string line;
                    std::getline(s2v_text_file,line);

                    OkanQuadraticTransformType::Pointer quad_trans= OkanQuadraticTransformType::New();
                    quad_trans->SetPhase(this->PE_strings[d]);
                    quad_trans->SetIdentity();

                    OkanQuadraticTransformType::ParametersType params=quad_trans->GetParameters();
                    line=line.substr(1);
                    for(int p=0;p<OkanQuadraticTransformType::NQUADPARAMS;p++)
                    {
                        int npos = line.find(", ");
                        std::string curr_p_string = line.substr(0,npos);

                        double val = atof(curr_p_string.c_str());
                        params[p]=val;
                        line=line.substr(npos+2);
                    }
                    quad_trans->SetParameters(params);
                    OkanQuadraticTransformType::ParametersType flags;
                    flags.SetSize(OkanQuadraticTransformType::NQUADPARAMS);
                    flags.Fill(0);
                    flags[0]=flags[1]=flags[2]=flags[3]=flags[4]=flags[5]=1;
                    quad_trans->SetParametersForOptimizationFlags(flags);
                    s2v_transformations[d][vol][k]    = quad_trans;

                }
            }
            s2v_text_file.close();
        }

        std::string native_inc_name= this->temp_folder + "/" + basename + "_native_weight.nii";
        if(fs::exists(native_inc_name) && repol_type)
        {
            this->native_weight_img[d].resize(Nvols[d]);
            for(int v=0;v<Nvols[d];v++)
            {
                this->native_weight_img[d][v]= read_3D_volume_from_4D(native_inc_name,v);
            }
        }


        std::string drift_name= this->temp_folder + "/" + basename + "_moteddy_drift.txt";
        if(fs::exists(drift_name) && drift_type!="off")
        {
            std::ifstream infile(drift_name);
            std::string line;
            std::getline(infile, line);
            if(line=="linear")
            {
                this->drift_params[d].resize(2);
                infile>>this->drift_params[d][0]>>  this->drift_params[d][1];
            }
            else
            {
                this->drift_params[d].resize(3);
                infile>>this->drift_params[d][0]>> this->drift_params[d][1] >> this->drift_params[d][2];
            }
            infile.close();

        }
    } //for d
}


template <typename ImageType>
typename ImageType::Pointer FINALDATA::ChangeImageHeaderToDP(typename ImageType::Pointer img)
{
    std::string rot_center= RegistrationSettings::get().getValue<std::string>(std::string("rot_eddy_center"));

    /*********************************************************************************
     We are doing this for several reasons.
     1:  We have to operate on Read/Phase/Slice coordinate system, not x/y/z like ITK.
         So we set the image direction to identity.
         Yes this causes an inconsistency between images and transformations so
         we have to be really careful everytime we register/transform an image.
     2:  Eddy currents do not affect the scanner isocenter (besides a translation which is accounted for in motion correction).
         If the image header is correct and the image coordinate (0,0,0) is indeed the scanner isocenter
         we should use that one.
         But if the header is wrong, we can use the closest thing which is the center voxel of the image.
    **************************************************************************************/

    // do not want to touch the original image so we duplicate it
    using DupType= itk::ImageDuplicator<ImageType>;
    typename DupType::Pointer dup= DupType::New();
    dup->SetInputImage(img);
    dup->Update();
    typename ImageType::Pointer nimg= dup->GetOutput();

    typename ImageType::DirectionType id_dir;     id_dir.SetIdentity();
    nimg->SetDirection(id_dir);
    typename ImageType::PointType new_orig;

    if(rot_center=="isocenter")
    {
        // The center is the isocenter.
        // If we werent changing the Direction matrix, we would not have to do ANYTHING here.
        // But we are, so keep  the same location as the (0,0,0) coordinate with the new Id direction matrix.

        vnl_matrix<double> Sinv(3,3,0);
        Sinv(0,0)= 1./img->GetSpacing()[0];
        Sinv(1,1)= 1./img->GetSpacing()[1];
        Sinv(2,2)= 1./img->GetSpacing()[2];
        vnl_matrix<double> S(3,3,0);
        S(0,0)= img->GetSpacing()[0];
        S(1,1)= img->GetSpacing()[1];
        S(2,2)= img->GetSpacing()[2];

        vnl_vector<double> indo= Sinv*img->GetDirection().GetTranspose() * (-1.*img->GetOrigin().GetVnlVector());   //this is the continuous index (i,j,k) of the isocenter
        vnl_vector<double> new_orig_v= -S*indo;
        new_orig[0]=new_orig_v[0];
        new_orig[1]=new_orig_v[1];
        new_orig[2]=new_orig_v[2];
    }
    else
    {
        if(rot_center=="center_voxel")
        {
            //Make the rotation and eddy center the image center voxel.
            new_orig[0]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[0])-1)/2. * img->GetSpacing()[0];
            new_orig[1]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[1])-1)/2. * img->GetSpacing()[1];
            new_orig[2]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[2])-1)/2. * img->GetSpacing()[2];
        }
        else
        {
            //center_slice
            vnl_matrix<double> Sinv(3,3,0);
            Sinv(0,0)= 1./img->GetSpacing()[0];
            Sinv(1,1)= 1./img->GetSpacing()[1];
            Sinv(2,2)= 1./img->GetSpacing()[2];
            vnl_matrix<double> S(3,3,0);
            S(0,0)= img->GetSpacing()[0];
            S(1,1)= img->GetSpacing()[1];
            S(2,2)= img->GetSpacing()[2];


            vnl_vector<double> center_voxel_index(3,0);
            center_voxel_index[0]= ((int)(img->GetLargestPossibleRegion().GetSize()[0])-1)/2.;
            center_voxel_index[1]= ((int)(img->GetLargestPossibleRegion().GetSize()[1])-1)/2.;
            center_voxel_index[2]= ((int)(img->GetLargestPossibleRegion().GetSize()[2])-1)/2.;

            vnl_vector<double> center_voxel_point = img->GetDirection().GetVnlMatrix()*S*center_voxel_index + img->GetOrigin().GetVnlVector();

            vnl_vector<double> center_point(3,0);
            center_point[2]= center_voxel_point[2];

            vnl_vector<double> indo= Sinv*img->GetDirection().GetTranspose() * (center_voxel_point- img->GetOrigin().GetVnlVector());   //this is the continuous index (i,j,k) of the isocenter
            vnl_vector<double> new_orig_v= -S*indo;
            new_orig[0]=new_orig_v[0];
            new_orig[1]=new_orig_v[1];
            new_orig[2]=new_orig_v[2];
        }

    }
    nimg->SetOrigin(new_orig);

    return nimg;

}


FINALDATA::CompositeTransformType::Pointer FINALDATA::GenerateCompositeTransformForVolume(ImageType3D::Pointer ref_img,int PE, int vol)
{
    CompositeTransformType::Pointer all_trans= CompositeTransformType::New();

    ImageType3D::Pointer ref_img_DP= ChangeImageHeaderToDP<ImageType3D>(ref_img);

    if(this->dwi_transforms[PE].size())
    {

        // The motion eddy transformations are defined in IJK coordinate system not XYZ
        // So we convert them here first to displacement fields that live in both.
        OkanQuadraticTransformType::Pointer curr_mot_eddy_trans= this->dwi_transforms[PE][vol];
        DisplacementFieldType::Pointer mot_eddy_field=DisplacementFieldType::New();
        mot_eddy_field->SetRegions(ref_img->GetLargestPossibleRegion());
        mot_eddy_field->Allocate();
        mot_eddy_field->SetSpacing(ref_img->GetSpacing());
        mot_eddy_field->SetOrigin(ref_img->GetOrigin());
        mot_eddy_field->SetDirection(ref_img->GetDirection());

        itk::ImageRegionIteratorWithIndex<ImageType3D> it(ref_img_DP,ref_img_DP->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType3D::IndexType ind3= it.GetIndex();
            ImageType3D::PointType pt,pt_trans;
            ref_img_DP->TransformIndexToPhysicalPoint(ind3,pt);
            pt_trans=curr_mot_eddy_trans->TransformPoint(pt);

            itk::ContinuousIndex<double,3> cint;
            ref_img_DP->TransformPhysicalPointToContinuousIndex(pt_trans,cint);

            ref_img->TransformIndexToPhysicalPoint(ind3,pt);
            ref_img->TransformContinuousIndexToPhysicalPoint(cint,pt_trans);


            auto vec= pt_trans- pt;
            //auto vec2= ref_img->GetDirection() * vec;
            mot_eddy_field->SetPixel(ind3,vec);
        }
        DisplacementFieldTransformType::Pointer mot_eddy_trans= DisplacementFieldTransformType::New();
        mot_eddy_trans->SetDisplacementField(mot_eddy_field);
        all_trans->AddTransform(mot_eddy_trans);
    }
    if(this->gradwarp_field)
    {
        DisplacementFieldTransformType::Pointer gradwarp_trans= DisplacementFieldTransformType::New();
        gradwarp_trans->SetDisplacementField(this->gradwarp_field);
        all_trans->AddTransform(gradwarp_trans);
    }
    if(PE==1 && this->b0down_t0_b0up_trans)
    {
        all_trans->AddTransform(this->b0down_t0_b0up_trans);
    }
    if(this->epi_trans[PE])
    {
        OkanQuadraticTransformType::Pointer curr_mot_eddy_trans=this->dwi_transforms[PE][vol];
        auto rotmat1= curr_mot_eddy_trans->GetMatrix().GetVnlMatrix();

        vnl_matrix<double> dirmat=ref_img->GetDirection().GetVnlMatrix();
        auto rotmat2= dirmat * rotmat1 *dirmat.transpose();
        auto rotmat2T= rotmat2.transpose();

        DisplacementFieldType::Pointer new_field= DisplacementFieldType::New();
        new_field->SetRegions(this->epi_trans[PE]->GetLargestPossibleRegion());
        new_field->Allocate();
        DisplacementFieldType::PixelType zero; zero.Fill(0);
        new_field->FillBuffer(zero);
        new_field->SetSpacing(this->epi_trans[PE]->GetSpacing());
        new_field->SetOrigin(this->epi_trans[PE]->GetOrigin());
        new_field->SetDirection(this->epi_trans[PE]->GetDirection());

        itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(new_field,new_field->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            DisplacementFieldType::IndexType ind3= it.GetIndex();
            DisplacementFieldType::PixelType vec = this->epi_trans[PE]->GetPixel(ind3);
            vnl_vector<double> vecR= rotmat2T *vec.GetVnlVector();
            vec[0]=vecR[0];
            vec[1]=vecR[1];
            vec[2]=vecR[2];
            it.Set(vec);
        }

        DisplacementFieldTransformType::Pointer mepi_trans= DisplacementFieldTransformType::New();
        mepi_trans->SetDisplacementField(new_field);
        all_trans->AddTransform(mepi_trans);
    }
    if(this->b0_t0_str_trans)
    {
        all_trans->AddTransform(this->b0_t0_str_trans);
    }

    return all_trans;
}



void FINALDATA::GenerateFinalData(std::vector< std::vector<ImageType3D::Pointer> >  final_DWIs)
{
    std::string data_combination_method = RegistrationSettings::get().getValue<std::string>("output_data_combination");

    (*stream)<<std::endl<<"Writing final data..."<<std::endl;

    vnl_matrix<double> up_Bmatrix, down_Bmatrix;
    {
        std::string name = data_names[0];
        fs::path path(name);
        std::string basename= fs::path(path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string bmtxt_name=  this->temp_folder + "/" + basename + "_final_temp.bmtxt";
        up_Bmatrix= read_bmatrix_file(bmtxt_name);
    }
    if(data_names[1]!="")
    {
        std::string name = data_names[1];
        fs::path path(name);
        std::string basename= fs::path(path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string bmtxt_name=  this->temp_folder + "/" + basename + "_final_temp.bmtxt";
        down_Bmatrix= read_bmatrix_file(bmtxt_name);
    }


    if(data_combination_method=="Merge")
    {
        int Nvols=up_Bmatrix.rows();
        std::vector<ImageType3D::Pointer> final_data;
        final_data.resize(Nvols);

        #pragma omp parallel for
        for(int v=0;v<Nvols;v++)
        {
            final_data[v]= ImageType3D::New();
            final_data[v]->SetRegions(final_DWIs[0][v]->GetLargestPossibleRegion());
            final_data[v]->Allocate();
            final_data[v]->SetSpacing(final_DWIs[0][v]->GetSpacing());
            final_data[v]->SetOrigin(final_DWIs[0][v]->GetOrigin());
            final_data[v]->SetDirection(final_DWIs[0][v]->GetDirection());
            final_data[v]->FillBuffer(0);


            itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_data[v],final_data[v]->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ImageType3D::IndexType ind3=it.GetIndex();

                float val0= final_DWIs[0][v]->GetPixel(ind3);
                float val1= final_DWIs[1][v]->GetPixel(ind3);

                float val=0;
                if(val0 > 1E-5 && val1 > 1E-5)
                    val= 2*val0*val1/(val0+val1);
                it.Set(val);
            }
        }

        for(int v=0;v<Nvols;v++)
        {
            write_3D_image_to_4D_file<float>(final_data[v],this->output_name,v,Nvols);
        }

        std::string final_bmtxt_name = this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bmtxt";
        std::string bvecs_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvecs";
        std::string bvals_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvals";

        vnl_matrix<double> final_Bmat = (up_Bmatrix + down_Bmatrix)/2.;
        vnl_matrix<double> bvecs(3,Nvols);
        vnl_matrix<double> bvals= tortoise_bmatrix_to_fsl_bvecs(final_Bmat,bvecs);


        std::ofstream bmtxt_file(final_bmtxt_name);
        bmtxt_file<<final_Bmat;
        bmtxt_file.close();
        std::ofstream bvecs_file(bvecs_fname);
        bvecs_file<< bvecs;
        bvecs_file.close();
        std::ofstream bvals_file(bvals_fname);
        bvals_file<<bvals;
        bvals_file.close();
    }
    else
    {
        if( data_names[1]!="")
        {
            ImageType3D::Pointer blip_up_b0_corrected= readImageD<ImageType3D>(temp_folder+"/blip_up_b0_corrected.nii");
            ImageType3D::Pointer blip_down_b0_corrected= readImageD<ImageType3D>(temp_folder+"/blip_down_b0_corrected.nii");
            ImageType3D::Pointer b0_corrected_final= readImageD<ImageType3D>(temp_folder+"/b0_corrected_final.nii");

            typedef itk::ResampleImageFilter<ImageType3D,ImageType3D>  ResampleImageFilterType;
            ResampleImageFilterType::Pointer resampleFilteru = ResampleImageFilterType::New();
            resampleFilteru->SetOutputParametersFromImage(template_structural);
            resampleFilteru->SetInput(blip_up_b0_corrected);
            resampleFilteru->SetTransform( b0_t0_str_trans);
            resampleFilteru->Update();
            blip_up_b0_corrected= resampleFilteru->GetOutput();


            ResampleImageFilterType::Pointer resampleFilterd = ResampleImageFilterType::New();
            resampleFilterd->SetOutputParametersFromImage(template_structural);
            resampleFilterd->SetInput(blip_down_b0_corrected);
            resampleFilterd->SetTransform( b0_t0_str_trans);
            resampleFilterd->Update();
            blip_down_b0_corrected= resampleFilterd->GetOutput();

            ResampleImageFilterType::Pointer resampleFilterc = ResampleImageFilterType::New();
            resampleFilterc->SetOutputParametersFromImage(template_structural);
            resampleFilterc->SetInput(b0_corrected_final);
            resampleFilterc->SetTransform( b0_t0_str_trans);
            resampleFilterc->Update();
            b0_corrected_final= resampleFilterc->GetOutput();


            ImageType3D::Pointer b0_imgs[2]={blip_up_b0_corrected,blip_down_b0_corrected};
            for(int PE=0;PE<2;PE++)
            {
                int Nvols= final_DWIs[PE].size();
                for(int v=0;v<Nvols;v++)
                {
                    itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_DWIs[PE][v],final_DWIs[PE][v]->GetLargestPossibleRegion());
                    for(it.GoToBegin();!it.IsAtEnd();++it)
                    {
                        ImageType3D::IndexType ind3= it.GetIndex();

                        double det= b0_corrected_final->GetPixel(ind3)/b0_imgs[PE]->GetPixel(ind3);
                        float val1=it.Get();
                        double fval = val1*det;
                        if(std::isnan(fval))
                            fval=0;

                        it.Set(fval);
                    }
                }
            }

            if(data_combination_method=="JacConcat")
            {
                int total_Nvols= final_DWIs[0].size() + final_DWIs[1].size();

                for(int v=0;v<final_DWIs[0].size();v++)
                {
                    write_3D_image_to_4D_file<float>(final_DWIs[0][v],this->output_name,v,total_Nvols);
                }
                for(int v=0;v<final_DWIs[1].size();v++)
                {
                    write_3D_image_to_4D_file<float>(final_DWIs[1][v],this->output_name,final_DWIs[0].size() + v,total_Nvols);
                }

                std::string final_bmtxt_name = this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bmtxt";
                std::string bvecs_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvecs";
                std::string bvals_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvals";

                vnl_matrix<double> final_Bmat(total_Nvols,6);
                final_Bmat.update(up_Bmatrix,0,0);
                final_Bmat.update(down_Bmatrix,final_DWIs[0].size(),0);
                vnl_matrix<double> bvecs(3,total_Nvols);
                vnl_matrix<double> bvals= tortoise_bmatrix_to_fsl_bvecs(final_Bmat,bvecs);
                std::ofstream bmtxt_file(final_bmtxt_name);
                bmtxt_file<<final_Bmat;
                bmtxt_file.close();
                std::ofstream bvecs_file(bvecs_fname);
                bvecs_file<< bvecs;
                bvecs_file.close();
                std::ofstream bvals_file(bvals_fname);
                bvals_file<<bvals;
                bvals_file.close();
            }
            else
            {
                std::string final_folder= fs::path(this->output_name).parent_path().string();
                if(final_folder=="")
                    final_folder="./";
                {
                    //std::string name = data_names[0];
                    //fs::path path(name);
                    //std::string basename= fs::path(path).filename().string();
                    //basename=basename.substr(0,basename.rfind(".nii"));
                    //std::string final_name= final_folder + "/"+basename + "_TORTOISE_final.nii";
                   // std::string final_bmtxt_name= final_folder + "/"+basename + "_TORTOISE_final.bmtxt";
                    //std::string bvecs_fname= final_folder + "/"+basename + "_TORTOISE_final.bvecs";
                    //std::string bvals_fname= final_folder + "/"+basename + "_TORTOISE_final.bvals";

                    std::string final_bmtxt_name = this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bmtxt";
                    std::string bvecs_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvecs";
                    std::string bvals_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvals";


                    for(int v=0;v<final_DWIs[0].size();v++)
                    {
                        write_3D_image_to_4D_file<float>(final_DWIs[0][v],this->output_name,v,final_DWIs[0].size());
                    }

                    vnl_matrix<double> bvecs(3,final_DWIs[0].size());
                    vnl_matrix<double> bvals= tortoise_bmatrix_to_fsl_bvecs(up_Bmatrix,bvecs);
                    std::ofstream bmtxt_file(final_bmtxt_name);
                    bmtxt_file<<up_Bmatrix;
                    bmtxt_file.close();
                    std::ofstream bvecs_file(bvecs_fname);
                    bvecs_file<< bvecs;
                    bvecs_file.close();
                    std::ofstream bvals_file(bvals_fname);
                    bvals_file<<bvals;
                    bvals_file.close();
                }
                {
                    std::string name = data_names[1];
                    fs::path path(name);
                    std::string basename= fs::path(path).filename().string();
                    basename=basename.substr(0,basename.rfind(".nii"));
                    std::string final_name= final_folder + "/"+basename + "_TORTOISE_final.nii";
                    std::string final_bmtxt_name= final_folder + "/"+basename + "_TORTOISE_final.bmtxt";
                    std::string bvecs_fname= final_folder + "/"+basename + "_TORTOISE_final.bvecs";
                    std::string bvals_fname= final_folder + "/"+basename + "_TORTOISE_final.bvals";

                    for(int v=0;v<final_DWIs[1].size();v++)
                    {
                        write_3D_image_to_4D_file<float>(final_DWIs[1][v],final_name,v,final_DWIs[1].size());
                    }

                    vnl_matrix<double> bvecs(3,final_DWIs[1].size());
                    vnl_matrix<double> bvals= tortoise_bmatrix_to_fsl_bvecs(down_Bmatrix,bvecs);
                    std::ofstream bmtxt_file(final_bmtxt_name);
                    bmtxt_file<<down_Bmatrix;
                    bmtxt_file.close();
                    std::ofstream bvecs_file(bvecs_fname);
                    bvecs_file<< bvecs;
                    bvecs_file.close();
                    std::ofstream bvals_file(bvals_fname);
                    bvals_file<<bvals;
                    bvals_file.close();
                }
            }
        }
        else         // just up data. Do Jacobian
        {

            ImageType3D::Pointer first_vol= read_3D_volume_from_4D(data_names[0],0);
            ImageType3D::Pointer first_vol_DP= ChangeImageHeaderToDP<ImageType3D>(first_vol);
            ImageType3D::SizeType sz= first_vol->GetLargestPossibleRegion().GetSize();

            int phase=0;
            if(this->PE_strings[0]=="vertical")
                phase=1;
            if(this->PE_strings[0]=="slice")
                phase=2;

            vnl_vector<double> phase_vector(3,0);
            phase_vector[phase]=1;
            vnl_matrix_fixed<double,3,3> dir = first_vol->GetDirection().GetVnlMatrix();
            vnl_vector_fixed<double,3> new_phase_vector = dir*phase_vector;
            int phase_xyz;
            if( (fabs(new_phase_vector[0])>fabs(new_phase_vector[1])) && (fabs(new_phase_vector[0])>fabs(new_phase_vector[2])))
                phase_xyz=0;
            else if( (fabs(new_phase_vector[1])>fabs(new_phase_vector[0])) && (fabs(new_phase_vector[1])>fabs(new_phase_vector[2])))
                phase_xyz=1;
            else phase_xyz=2;


            #pragma omp parallel for
            for(int v=0;v<final_DWIs[0].size();v++)
            {
                CompositeTransformType::Pointer all_trans= CompositeTransformType::New();

                using IdTransformType = itk::IdentityTransform<double,3> ;
                IdTransformType::Pointer id_trans = IdTransformType::New();
                id_trans->SetIdentity();
                all_trans->AddTransform(id_trans);


                DisplacementFieldType::Pointer mot_eddy_field=nullptr;
                if(this->dwi_transforms[0].size())
                {
                    OkanQuadraticTransformType::Pointer curr_mot_eddy_trans= this->dwi_transforms[0][v];
                    mot_eddy_field=DisplacementFieldType::New();
                    mot_eddy_field->SetRegions(first_vol->GetLargestPossibleRegion());
                    mot_eddy_field->Allocate();
                    mot_eddy_field->SetSpacing(first_vol->GetSpacing());
                    mot_eddy_field->SetOrigin(first_vol->GetOrigin());
                    mot_eddy_field->SetDirection(first_vol->GetDirection());

                    itk::ImageRegionIteratorWithIndex<ImageType3D> it(first_vol_DP,first_vol_DP->GetLargestPossibleRegion());
                    for(it.GoToBegin();!it.IsAtEnd();++it)
                    {
                        ImageType3D::IndexType ind3= it.GetIndex();
                        ImageType3D::PointType pt,pt_trans;
                        first_vol_DP->TransformIndexToPhysicalPoint(ind3,pt);
                        pt_trans=curr_mot_eddy_trans->TransformPoint(pt);

                        auto vec= pt_trans- pt;
                        auto vec2= first_vol->GetDirection() * vec;
                        mot_eddy_field->SetPixel(ind3,vec2);
                    }
                    DisplacementFieldTransformType::Pointer mot_eddy_trans= DisplacementFieldTransformType::New();
                    mot_eddy_trans->SetDisplacementField(mot_eddy_field);
                    all_trans->AddTransform(mot_eddy_trans);
                }
                if(this->gradwarp_field)
                {
                    DisplacementFieldTransformType::Pointer gradwarp_trans= DisplacementFieldTransformType::New();
                    gradwarp_trans->SetDisplacementField(this->gradwarp_field);
                    all_trans->AddTransform(gradwarp_trans);
                }
                if(this->epi_trans[0])
                {
                    DisplacementFieldTransformType::Pointer mepi_trans= DisplacementFieldTransformType::New();
                    mepi_trans->SetDisplacementField(this->epi_trans[0]);
                    all_trans->AddTransform(mepi_trans);
                }

                ImageType3D::Pointer det_img = ImageType3D::New();
                det_img->SetRegions(first_vol->GetLargestPossibleRegion());
                det_img->Allocate();
                det_img->SetSpacing(first_vol->GetSpacing());
                det_img->SetOrigin(first_vol->GetOrigin());
                det_img->SetDirection(first_vol->GetDirection());
                det_img->FillBuffer(1);

                itk::ImageRegionIteratorWithIndex<ImageType3D> it(det_img,det_img->GetLargestPossibleRegion());
                for(it.GoToBegin();!it.IsAtEnd();++it)
                {
                    ImageType3D::IndexType ind3= it.GetIndex();
                    if(ind3[phase]>0 && ind3[phase]<sz[phase]-1)
                    {
                        DisplacementFieldType::IndexType Nind=ind3;
                        Nind[phase]+=1;
                        ImageType3D::PointType pt,pt_trans;
                        det_img->TransformIndexToPhysicalPoint(Nind,pt);
                        pt_trans= all_trans->TransformPoint(pt);
                        double grad=  pt_trans[phase]-pt[phase];

                        Nind[phase]-=2;
                        det_img->TransformIndexToPhysicalPoint(Nind,pt);
                        pt_trans= all_trans->TransformPoint(pt);
                        grad-=  pt_trans[phase]-pt[phase];
                        grad*= 0.5/det_img->GetSpacing()[phase];

                        vnl_vector<double> temp(3,0);
                        temp[phase]=grad;
                        temp = det_img->GetDirection().GetVnlMatrix()* temp;

                        it.Set( 1 + temp[phase_xyz]);
                    }
                }

                typedef itk::ResampleImageFilter<ImageType3D, ImageType3D> ResamplerType;
                ResamplerType::Pointer resampler= ResamplerType::New();
                resampler->SetInput(det_img);
                resampler->SetOutputParametersFromImage(this->template_structural);
                resampler->SetDefaultPixelValue(1);
                if(this->b0_t0_str_trans)
                    resampler->SetTransform(this->b0_t0_str_trans);
                else
                    resampler->SetTransform(id_trans);
                resampler->Update();
                ImageType3D::Pointer det_img_on_str= resampler->GetOutput();


                itk::ImageRegionIteratorWithIndex<ImageType3D> it2(final_DWIs[0][v],final_DWIs[0][v]->GetLargestPossibleRegion());
                for(it2.GoToBegin();!it2.IsAtEnd();++it2)
                {
                    ImageType3D::IndexType ind3= it.GetIndex();
                    float val = it.Get()* det_img_on_str->GetPixel(ind3);
                    if(val<0)
                        val=0;
                    it.Set(val);
                }
            }  //for v


            for(int v=0;v<final_DWIs[0].size();v++)
            {
                write_3D_image_to_4D_file<float>(final_DWIs[0][v],this->output_name,v,final_DWIs[0].size());
            }

            std::string final_bmtxt_name = this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bmtxt";
            std::string bvecs_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvecs";
            std::string bvals_fname=this->output_name.substr(0,this->output_name.rfind(".nii")) + ".bvals";

            vnl_matrix<double> bvecs(3,final_DWIs[0].size());
            vnl_matrix<double> bvals= tortoise_bmatrix_to_fsl_bvecs(up_Bmatrix,bvecs);
            std::ofstream bmtxt_file(final_bmtxt_name);
            bmtxt_file<<up_Bmatrix;
            bmtxt_file.close();
            std::ofstream bvecs_file(bvecs_fname);
            bvecs_file<< bvecs;
            bvecs_file.close();
            std::ofstream bvals_file(bvals_fname);
            bvals_file<<bvals;
            bvals_file.close();



        } //else dti_names[1]
    } //else merge
}


std::vector< std::vector<ImageType3D::Pointer> >  FINALDATA::GenerateTransformedInterpolatedData()
{
    std::vector< std::vector<ImageType3D::Pointer> > final_imgs_to_return;
    final_imgs_to_return.resize(2);

    std::string output_gradnonlin_type = RegistrationSettings::get().getValue<std::string>("output_gradnonlin_Bmtxt_type");
    float THR=RegistrationSettings::get().getValue<float>("outlier_prob");

    for(int PE=0;PE<2;PE++)
    {
        int nvols= Nvols[PE];

        std::string up_name = data_names[PE];
        if(up_name=="")
            continue;

        std::string bmtxt_name = up_name.substr(0,up_name.rfind(".nii"))+".bmtxt";
        vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);

        std::vector<ImageType3D::Pointer> raw_data,final_data;
        std::vector<ImageType3D::Pointer> final_inclusion_imgs,final_weight_imgs;
        raw_data.resize(nvols);
        final_data.resize(nvols);
        if(this->native_weight_img[PE].size())
        {
            final_inclusion_imgs.resize(nvols);
            final_weight_imgs.resize(nvols);
        }

        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string native_synth_name= this->temp_folder + "/" + basename + "_native_native_synth.nii";

        ImageType3D::Pointer orig_mask2=nullptr;
        for(int vol=0;vol<nvols;vol++)
        {
            raw_data[vol]= read_3D_volume_from_4D(data_names[PE],vol);

            if(vol==0)
            {
                std::string b0_mask_img_fname = RegistrationSettings::get().getValue<std::string>("b0_mask_img");

                if(b0_mask_img_fname=="")
                {
                    orig_mask2= create_mask(raw_data[0]);
                }
                else
                {
                    orig_mask2= readImageD<ImageType3D>(b0_mask_img_fname);
                }

            }
            if(this->native_weight_img[PE].size())
            {
                ImageType3D::Pointer synth_img = read_3D_volume_from_4D(native_synth_name,vol);
                itk::ImageRegionIteratorWithIndex<ImageType3D> it(synth_img,synth_img->GetLargestPossibleRegion());

                std::vector<float> synth_vals;
                for(it.GoToBegin();!it.IsAtEnd();++it)
                {
                    if(it.Get() > 1 && orig_mask2->GetPixel(it.GetIndex())>0)
                        synth_vals.push_back(it.Get());
                }
                float med_val= median(synth_vals);

                for(it.GoToBegin();!it.IsAtEnd();++it)
                {
                    ImageType3D::IndexType ind3= it.GetIndex();
                    if(synth_img->GetPixel(ind3)>med_val/20.)
                    {
                        float weight= this->native_weight_img[PE][vol]->GetPixel(ind3);
                        if(weight<THR)
                        {
                            raw_data[vol]->SetPixel(ind3,it.Get());
                        }
                    }
                }
            }

        }


        std::string noise_name= this->temp_folder + "/" + basename + "_noise.nii";
        ImageType3D::Pointer noise_img= readImageD<ImageType3D>(noise_name);
        ImageType3D::Pointer final_noise_img=nullptr;

        (*stream)<<std::endl<<"Transforming Volume done: "<<std::flush;


        #pragma omp parallel for
        for(int vol=0;vol<nvols;vol++)
        {
            TORTOISE::EnableOMPThread();

            ImageType3D::Pointer final_img=ImageType3D::New();
            final_img->SetRegions(template_structural->GetLargestPossibleRegion());
            final_img->Allocate();
            final_img->SetSpacing(template_structural->GetSpacing());
            final_img->SetDirection(template_structural->GetDirection());
            final_img->SetOrigin(template_structural->GetOrigin());
            final_img->FillBuffer(0.);

            //Transformation + interpolation
            {
                using MeasurementVectorType = itk::Vector<float, 3>;
                using SampleType = itk::Statistics::ListSample<MeasurementVectorType>;
                using TreeGeneratorType = itk::Statistics::KdTreeGenerator<SampleType>;
                using TreeType = TreeGeneratorType::KdTreeType;

                std::vector<float> values;
                std::vector<ImageType3D::IndexType> orig_slice_inds;
                SampleType::Pointer sample = SampleType::New();
                sample->SetMeasurementVectorSize(3);
                TreeType::Pointer tree =nullptr;



                //If we can, we want to do backward interpolation with s2v.
                // So what what we do is to convert it to a displacement field slice by slice and then invert it.
                //This field might not be diffeomorphic so we might not be able to invert it accurately

                CompositeTransformType::Pointer all_trans_wo_s2v=GenerateCompositeTransformForVolume(raw_data[0],PE, vol);
                CompositeTransformType::Pointer all_trans=nullptr;
                DisplacementFieldType::Pointer forward_s2v_field=nullptr;

                if(vol==0)
                {
                    using NNInterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType3D, double>;
                    NNInterpolatorType::Pointer NNinterpolator = NNInterpolatorType::New();

                    using ResamplerType = itk::ResampleImageFilter<ImageType3D,ImageType3D>;
                    ResamplerType::Pointer resampler= ResamplerType::New();
                    resampler->SetDefaultPixelValue(0);
                    resampler->SetInput(noise_img);
                    resampler->SetInterpolator(NNinterpolator );
                    resampler->SetOutputParametersFromImage(template_structural);
                    resampler->SetTransform(all_trans_wo_s2v);
                    resampler->Update();
                    final_noise_img= resampler->GetOutput();
                }

                if(this->s2v_transformations[PE].size()!=0)
                {
                    forward_s2v_field = DisplacementFieldType::New();
                    forward_s2v_field->SetRegions(raw_data[0]->GetLargestPossibleRegion());
                    forward_s2v_field->Allocate();
                    forward_s2v_field->SetSpacing(raw_data[0]->GetSpacing());
                    forward_s2v_field->SetSpacing(raw_data[0]->GetSpacing());
                    forward_s2v_field->SetOrigin(raw_data[0]->GetOrigin());
                    forward_s2v_field->SetDirection(raw_data[0]->GetDirection());
                    DisplacementFieldType::PixelType zero; zero.Fill(0);
                    forward_s2v_field->FillBuffer(zero);


                    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(forward_s2v_field,forward_s2v_field->GetLargestPossibleRegion());
                    for(it.GoToBegin();!it.IsAtEnd();++it)
                    {
                        ImageType3D::IndexType ind3=it.GetIndex();

                        DisplacementFieldType::PointType pt,pt_trans;
                        forward_s2v_field->TransformIndexToPhysicalPoint(ind3,pt);
                        pt_trans = this->s2v_transformations[PE][vol][ind3[2]]->TransformPoint(pt);

                        DisplacementFieldType::PixelType vec;
                        vec[0]= pt_trans[0] - pt[0];
                        vec[1]= pt_trans[1] - pt[1];
                        vec[2]= pt_trans[2] - pt[2];
                        it.Set(vec);

                        if(  (this->native_weight_img[PE].size() && this->native_weight_img[PE][vol]->GetPixel(ind3)>THR) || this->native_weight_img[PE].size()==0 )
                        {
                            itk::ContinuousIndex<double,3> ind3_t;
                            raw_data[vol]->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_t);
                            MeasurementVectorType tt;
                            tt[0]=ind3_t[0];
                            tt[1]=ind3_t[1];
                            tt[2]=ind3_t[2];

                            sample->PushBack(tt);
                            values.push_back(raw_data[vol]->GetPixel(ind3));
                            orig_slice_inds.push_back(ind3);
                        }
                    }
                    TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
                    treeGenerator->SetSample(sample);
                    treeGenerator->SetBucketSize(16);
                    treeGenerator->Update();
                    tree = treeGenerator->GetOutput();

                    typedef itk::InvertDisplacementFieldImageFilterOkan<DisplacementFieldType> InverterType;
                    InverterType::Pointer inverter = InverterType::New();
                    inverter->SetInput( forward_s2v_field );
                    inverter->SetMaximumNumberOfIterations( 50 );
                    inverter->SetMeanErrorToleranceThreshold( 0.0004 );
                    inverter->SetMaxErrorToleranceThreshold( 0.04 );
                    //inverter->SetNumberOfWorkUnits(NITK);
                    inverter->SetNumberOfWorkUnits(1);
                    inverter->Update();
                    DisplacementFieldType::Pointer backward_s2v_field =inverter->GetOutput();

                    DisplacementFieldTransformType::Pointer backward_s2v_field_trans= DisplacementFieldTransformType::New();
                    backward_s2v_field_trans->SetDisplacementField(backward_s2v_field);

                    all_trans=CompositeTransformType::New();
                    all_trans->AddTransform(backward_s2v_field_trans);
                    all_trans->AddTransform(all_trans_wo_s2v);
                    all_trans->FlattenTransformQueue();

                } //if s2v

                ImageType3D::SizeType orig_sz= raw_data[vol]->GetLargestPossibleRegion().GetSize();
                ImageType3D::SpacingType orig_spc = raw_data[vol]->GetSpacing();
                ImageType3D::SizeType final_sz= final_img->GetLargestPossibleRegion().GetSize();

                using BSInterpolatorType = itk::BSplineInterpolateImageFunction<ImageType3D, double>;
                BSInterpolatorType::Pointer BSinterpolator = BSInterpolatorType::New();
                BSinterpolator->SetSplineOrder(3);
                BSinterpolator->SetInputImage(raw_data[vol]);

                using LinearInterpolatorType = itk::LinearInterpolateImageFunction<ImageType3D, double>;
                LinearInterpolatorType::Pointer Lininterpolator = LinearInterpolatorType::New();
                Lininterpolator->SetInputImage(raw_data[vol]);

                using NNInterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType3D, double>;
                NNInterpolatorType::Pointer NNinterpolator = NNInterpolatorType::New();
                if(this->native_weight_img[PE].size())
                {
                    NNinterpolator->SetInputImage(this->native_weight_img[PE][vol]);

                    final_inclusion_imgs[vol]=ImageType3D::New();
                    final_inclusion_imgs[vol]->SetRegions(template_structural->GetLargestPossibleRegion());
                    final_inclusion_imgs[vol]->Allocate();
                    final_inclusion_imgs[vol]->SetSpacing(template_structural->GetSpacing());
                    final_inclusion_imgs[vol]->SetDirection(template_structural->GetDirection());
                    final_inclusion_imgs[vol]->SetOrigin(template_structural->GetOrigin());
                    final_inclusion_imgs[vol]->FillBuffer(1);


                    final_weight_imgs[vol]=ImageType3D::New();
                    final_weight_imgs[vol]->SetRegions(template_structural->GetLargestPossibleRegion());
                    final_weight_imgs[vol]->Allocate();
                    final_weight_imgs[vol]->SetSpacing(template_structural->GetSpacing());
                    final_weight_imgs[vol]->SetDirection(template_structural->GetDirection());
                    final_weight_imgs[vol]->SetOrigin(template_structural->GetOrigin());
                    final_weight_imgs[vol]->FillBuffer(1);
                }



                const double DIST_POW=  RegistrationSettings::get().getValue<int>("interp_POW");
                unsigned int                           numberOfNeighbors = 16;


                itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_img,final_img->GetLargestPossibleRegion());
                for(it.GoToBegin();!it.IsAtEnd();++it)
                {
                    ImageType3D::IndexType ind3=it.GetIndex();
                    ImageType3D::PointType pt;
                    final_img->TransformIndexToPhysicalPoint(ind3,pt);
                    ImageType3D::PointType pt_trans_nos2v= all_trans_wo_s2v->TransformPoint(pt);

                    if(this->native_weight_img[PE].size())
                    {
                        if(NNinterpolator->IsInsideBuffer(pt_trans_nos2v))
                        {
                            ImageType3D::PixelType val = NNinterpolator->Evaluate(pt_trans_nos2v);
                            final_weight_imgs[vol]->SetPixel(ind3,val);
                            if(val<=THR)
                                final_inclusion_imgs[vol]->SetPixel(ind3,0);
                        }
                    }


                    if(this->s2v_transformations[PE].size()==0)
                    {
                        if(BSinterpolator->IsInsideBuffer(pt_trans_nos2v))
                        {
                            ImageType3D::PixelType val = BSinterpolator->Evaluate(pt_trans_nos2v);
                            if(val<0)
                                val=0;
                            final_img->SetPixel(ind3,val);
                        }
                    }
                    else
                    {
                        ImageType3D::PointType pt_trans=all_trans->TransformPoint(pt);

                        itk::ContinuousIndex<double,3> ind3_t;
                        raw_data[vol]->TransformPhysicalPointToContinuousIndex(pt_trans_nos2v,ind3_t);
                        if(ind3_t[0]<-1 || ind3_t[0]>orig_sz[0] || ind3_t[1]<-1 || ind3_t[1]>orig_sz[1] || ind3_t[2]<-1 || ind3_t[2]>orig_sz[2] )
                        {
                            it.Set(0);
                            continue;
                        }

                        itk::ContinuousIndex<double,3> ind3_ts2v;
                        raw_data[vol]->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_ts2v);

                        bool forward_interp=false;


                        //We check voxel by voxel if the inverted, i.e. backward s2v deformation field  and the forward sv2 transformation
                        // are consistent, i.e. give close displacements.
                        // if not, most likely not diffeomorphic so not invertable
                        // then we revert back to forward interpolation with idw.

                        int SL= (int)std::round(ind3_ts2v[2]);
                        if(SL<0 || SL > orig_sz[2]-1)
                            forward_interp=true;
                        else
                        {
                            ImageType3D::PointType pt_trans_s2v_inv= this->s2v_transformations[PE][vol][SL]->TransformPoint(pt_trans);

                            DisplacementFieldType::PixelType diff;
                            diff[0]= pt_trans_s2v_inv[0] - pt_trans_nos2v[0];
                            diff[1]= pt_trans_s2v_inv[1] - pt_trans_nos2v[1];
                            diff[2]= pt_trans_s2v_inv[2] - pt_trans_nos2v[2];
                            double diff_mag = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);

                            if(diff_mag > 0.1*orig_spc[2])
                                forward_interp=true;
                        }


                        if(!forward_interp || values.size()==0)
                        {
                            if(BSinterpolator->IsInsideBuffer(pt_trans))
                            {
                                ImageType3D::PixelType val = BSinterpolator->Evaluate(pt_trans);
                                if(val<0)
                                    val=0;
                                final_img->SetPixel(ind3,val);
                            }
                        }
                        else
                        {
                            MeasurementVectorType queryPoint;
                            queryPoint[0]=ind3_t[0];
                            queryPoint[1]=ind3_t[1];
                            queryPoint[2]=ind3_t[2];

                            TreeType::InstanceIdentifierVectorType neighbors;
                            std::vector<double> dists;
                            tree->Search(queryPoint, numberOfNeighbors, neighbors,dists);

                            std::vector<double>::iterator mini = std::min_element(dists.begin(), dists.end());
                            double mn= *mini;
                            int mn_id=std::distance(dists.begin(), mini);

                            if(mn<0.1)
                            {
                                float val= values[neighbors[mn_id]];
                                it.Set(val);
                            }
                            else
                            {
                                // If neigbor exists within one voxel distance
                                // do inverse powered distance weighted interpolation
                                // power is 6 to make images sharper.

                                double sm_weight=0;
                                double sm_val=0;
                                int Nl1=0;
                                for(int n=0;n<numberOfNeighbors;n++)
                                {
                                    int neighbor= neighbors[n];
                                    float dist = dists[n];
                                    if(dist <= 1)
                                        Nl1++;

                                    double dist2= 1./pow(dist,DIST_POW);
                                    if(dist2>1E-50)
                                    {
                                        sm_val+= values[neighbor] *dist2;
                                        sm_weight+= dist2;
                                    }
                                }
                                float idw_val = 0;
                                if(sm_weight!=0)
                                    idw_val=sm_val/sm_weight;


                                //if(mn<1.)
                                if(Nl1>3)
                                {
                                    it.Set(idw_val);
                                }
                                else
                                {

                                    int slice_id_min = orig_slice_inds[neighbors[mn_id]][2];

                                    auto min_params= this->s2v_transformations[PE][vol][slice_id_min]->GetParameters();
                                    if(fabs(min_params[0])+fabs(min_params[1])+fabs(min_params[2])+fabs(min_params[3])+fabs(min_params[4])+fabs(min_params[5])<1E-10)
                                    {
                                        if(BSinterpolator->IsInsideBuffer(pt_trans_nos2v))
                                        {
                                            ImageType3D::PixelType val = BSinterpolator->Evaluate(pt_trans_nos2v);
                                            if(val<0)
                                                val=0;
                                            final_img->SetPixel(ind3,val);
                                        }
                                    }
                                    else
                                    {
                                        it.Set(idw_val);



                                    } // not BSP
                                } //if not idw
                            } //if mn>0.1
                        } //if not backward
                    }  //if s2v
                } //for ind3
            } //dummy interpolation


            //final data transformed.. Now first apply drift.
            if(this->drift_params[PE].size())
            {
                double scale;
                if(this->drift_params[PE].size()==2)
                    scale= this->drift_params[PE][0]/(this->drift_params[PE][1]*vol+this->drift_params[PE][0]);
                else
                    scale= this->drift_params[PE][0] / (this->drift_params[PE][1]*vol*vol+ this->drift_params[PE][2]*vol + this->drift_params[PE][0] );


                itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_img,final_img->GetLargestPossibleRegion());
                for(it.GoToBegin();!it.IsAtEnd();++it)
                {
                    it.Set(it.Get()*scale);
                }
            }
            final_data[vol]=final_img;

            fs::path up_path(up_name);
            std::string basename= fs::path(up_path).filename().string();
            basename=basename.substr(0,basename.rfind(".nii"));
            std::string new_nii_name=  this->temp_folder + "/" + basename + "_final_temp_norepol.nii";
            std::string new_inc_name= this->temp_folder + "/" + basename + "_final_temp_inc.nii";


            #pragma omp critical
            {
                (*stream)<<vol<<", "<<std::flush;
                if(final_inclusion_imgs.size())
                    write_3D_image_to_4D_file<float>(final_img,new_nii_name,vol,Nvols[PE]);

            }
            TORTOISE::DisableOMPThread();
        } //for vol



        vnl_matrix<double> rot_Bmat;

        {
            //Compute rotated Bmatrix, overall version
            vnl_matrix_fixed<double,3,3> id_trans; id_trans.set_identity();
            rot_Bmat= RotateBMatrix(Bmatrix,this->dwi_transforms[PE],id_trans,id_trans);
            if(this->b0_t0_str_trans)
                rot_Bmat= RotateBMatrix(rot_Bmat,this->b0_t0_str_trans->GetMatrix().GetVnlMatrix(), template_structural->GetDirection().GetVnlMatrix(),  raw_data[0]->GetDirection().GetVnlMatrix());


            fs::path up_path(up_name);
            std::string basename= fs::path(up_path).filename().string();
            basename=basename.substr(0,basename.rfind(".nii"));
            std::string new_bmtxt_name=  this->temp_folder + "/" + basename + "_final_temp.bmtxt";
            std::ofstream outfile(new_bmtxt_name);
            outfile<<rot_Bmat;
            outfile.close();

        }


        // and finally  apply outlier replacement
        if(final_inclusion_imgs.size())
        {
            (*stream)<<std::endl<<"Replacing final outliers..."<<std::endl;


            //Get average b=0 image and mask it
            vnl_vector<double> bvals = Bmatrix.get_column(0) + Bmatrix.get_column(3)+ Bmatrix.get_column(5);

            ImageType3D::Pointer orig_mask=nullptr;
            std::string b0_mask_img_fname = RegistrationSettings::get().getValue<std::string>("b0_mask_img");
            if(b0_mask_img_fname=="")
            {
                orig_mask= create_mask(raw_data[0]);
            }
            else
            {
                orig_mask=readImageD<ImageType3D>(b0_mask_img_fname);
            }

            orig_mask->SetDirection(this->native_weight_img[PE][0]->GetDirection());

            #pragma omp parallel for
            for(int vol=0;vol<Nvols[PE];vol++)
            {
                TORTOISE::EnableOMPThread();

                //Generate a final mask that includes both the brain mask and outlier mask
                using FilterType = itk::MultiplyImageFilter<ImageType3D, ImageType3D, ImageType3D>;
                FilterType::Pointer filter = FilterType::New();
                filter->SetInput2(orig_mask);
                filter->SetInput1(this->native_weight_img[PE][vol]);
                filter->Update();
                ImageType3D::Pointer weight_img= filter->GetOutput();

                //Check if we have a valid voxel
                bool allzeros=true;
                itk::ImageRegionIterator<ImageType3D> it(weight_img,weight_img->GetLargestPossibleRegion());
                it.GoToBegin();
                while(!it.IsAtEnd())
                {
                    if(it.Get()>THR)
                    {
                        allzeros=false;
                        break;
                    }
                    ++it;
                }
                if(allzeros)
                    final_inclusion_imgs[vol]->FillBuffer(0);

                fs::path up_path(up_name);
                std::string basename= fs::path(up_path).filename().string();
                basename=basename.substr(0,basename.rfind(".nii"));
                std::string new_inc_name= this->temp_folder + "/" + basename + "_final_temp_inc.nii";
                #pragma omp critical
                {
                    write_3D_image_to_4D_file<float>(final_inclusion_imgs[vol], new_inc_name,vol,Nvols[PE]);
                }
            }


            // FIRST, resynthesize images in the final space using the final_data
            float dti_bval_cutoff= RegistrationSettings::get().getValue<float>(std::string("dti_bval"));
            float mapmri_bval_cutoff= RegistrationSettings::get().getValue<float>(std::string("hardi_bval"));


            std::vector<int> low_DT_indices, MAPMRI_indices;
            for(int v=0;v<Nvols[PE];v++)
            {
                if( bvals[v] <=mapmri_bval_cutoff)
                {
                    low_DT_indices.push_back(v);
                }
                else
                    MAPMRI_indices.push_back(v);
            }

            //DTI fitting

            std::vector<ImageType3D::Pointer> synth_imgs;
            synth_imgs.resize(Nvols[PE]);
            raw_data.clear();raw_data.resize(0);

            ImageType3D::Pointer final_mask;
            {
                std::vector<std::vector<ImageType3D::Pointer> > dummyv;
                std::vector<int> dummy;
                DTIModel dti_estimator;
                dti_estimator.SetBmatrix(rot_Bmat);
                dti_estimator.SetDWIData(final_data);
                dti_estimator.SetWeightImage(final_weight_imgs);
                dti_estimator.SetVoxelwiseBmatrix(dummyv);
                dti_estimator.SetMaskImage(nullptr);
                dti_estimator.SetVolIndicesForFitting(low_DT_indices);
                dti_estimator.SetFittingMode("WLLS");
                dti_estimator.PerformFitting();
                final_mask= create_mask(dti_estimator.GetA0Image());


                // MAPMRI FITTING
                const unsigned int FINAL_STAGE_MAPMRI_DEGREE=6;
                MAPMRIModel mapmri_estimator2, mapmri_estimator4,mapmri_estimator6;
                if(MAPMRI_indices.size()>0)
                {
                    double max_bval= bvals.max_value();
                    float small_delta,big_delta;

                    if(this->jsons[PE]["SmallDelta"]==json::value_t::null || this->jsons[PE]["BigDelta"]==json::value_t::null)
                    {
                        float bd= RegistrationSettings::get().getValue<float>("big_delta");
                        float sd= RegistrationSettings::get().getValue<float>("small_delta");

                        if(bd!=0 && sd!=0)
                        {
                            big_delta=bd;
                            small_delta=sd;
                        }
                        else
                        {
                            //If the small and big deltas are unknown, just make a guesstimate
                            //using the max bvalue and assumed gradient strength
                            double gyro= 267.51532*1E6;
                            double G= 40*1E-3;  //well most scanners are either 40 mT/m or 80mT/m.
                            if(this->jsons[PE]["ManufacturersModelName"]!=json::value_t::null)
                            {
                                std::string scanner_model=this->jsons[PE]["ManufacturersModelName"];
                                if(scanner_model.find("Prisma")!=std::string::npos)
                                    G= 80*1E-3;
                            }
                            double temp= max_bval/gyro/gyro/G/G/2.*1E6;
                            // assume that big_delta = 3 * small_delta
                            // deltas are in miliseconds
                            small_delta= pow(temp,1./3.)*1000.;
                            big_delta= small_delta*3;
                        }
                        this->jsons[PE]["BigDelta"]= big_delta;
                        this->jsons[PE]["SmallDelta"]= small_delta;
                    }
                    else
                    {
                        big_delta=this->jsons[PE]["BigDelta"];
                        small_delta=this->jsons[PE]["SmallDelta"];
                    }

                    mapmri_estimator2.SetMAPMRIDegree(2);
                    mapmri_estimator2.SetDTImg(dti_estimator.GetOutput());
                    mapmri_estimator2.SetA0Image(dti_estimator.GetA0Image());
                    mapmri_estimator2.SetBmatrix(rot_Bmat);
                    mapmri_estimator2.SetDWIData(final_data);
                    mapmri_estimator2.SetWeightImage(final_weight_imgs);
                    mapmri_estimator2.SetVoxelwiseBmatrix(dummyv);
                    mapmri_estimator2.SetMaskImage(final_mask);
                    mapmri_estimator2.SetVolIndicesForFitting(dummy);
                    mapmri_estimator2.SetSmallDelta(small_delta);
                    mapmri_estimator2.SetBigDelta(big_delta);
                    mapmri_estimator2.PerformFitting();

                    mapmri_estimator4.SetMAPMRIDegree(4);
                    mapmri_estimator4.SetDTImg(dti_estimator.GetOutput());
                    mapmri_estimator4.SetA0Image(dti_estimator.GetA0Image());
                    mapmri_estimator4.SetBmatrix(rot_Bmat);
                    mapmri_estimator4.SetDWIData(final_data);
                    mapmri_estimator4.SetWeightImage(final_weight_imgs);
                    mapmri_estimator4.SetVoxelwiseBmatrix(dummyv);
                    mapmri_estimator4.SetMaskImage(final_mask);
                    mapmri_estimator4.SetVolIndicesForFitting(dummy);
                    mapmri_estimator4.SetSmallDelta(small_delta);
                    mapmri_estimator4.SetBigDelta(big_delta);
                    mapmri_estimator4.PerformFitting();

                  //  mapmri_estimator6.SetMAPMRIDegree(FINAL_STAGE_MAPMRI_DEGREE);
                  //  mapmri_estimator6.SetDTImg(dti_estimator.GetOutput());
                  // mapmri_estimator6.SetA0Image(dti_estimator.GetA0Image());
                  //  mapmri_estimator6.SetBmatrix(rot_Bmat);
                  //  mapmri_estimator6.SetDWIData(final_data);
                  //  mapmri_estimator6.SetWeightImage(final_weight_imgs);
                  //  mapmri_estimator6.SetVoxelwiseBmatrix(dummyv);
                  //  mapmri_estimator6.SetMaskImage(final_mask);
                  //  mapmri_estimator6.SetVolIndicesForFitting(dummy);
                  //  mapmri_estimator6.SetSmallDelta(small_delta);
                  //  mapmri_estimator6.SetBigDelta(big_delta);
                  //  mapmri_estimator6.PerformFitting();
                }



                #pragma omp parallel for
                for(int vol=0;vol<Nvols[PE];vol++)
                {
                    TORTOISE::EnableOMPThread();

                    ImageType3D::Pointer synth_img_dt= dti_estimator.SynthesizeDWI( rot_Bmat.get_row(vol) );
                    ImageType3D::Pointer synth_img_mapmri2=nullptr, synth_img_mapmri4=nullptr,synth_img_mapmri6=nullptr;

                    if(MAPMRI_indices.size()>0)
                    {
                        synth_img_mapmri2= mapmri_estimator2.SynthesizeDWI( rot_Bmat.get_row(vol));
                        synth_img_mapmri4= mapmri_estimator4.SynthesizeDWI( rot_Bmat.get_row(vol));
                   //     synth_img_mapmri6= mapmri_estimator6.SynthesizeDWI( rot_Bmat.get_row(vol));

                        itk::ImageRegionIteratorWithIndex<ImageType3D> it(synth_img_dt,synth_img_dt->GetLargestPossibleRegion());

                        std::vector<float> non_brain_dt_vals;
                        for(it.GoToBegin();!it.IsAtEnd();++it)
                        {
                            ImageType3D::IndexType ind3= it.GetIndex();
                            if(final_mask->GetPixel(ind3)==0)
                            {
                                non_brain_dt_vals.push_back(it.Get());
                            }
                        }
                        double nonbrain_dt_median=median(non_brain_dt_vals);


                        for(it.GoToBegin();!it.IsAtEnd();++it)
                        {
                            ImageType3D::IndexType ind3= it.GetIndex();
                            int Ncount=0;
                            for(int vv=0;vv<Nvols[PE];vv++)
                            {
                                if(final_weight_imgs[vv]->GetPixel(ind3)> THR)
                                    Ncount++;
                            }
                            if(final_mask->GetPixel(ind3))
                            {
                                double dti_val= it.Get();
                                double map_val=0;

                                //if(bvals[vol]> 55 && Ncount>12 && Ncount < 30)
                                 //   map_val=synth_img_mapmri2->GetPixel(ind3);
                                //else if(bvals[vol]> 55 && Ncount>30 && Ncount < 70)
                                //    map_val=synth_img_mapmri4->GetPixel(ind3);
                                //else if(bvals[vol]> 55  && Ncount > 70)
                                //    map_val=synth_img_mapmri6->GetPixel(ind3);

                                if(bvals[vol]> 55 && Ncount>12 && Ncount < 30)
                                    map_val=synth_img_mapmri2->GetPixel(ind3);
                                else if(bvals[vol]> 55 && Ncount>30 )
                                    map_val=synth_img_mapmri4->GetPixel(ind3);


                                if(map_val> 0.5*nonbrain_dt_median)
                                    it.Set(map_val);

                            }
                        }

                    }
                    synth_imgs[vol] =synth_img_dt;



                    /*
                        synth_imgs[vol] = mapmri_estimator.SynthesizeDWI( rot_Bmat.get_row(vol));
                    else
                        synth_imgs[vol]= dti_estimator.SynthesizeDWI( rot_Bmat.get_row(vol) );
                        */

                    TORTOISE::DisableOMPThread();
                }
            }


            // SECOND.  Sometimes there is a big signal scale difference between the synthesized and actual images.
            // This can be due to many things. B1 inhomogeneity difference due to motion, being one of them.
            // So lets try to make the synthesized and actual images as close as possible with some
            // spatially dependent signal normalization and using non-outlier noise from images on the same shell


            #pragma omp parallel for
            for(int vol=0;vol<Nvols[PE];vol++)
            {
                TORTOISE::EnableOMPThread();

                //Generate a final mask that includes both the brain mask and outlier mask
                using FilterType = itk::MultiplyImageFilter<ImageType3D, ImageType3D, ImageType3D>;
                FilterType::Pointer filter2 = FilterType::New();
                filter2->SetInput2(final_mask);
                filter2->SetInput1(final_inclusion_imgs[vol]);
                filter2->Update();
                ImageType3D::Pointer weight_img_final= filter2->GetOutput();

                //Check if we have a valid voxel
                bool allzeros=true;
                itk::ImageRegionIterator<ImageType3D> it(weight_img_final,weight_img_final->GetLargestPossibleRegion());
                it.GoToBegin();
                while(!it.IsAtEnd())
                {
                    if(it.Get())
                    {
                        allzeros=false;
                        break;
                    }
                    ++it;
                }

                // get the volume ids with the closest bmat_vec by sorting the distances to the current one
                // we will use the noise from these images in background regions if necessary
                vnl_vector<double> bmat_vec = rot_Bmat.get_row(vol);
                std::vector<mypair> dists;
                for(int v2=0;v2<Nvols[PE];v2++)
                {
                    float dist = (rot_Bmat.get_row(v2) -bmat_vec).magnitude();
                    dists.push_back({dist,v2});
                }
                std::sort (dists.begin(), dists.end(), comparator);

                ImageType3D::SizeType sz= final_data[vol]->GetLargestPossibleRegion().GetSize();
                ImageType3D::Pointer synth_img=synth_imgs[vol];

                // if the native space image is only outliers
                // replace the entire foreground with predicted signals
                // and background from the closest inliner image voxel
                if(allzeros)
                {
                    final_data[vol]->FillBuffer(0);
                    itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_data[vol],final_data[vol]->GetLargestPossibleRegion() );
                    for(it.GoToBegin();!it.IsAtEnd();++it)
                    {
                        ImageType3D::IndexType ind3=it.GetIndex();
                        if(final_mask->GetPixel(ind3))
                            it.Set(synth_img->GetPixel(ind3));

                        else
                        {
                            //get noise from a similar image for the background
                            for(int v2=0;v2<Nvols[PE];v2++)
                            {
                                if( fabs(bvals[vol]-bvals[dists[v2].second])<10)
                                {
                                    if(final_inclusion_imgs[dists[v2].second]->GetPixel(ind3)==1)
                                    {
                                        float val=final_data[dists[v2].second]->GetPixel(ind3);
                                        it.Set(val);
                                        break;
                                    }
                                }
                            }
                        }

                    }
                }
                else
                {
                    //Not only outliers

                    // Perform B1 inhomogeneity correction on volume
                    // This causes the predicted and actual signal to differ sometimes.

                    using N4FilterType= itk::N4BiasFieldCorrectionImageFilter<ImageType3D, ImageType3D, ImageType3D>;
                    N4FilterType::Pointer n4_filter= N4FilterType::New();
                    n4_filter->SetInput(final_data[vol]);
                    n4_filter->SetMaskImage(weight_img_final);
                    n4_filter->Update();
                    ImageType3D::Pointer final_data2 = n4_filter->GetOutput();



                    // Okay.. The next part is a little bit heuristic.
                    // Well really heuristic

                    // The idea is as follows:
                    // If we have a band of outlier voxels, adjacent to a band of inlier voxels,
                    // we want a smooth transition in between.
                    // sometimes this does not happen because of MANY issues
                    // Even though quantitatively this is okay, visually it looks weird.
                    // in case this happens, there is some weird spatially dependent signal scaling going on.
                    // so lets try to find what it is by comparing the signals within the two bands.

                    for(int k=0;k<sz[2];k++)
                    {
                        ImageType3D::IndexType ind3;
                        ind3[2]=k;

                        //First create a slice image
                        using ImageType2DBool= itk::Image<char,2>;

                        ImageType2DBool::SizeType sz2;
                        sz2[0]=sz[0];
                        sz2[1]=sz[1];
                        ImageType2DBool::IndexType st2; st2.Fill(0);
                        ImageType2DBool::RegionType reg2(st2,sz2);
                        ImageType2DBool::Pointer sl_img=ImageType2DBool::New();
                        sl_img->SetRegions(reg2);
                        sl_img->Allocate();

                        for(int j=0;j<sz[1];j++)
                        {
                            ind3[1]=j;
                            ImageType2DBool::IndexType ind2;
                            ind2[1]=j;
                            for(int i=0;i<sz[0];i++)
                            {
                                ind3[0]=i;
                                ind2[0]=i;
                                sl_img->SetPixel(ind2, final_inclusion_imgs[vol]->GetPixel(ind3));
                            }
                        }

                        //Then find inlinear and outlier intersection on the slice with morphological operations
                        using StructuringElementType = itk::FlatStructuringElement<2>;
                        StructuringElementType::RadiusType radius;
                        radius.Fill(1);
                        StructuringElementType structuringElement = StructuringElementType::Ball(radius);

                        using BinaryErodeImageFilterType = itk::BinaryErodeImageFilter<ImageType2DBool, ImageType2DBool, StructuringElementType>;
                        BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
                        erodeFilter->SetInput(sl_img);
                        erodeFilter->SetKernel(structuringElement);
                        erodeFilter->SetForegroundValue(1); // Intensity value to erode
                        erodeFilter->SetBackgroundValue(0);   // Replacement value for eroded voxels
                        erodeFilter->Update();
                        ImageType2DBool::Pointer sl_img_erode= erodeFilter->GetOutput();



                        bool entered=false;

                        for(int j=0;j<sz[1];j++)
                        {
                            ImageType2DBool::IndexType ind2;
                            ind3[1]=j;
                            ind2[1]=j;
                            for(int i=0;i<sz[0];i++)
                            {
                                ind3[0]=i;
                                ind2[0]=i;


                                if(final_inclusion_imgs[vol]->GetPixel(ind3)==0)
                                {
                                    float val=final_data2->GetPixel(ind3);

                                    if(final_mask->GetPixel(ind3)==0)
                                    {
                                        //get noise from a similar image for the background
                                        for(int v2=0;v2<Nvols[PE];v2++)
                                        {
                                            if( fabs(bvals[vol]-bvals[dists[v2].second])<10)
                                            {
                                                if(final_inclusion_imgs[dists[v2].second]->GetPixel(ind3)==1)
                                                {
                                                    val=final_data[dists[v2].second]->GetPixel(ind3);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        // foreground region
                                        val=  synth_imgs[vol]->GetPixel(ind3);
                                    }
                                    final_data[vol]->SetPixel(ind3,val);
                                }
                                else
                                {
                                    // On the intersection line (still in outlier region), average the two values
                                    float val=final_data2->GetPixel(ind3);
                                    if(sl_img_erode->GetPixel(ind2)==0 && final_mask->GetPixel(ind3) && synth_img->GetPixel(ind3)>0)
                                    {
                                          val= 0.5* (val + synth_img->GetPixel(ind3));
                                    }
                                    else
                                    {
                                        // here we use weights instead of binarized versions.
                                        // that is because if the weight is for example  0.06 and the THR is 0.05
                                        // it is still possible for the voxel to be outlier
                                        // in case clustering was not 100% accurate

                                        if(final_mask->GetPixel(ind3) && synth_img->GetPixel(ind3)>0)
                                        {
                                            float val_real= final_data2->GetPixel(ind3);
                                            float val_synth= synth_imgs[vol]->GetPixel(ind3);

                                            float w= final_weight_imgs[vol]->GetPixel(ind3);
                                            val=  w * val_real + (1-w) * val_synth;
                                        }
                                    }

                                    final_data[vol]->SetPixel(ind3,val);
                                }

                            } //for i
                        } //for j
                    } //for k


                } //if not allzeros


                TORTOISE::DisableOMPThread();
            } //for vol
            (*stream)<<std::endl;

            fs::path up_path(up_name);
            std::string basename= fs::path(up_path).filename().string();
            basename=basename.substr(0,basename.rfind(".nii"));
            std::string new_synth_name=  this->temp_folder + "/" + basename + "_final_temp_synth.nii";
            for(int v=0;v<Nvols[PE];v++)
            {
                write_3D_image_to_4D_file<float>(synth_imgs[v],new_synth_name,v,Nvols[PE]);
            }

        } //if repol



        std::string new_nii_name=  this->temp_folder + "/" + basename + "_final_temp.nii";
        for(int v=0;v<Nvols[PE];v++)
        {
            write_3D_image_to_4D_file<float>(final_data[v],new_nii_name,v,Nvols[PE]);
        }

        std::string new_noise_name=  this->temp_folder + "/" + basename + "_final_temp_noise.nii";
        writeImageD<ImageType3D>(final_noise_img,new_noise_name);



        final_imgs_to_return[PE]=final_data;

    } //for PE

    return final_imgs_to_return;
}



InternalMatrixType  FINALDATA::pixel_bmatrix(const GradCoef &E, ImageType3D::PointType point,const vnl_vector<double> &norms)
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


std::vector<ImageType3D::Pointer> FINALDATA::ComputeVBMatImgFromCoeffs(int PE)
{

    std::vector<ImageType3D::Pointer> inverse_slice_id_img;
    if(this->s2v_transformations[PE].size())
    {
        inverse_slice_id_img= ComputeS2VInverse(PE);
    }

    std::string coeffs_file = RegistrationSettings::get().getValue<std::string>("grad_nonlin_coeffs");

    GRADCAL *grads = nullptr;
    GradCoef E;

    if(coeffs_file!="")
    {
        grads = new GRADCAL(coeffs_file);
        E = grads->get_struct();
    }

    bool is_GE= RegistrationSettings::get().getValue<bool>("grad_nonlin_isGE");

    std::string nii_name= data_names[PE];
    std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii"))+".bmtxt";
    vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);

    ImageType3D::Pointer first_vol = read_3D_volume_from_4D(data_names[PE],0);
    ImageType3D::Pointer first_vol_DP = ChangeImageHeaderToDP<ImageType3D>(first_vol);
    ImageType3D::Pointer first_vol_grad=first_vol;


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


    int nvols= Bmatrix.rows();
    std::vector<ImageType3D::Pointer> vbmat;
    vbmat.resize(nvols*6);

    for(int vol=0;vol<nvols*6;vol++)
    {
        vbmat[vol]=ImageType3D::New();
        vbmat[vol]->SetRegions(this->template_structural->GetLargestPossibleRegion());
        vbmat[vol]->Allocate();
        vbmat[vol]->SetDirection(this->template_structural->GetDirection());
        vbmat[vol]->SetSpacing(this->template_structural->GetSpacing());
        vbmat[vol]->SetOrigin(this->template_structural->GetOrigin());
        vbmat[vol]->FillBuffer(0);
    }

    ImageType3D::SizeType sz= this->template_structural->GetLargestPossibleRegion().GetSize();


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
                this->template_structural->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(this->b0_t0_str_trans)
                    pt_trans=this->b0_t0_str_trans->TransformPoint(pt);

                if(PE==1 && this->b0down_t0_b0up_trans)
                    pt_trans= this->b0down_t0_b0up_trans->TransformPoint(pt_trans);

                itk::ContinuousIndex<double,3> cind;
                first_vol->TransformPhysicalPointToContinuousIndex(pt_trans,cind);


                ImageType3D::PointType pt_DP;
                first_vol_DP->TransformContinuousIndexToPhysicalPoint(cind,pt_DP);

                for(int vol=0;vol<nvols;vol++)
                {
                    // Get the point the DWI physical space

                    //For localizing the position of the voxel in actual physical space
                    // only consider motion, not eddy currents and other trans

                    OkanQuadraticTransformType::Pointer temp_trans=OkanQuadraticTransformType::New();
                    temp_trans->SetPhase(this->PE_strings[PE]);
                    temp_trans->SetIdentity();

                    OkanQuadraticTransformType::ParametersType quad_params= dwi_transforms[PE][vol]->GetParameters();
                    OkanQuadraticTransformType::ParametersType temp_params= temp_trans->GetParameters();
                    temp_params[0]=quad_params[0];
                    temp_params[1]=quad_params[1];
                    temp_params[2]=quad_params[2];
                    temp_params[3]=quad_params[3];
                    temp_params[4]=quad_params[4];
                    temp_params[5]=quad_params[5];
                    temp_trans->SetParameters(temp_params);

                    ImageType3D::PointType pt_DP_trans= temp_trans->TransformPoint(pt_DP);
                    first_vol_DP->TransformPhysicalPointToContinuousIndex(pt_DP_trans,cind);
                    first_vol->TransformContinuousIndexToPhysicalPoint(cind,pt_trans);


                    ImageType3D::PointType pt_scanner_space= pt_trans;
                    vnl_matrix_fixed<double,1,6> curr_bmat_row = Bmatrix.get_n_rows(vol,1);
                    OkanQuadraticTransformType::MatrixType  R; R.SetIdentity();
                    if(this->s2v_transformations[PE].size())
                    {
                        ImageType3D::IndexType ind3_n;
                        for(int ti=0;ti<ImageType3D::ImageDimension;ti++)
                        {
                            if(cind[ti]<0)
                                ind3_n[ti]=0;
                            else
                                ind3_n[ti]=(unsigned int) std::round(cind[ti]);

                            if(cind[ti]>first_vol->GetLargestPossibleRegion().GetSize()[ti]-1)
                                ind3_n[ti]=first_vol->GetLargestPossibleRegion().GetSize()[ti]-1;
                        }

                        int reverse_slice_id = inverse_slice_id_img[vol]->GetPixel(ind3_n);
                        OkanQuadraticTransformType::Pointer curr_s2v_trans = this->s2v_transformations[PE][vol][reverse_slice_id];
                        R= curr_s2v_trans->GetMatrix();
                        OkanQuadraticTransformType::OutputVectorType  T= curr_s2v_trans->GetTranslation();

                        vnl_vector<double>  pt_scanner_space_vnl= R.GetVnlMatrix().transpose() *  (pt_trans.GetVnlVector() - T.GetVnlVector());
                        pt_scanner_space[0]=pt_scanner_space_vnl[0];
                        pt_scanner_space[1]=pt_scanner_space_vnl[1];
                        pt_scanner_space[2]=pt_scanner_space_vnl[2];

                        first_vol->TransformPhysicalPointToContinuousIndex(pt_scanner_space,cind);
                    }


                    if(grads)
                    {
                        InternalMatrixType Lmat; Lmat.set_identity();

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


                        //We have output Lmat in its correct form as Jacobian
                        //Howerver, because forward backward nature of the transformation
                        //we should have taken its transpose (actually inverse but noone does it that way, which I think is incorrect)
                        //so to make it directly applicable to Bmatrix, we have to take another transpose
                        Lmat=Lmat.transpose();

                        // now let's transform Lmat to ijk space  of the native space image
                        Lmat = first_vol_grad->GetDirection().GetTranspose() *
                               Lmat  *
                               first_vol_grad->GetDirection().GetVnlMatrix();


                        //curr_bmat_mat is in native ijk space
                        vnl_matrix_fixed<double,3,3> curr_bmat_mat;
                        curr_bmat_mat(0,0)=curr_bmat_row(0,0);curr_bmat_mat(0,1)=curr_bmat_row(0,1)/2; curr_bmat_mat(0,2)=curr_bmat_row(0,2)/2;
                        curr_bmat_mat(1,0)=curr_bmat_row(0,1)/2;curr_bmat_mat(1,1)=curr_bmat_row(0,3); curr_bmat_mat(1,2)=curr_bmat_row(0,4)/2;
                        curr_bmat_mat(2,0)=curr_bmat_row(0,2)/2;curr_bmat_mat(2,1)=curr_bmat_row(0,4)/2; curr_bmat_mat(2,2)=curr_bmat_row(0,5);

                         // let's transform the Bmatrix
                        curr_bmat_mat =  Lmat * curr_bmat_mat *  Lmat.transpose();
                        //curr_bmat_mat is still in native ijk space so is curr_bmat_row

                        curr_bmat_row(0,0)=curr_bmat_mat(0,0);
                        curr_bmat_row(0,1)=curr_bmat_mat(0,1)*2;
                        curr_bmat_row(0,2)=curr_bmat_mat(0,2)*2;
                        curr_bmat_row(0,3)=curr_bmat_mat(1,1);
                        curr_bmat_row(0,4)=curr_bmat_mat(1,2)*2;
                        curr_bmat_row(0,5)=curr_bmat_mat(2,2);
                    }

                    // Let's rotate the Bmatrix with ALL the transforms, i.e. s2v, moteddy, structural alignment
                    if(this->s2v_transformations[PE].size())
                        curr_bmat_row= RotateBMatrix(curr_bmat_row,R.GetTranspose(),first_vol->GetDirection().GetVnlMatrix(),first_vol->GetDirection().GetVnlMatrix());
                    curr_bmat_row= RotateBMatrix(curr_bmat_row,dwi_transforms[PE][vol]->GetMatrix().GetVnlMatrix(),id_trans,id_trans);
                    if(this->b0_t0_str_trans)
                        curr_bmat_row= RotateBMatrix(curr_bmat_row,this->b0_t0_str_trans->GetMatrix().GetVnlMatrix(),this->template_structural->GetDirection().GetVnlMatrix(),first_vol->GetDirection().GetVnlMatrix());
                    //curr_bmat_row is now in ijk space of template

                    for(int vv=0;vv<6;vv++)
                        vbmat[6*vol+vv]->SetPixel(ind3,curr_bmat_row(0,vv));

                } //for vol
            } //for i
        } //for j
    } //for k

    delete grads;
    return vbmat;
}


std::vector<ImageType3D::Pointer> FINALDATA::ComputeLImgFromCoeffs()
{
    std::string coeffs_file = RegistrationSettings::get().getValue<std::string>("grad_nonlin_coeffs");

    GRADCAL *grads = nullptr;
    GradCoef E;

    if(coeffs_file!="")
    {
        grads = new GRADCAL(coeffs_file);
        E = grads->get_struct();
    }

    bool is_GE= RegistrationSettings::get().getValue<bool>("grad_nonlin_isGE");


    ImageType3D::Pointer first_vol = read_3D_volume_from_4D(data_names[0],0);
    ImageType3D::Pointer first_vol_grad=first_vol;

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
        Limg[v]->SetRegions(this->template_structural->GetLargestPossibleRegion());
        Limg[v]->Allocate();
        Limg[v]->SetDirection(this->template_structural->GetDirection());
        Limg[v]->SetSpacing(this->template_structural->GetSpacing());
        Limg[v]->SetOrigin(this->template_structural->GetOrigin());
        Limg[v]->FillBuffer(0);
    }

    ImageType3D::SizeType sz= this->template_structural->GetLargestPossibleRegion().GetSize();


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
                this->template_structural->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(this->b0_t0_str_trans)
                    pt_trans=this->b0_t0_str_trans->TransformPoint(pt);

                itk::ContinuousIndex<double,3> cind;
                first_vol->TransformPhysicalPointToContinuousIndex(pt_trans,cind);

                InternalMatrixType Lmat; Lmat.set_identity();
                if(grads)
                {
                    ImageType3D::PointType pt_itk_space, pt_scanner_space;
                    first_vol_grad->TransformContinuousIndexToPhysicalPoint(cind,pt_itk_space);
                    // pt_scanner_space is in ITK xyz coordinate system

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

                //It doesnt matter if subtract Id before this.. It is identical
                // but we cant subtract it after template warping so we do it here

                //  Lmat= Lmat - id_trans;

                // ON 07/27/23, I decided to NOT output in HCP format with id subtracted so commented the above line
                // Mathematically it makes things insanely complicated if we want to save L in ijk space
                // That is why HCP people save it in xyz space but I dont want to do it that way.
                // The Bmatrix AND Lmat should live in the same space
                // so NO ID subtracted.


                //Now get L from native ijk space to template ijk space
                //yes some double unnecessary operations that cancel each other out here
                // but makes reading wrt math easier


                if(this->template_structural)
                {
                    Lmat= first_vol_grad->GetDirection().GetVnlMatrix() *Lmat * first_vol_grad->GetDirection().GetTranspose();
                    if(this->b0_t0_str_trans)
                    {
                        Lmat= this->b0_t0_str_trans->GetMatrix().GetTranspose() *Lmat * this->b0_t0_str_trans->GetMatrix().GetVnlMatrix();
                    }
                    Lmat = this->template_structural->GetDirection().GetTranspose() *Lmat *this->template_structural->GetDirection().GetVnlMatrix() ;
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



std::vector<ImageType3D::Pointer>  FINALDATA::ComputeS2VInverse(int PE)
{
    std::string nii_name= data_names[PE];
    std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii"))+".bmtxt";
    vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);

    ImageType3D::Pointer first_vol = read_3D_volume_from_4D(data_names[PE],0);
    int nvols= Bmatrix.rows();

    ImageType3D::SizeType sz= first_vol->GetLargestPossibleRegion().GetSize();

    std::vector<ImageType3D::Pointer> inverse_slice_id_img;
    inverse_slice_id_img.resize(nvols);

    #pragma omp parallel for
    for(int vol=0;vol<nvols;vol++)
    {
        inverse_slice_id_img[vol]= ImageType3D::New();
        inverse_slice_id_img[vol]->SetRegions(first_vol->GetLargestPossibleRegion());
        inverse_slice_id_img[vol]->Allocate();
        inverse_slice_id_img[vol]->FillBuffer(-1);

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

                    ImageType3D::PointType pt,pt_trans;
                    first_vol->TransformIndexToPhysicalPoint(ind3,pt);
                    pt_trans=this->s2v_transformations[PE][vol][k]->TransformPoint(pt);

                    itk::ContinuousIndex<double,3> cind3;
                    first_vol->TransformPhysicalPointToContinuousIndex(pt_trans,cind3);

                    ImageType3D::IndexType ind3_n;
                    for(int ti=0;ti<3;ti++)
                    {
                        if(cind3[ti]<0)
                            ind3_n[ti]=0;
                        else
                            ind3_n[ti]=(unsigned int) std::round(cind3[ti]);

                        if(cind3[ti]>sz[ti]-1)
                            ind3_n[ti]=sz[ti]-1;
                    }


                    float dist= (cind3[0]- ind3_n[0])*(cind3[0]- ind3_n[0])+
                                (cind3[1]- ind3_n[1])*(cind3[1]- ind3_n[1])+
                                (cind3[2]- ind3_n[2])*(cind3[2]- ind3_n[2]);
                    dist=sqrt(dist)/1000.;

                    if(inverse_slice_id_img[vol]->GetPixel(ind3_n)==-1)
                    {
                        inverse_slice_id_img[vol]->SetPixel(ind3_n,k+dist);
                    }
                    else
                    {
                        float curr_dist =  inverse_slice_id_img[vol]->GetPixel(ind3_n) -(int)(inverse_slice_id_img[vol]->GetPixel(ind3_n));
                        if(dist<curr_dist)
                        {
                            inverse_slice_id_img[vol]->SetPixel(ind3_n,k+dist);
                        }
                    }
                }
            }
        }

        // look for unfillled pixels and interpolate
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
                    if(inverse_slice_id_img[vol]->GetPixel(ind3)==-1)
                    {
                        bool found=false;
                        int NI=1;

                        while(!found)
                        {
                            ImageType3D::IndexType ind3t;
                            double val=0;
                            double sm_weight=0;

                            for(int kk=k-NI;kk<=k+NI;kk++)
                            {
                                if(kk>=0 && kk<sz[2])
                                {
                                    ind3t[2]=kk;
                                    for(int jj=j-NI;jj<=j+NI;jj++)
                                    {
                                        if(jj>=0 && jj<sz[1])
                                        {
                                            ind3t[1]=jj;
                                            for(int ii=i-NI;ii<=i+NI;ii++)
                                            {
                                                if(ii>=0 && ii<sz[0])
                                                {
                                                    ind3t[0]=ii;

                                                    if(abs(ii-i)==NI || abs(jj-j)==NI || abs(kk-k)==NI)  //traverse the faces
                                                    {
                                                        if(inverse_slice_id_img[vol]->GetPixel(ind3t)!=-1)
                                                        {
                                                            found =true;
                                                            float dist= abs(ii-i) + abs(jj-j)+abs(kk-k);
                                                            val += inverse_slice_id_img[vol]->GetPixel(ind3t) * (1./dist);
                                                            sm_weight+= (1./dist);
                                                        }
                                                    }
                                                }
                                            }

                                        }
                                    }

                                }
                            }
                            if(found)
                            {
                               val/=sm_weight;
                               int vali = (int) std::round(val);
                               inverse_slice_id_img[vol]->SetPixel(ind3,vali);
                            }
                            else
                                NI++;
                        } //while !found
                    } //if -1
                }
            }
        }
    } //for vol

    return inverse_slice_id_img;


}


std::vector<ImageType3D::Pointer> FINALDATA::ComputeVBMatImgFromField(int PE)
{
    std::vector<ImageType3D::Pointer> inverse_slice_id_img;
    if(this->s2v_transformations[PE].size())
    {
        inverse_slice_id_img= ComputeS2VInverse(PE);
    }

    std::string nii_name= data_names[PE];
    std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii"))+".bmtxt";
    vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);


    ImageType3D::Pointer first_vol = read_3D_volume_from_4D(data_names[PE],0);
    ImageType3D::Pointer first_vol_DP = ChangeImageHeaderToDP<ImageType3D>(first_vol);

    int nvols= Bmatrix.rows();
    std::vector<ImageType3D::Pointer> vbmat;
    vbmat.resize(nvols*6);

    for(int vol=0;vol<nvols*6;vol++)
    {
        vbmat[vol]=ImageType3D::New();
        vbmat[vol]->SetRegions(this->template_structural->GetLargestPossibleRegion());
        vbmat[vol]->Allocate();
        vbmat[vol]->SetDirection(this->template_structural->GetDirection());
        vbmat[vol]->SetSpacing(this->template_structural->GetSpacing());
        vbmat[vol]->SetOrigin(this->template_structural->GetOrigin());
        vbmat[vol]->FillBuffer(0);
    }

    ImageType3D::SizeType sz= this->template_structural->GetLargestPossibleRegion().GetSize();

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
                this->template_structural->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(this->b0_t0_str_trans)
                    pt_trans=this->b0_t0_str_trans->TransformPoint(pt);

                if(PE==1 && this->b0down_t0_b0up_trans)
                    pt_trans= this->b0down_t0_b0up_trans->TransformPoint(pt_trans);

                itk::ContinuousIndex<double,3> cind;
                first_vol->TransformPhysicalPointToContinuousIndex(pt_trans,cind);

                ImageType3D::PointType pt_DP;
                first_vol_DP->TransformContinuousIndexToPhysicalPoint(cind,pt_DP);


                for(int vol=0;vol<nvols;vol++)
                {
                    // Get the point the DWI physical space

                    //For localizing the position of the voxel in actual physical space
                    // only consider motion, not eddy currents and other trans

                    OkanQuadraticTransformType::Pointer temp_trans=OkanQuadraticTransformType::New();
                    temp_trans->SetPhase(this->PE_strings[PE]);
                    temp_trans->SetIdentity();

                    OkanQuadraticTransformType::ParametersType quad_params= dwi_transforms[PE][vol]->GetParameters();
                    OkanQuadraticTransformType::ParametersType temp_params= temp_trans->GetParameters();
                    temp_params[0]=quad_params[0];
                    temp_params[1]=quad_params[1];
                    temp_params[2]=quad_params[2];
                    temp_params[3]=quad_params[3];
                    temp_params[4]=quad_params[4];
                    temp_params[5]=quad_params[5];
                    temp_trans->SetParameters(temp_params);


                    ImageType3D::PointType pt_DP_trans= temp_trans->TransformPoint(pt_DP);
                    first_vol_DP->TransformPhysicalPointToContinuousIndex(pt_DP_trans,cind);
                    first_vol->TransformContinuousIndexToPhysicalPoint(cind,pt_trans);

                    ImageType3D::PointType pt_scanner_space= pt_trans;

                    vnl_matrix<double> curr_bmat_row = Bmatrix.get_n_rows(vol,1);
                    OkanQuadraticTransformType::MatrixType  R; R.SetIdentity();
                    if(this->s2v_transformations[PE].size())
                    {
                        ImageType3D::IndexType ind3_n;
                        for(int ti=0;ti<3;ti++)
                        {
                            if(cind[ti]<0)
                                ind3_n[ti]=0;
                            else
                                ind3_n[ti]=(unsigned int) std::round(cind[ti]);

                            if(cind[ti]>first_vol->GetLargestPossibleRegion().GetSize()[ti]-1)
                                ind3_n[ti]=first_vol->GetLargestPossibleRegion().GetSize()[ti]-1;
                        }

                        int reverse_slice_id = inverse_slice_id_img[vol]->GetPixel(ind3_n);
                        OkanQuadraticTransformType::Pointer curr_s2v_trans = this->s2v_transformations[PE][vol][reverse_slice_id];
                        R= curr_s2v_trans->GetMatrix();
                        OkanQuadraticTransformType::OutputVectorType  T= curr_s2v_trans->GetTranslation();

                        vnl_vector<double>  pt_scanner_space_vnl= R.GetVnlMatrix().transpose() *  (pt_trans.GetVnlVector() - T.GetVnlVector());
                        pt_scanner_space[0]=pt_scanner_space_vnl[0];
                        pt_scanner_space[1]=pt_scanner_space_vnl[1];
                        pt_scanner_space[2]=pt_scanner_space_vnl[2];                                                
                    }

                    if(this->gradwarp_field)
                    {
                        this->gradwarp_field->TransformPhysicalPointToContinuousIndex(pt_scanner_space,cind);
                        ImageType3D::IndexType jac_ind3;


                        for(int yy=0;yy<3;yy++)
                        {
                            if(cind[yy]<0)
                                jac_ind3[yy]=0;
                            else
                                jac_ind3[yy]= (unsigned int)std::round(cind[yy]);
                        }

                        // A is in ITK xyz space
                        InternalMatrixType A= ComputeJacobian(this->gradwarp_field,jac_ind3);

                        //A is still in backward convention, so let's take its transopose
                        // for the millionth time should have been inverse
                        A=A.transpose();

                        // now let's transform A to ijk space  of the native space image
                        A = first_vol->GetDirection().GetTranspose() *  A   * first_vol->GetDirection().GetVnlMatrix();

                        //curr_bmat_mat is in native ijk space
                        vnl_matrix_fixed<double,3,3> curr_bmat_mat;
                        curr_bmat_mat(0,0)=curr_bmat_row(0,0);curr_bmat_mat(0,1)=curr_bmat_row(0,1)/2; curr_bmat_mat(0,2)=curr_bmat_row(0,2)/2;
                        curr_bmat_mat(1,0)=curr_bmat_row(0,1)/2;curr_bmat_mat(1,1)=curr_bmat_row(0,3); curr_bmat_mat(1,2)=curr_bmat_row(0,4)/2;
                        curr_bmat_mat(2,0)=curr_bmat_row(0,2)/2;curr_bmat_mat(2,1)=curr_bmat_row(0,4)/2; curr_bmat_mat(2,2)=curr_bmat_row(0,5);


                        // generate the voxelwise Bmatrix
                        curr_bmat_mat = A* curr_bmat_mat * A.transpose();                        

                        curr_bmat_row(0,0)=curr_bmat_mat(0,0);
                        curr_bmat_row(0,1)=curr_bmat_mat(0,1)*2;
                        curr_bmat_row(0,2)=curr_bmat_mat(0,2)*2;
                        curr_bmat_row(0,3)=curr_bmat_mat(1,1);
                        curr_bmat_row(0,4)=curr_bmat_mat(1,2)*2;
                        curr_bmat_row(0,5)=curr_bmat_mat(2,2);

                    }

                    // Let's rotate the Bmatrix with ALL the transforms, i.e. s2v, moteddy, structural alignment
                    if(this->s2v_transformations[PE].size())
                        curr_bmat_row= RotateBMatrix(curr_bmat_row,R.GetTranspose(),first_vol->GetDirection().GetVnlMatrix(),first_vol->GetDirection().GetVnlMatrix());
                    curr_bmat_row= RotateBMatrix(curr_bmat_row,dwi_transforms[PE][vol]->GetMatrix().GetVnlMatrix(),id_trans,id_trans);
                    if(this->b0_t0_str_trans)
                        curr_bmat_row= RotateBMatrix(curr_bmat_row,this->b0_t0_str_trans->GetMatrix().GetVnlMatrix(),template_structural->GetDirection().GetVnlMatrix(),first_vol->GetDirection().GetVnlMatrix());

                    for(int vv=0;vv<6;vv++)
                        vbmat[6*vol+vv]->SetPixel(ind3,curr_bmat_row(0,vv));

                } //for vol
            } //for i
        } //for j
    } //for k

    return vbmat;

}

std::vector<ImageType3D::Pointer> FINALDATA::ComputeLImgFromField()
{
    ImageType3D::Pointer first_vol = read_3D_volume_from_4D(data_names[0],0);


    std::vector<ImageType3D::Pointer> Limg;
    Limg.resize(9);
    for(int v=0;v<9;v++)
    {
        Limg[v]=ImageType3D::New();
        Limg[v]->SetRegions(this->template_structural->GetLargestPossibleRegion());
        Limg[v]->Allocate();
        Limg[v]->SetDirection(this->template_structural->GetDirection());
        Limg[v]->SetSpacing(this->template_structural->GetSpacing());
        Limg[v]->SetOrigin(this->template_structural->GetOrigin());
        Limg[v]->FillBuffer(0);
    }

    ImageType3D::SizeType sz= this->template_structural->GetLargestPossibleRegion().GetSize();


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
                this->template_structural->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(this->b0_t0_str_trans)
                    pt_trans=this->b0_t0_str_trans->TransformPoint(pt);


                itk::ContinuousIndex<double,3> cind;
                first_vol->TransformPhysicalPointToContinuousIndex(pt_trans,cind);

                InternalMatrixType A; A.set_identity();
                if(this->gradwarp_field)
                {
                    this->gradwarp_field->TransformPhysicalPointToContinuousIndex(pt_trans,cind);
                    ImageType3D::IndexType jac_ind3;

                    for(int yy=0;yy<3;yy++)
                    {
                        if(cind[yy]<0)
                            jac_ind3[yy]=0;
                        else
                            jac_ind3[yy]= (unsigned int)std::round(cind[yy]);
                    }
                    A=ComputeJacobian(this->gradwarp_field,jac_ind3);  //A is in ITK xyz space


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

                //Now get L from native ijk space to template ijk space
                //yes some double unnecessary operations that cancel each other out here
                // but makes reading wrt math easier
                if(this->template_structural)
                {
                    A= first_vol->GetDirection().GetVnlMatrix() *A * first_vol->GetDirection().GetTranspose();
                    if(this->b0_t0_str_trans)
                    {
                        A= this->b0_t0_str_trans->GetMatrix().GetTranspose() *A * this->b0_t0_str_trans->GetMatrix().GetVnlMatrix();
                    }
                    A = this->template_structural->GetDirection().GetTranspose() *A *this->template_structural->GetDirection().GetVnlMatrix() ;
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



void FINALDATA::GenerateGradNonlinOutput()
{
    std::string output_gradnonlin_type = RegistrationSettings::get().getValue<std::string>("output_gradnonlin_Bmtxt_type");
    std::string data_combination_method = RegistrationSettings::get().getValue<std::string>("output_data_combination");
    std::string coeffs_file = RegistrationSettings::get().getValue<std::string>("grad_nonlin_coeffs");


    std::vector<ImageType3D::Pointer> graddev_vbmat_img_up,graddev_vbmat_img_down;

    if(!this->gradwarp_field && output_gradnonlin_type!="vbmat")
    {
        (*stream)<<"Voxelwise Bmatrices due to S2V can only be written with  output_gradnonlin_type set to vbmat."<<std::endl;
        return;
    }


    if(coeffs_file!="")
    {
        if(output_gradnonlin_type=="vbmat")
        {
            graddev_vbmat_img_up= ComputeVBMatImgFromCoeffs(0);
            if(data_combination_method!="Merge" && data_names[1]!="")
            {
                graddev_vbmat_img_down= ComputeVBMatImgFromCoeffs(1);
            }
        }
        else
        {
            graddev_vbmat_img_up= ComputeLImgFromCoeffs();
        }
    }
    else
    {
        if(output_gradnonlin_type=="vbmat")
        {
            graddev_vbmat_img_up= ComputeVBMatImgFromField(0);
            if(data_combination_method!="Merge" && data_names[1]!="")
            {
                graddev_vbmat_img_down= ComputeVBMatImgFromCoeffs(1);
            }
        }
        else
        {
            graddev_vbmat_img_up= ComputeLImgFromField();
        }
    }


    if(data_combination_method=="JacSep")
    {
        std::string final_folder= fs::path(this->output_name).parent_path().string();

        {
            std::string name = data_names[0];
            fs::path path(name);
            std::string basename= fs::path(path).filename().string();
            basename=basename.substr(0,basename.rfind(".nii"));
            std::string gb_name;
            if(output_gradnonlin_type=="vbmat")
                gb_name= final_folder + "/"+basename + "_TORTOISE_final_vbmat.nii";
            else
                gb_name= final_folder + "/"+basename + "_TORTOISE_final_graddev.nii";

            for(int v=0;v<graddev_vbmat_img_up.size();v++)
            {
                write_3D_image_to_4D_file<float>(graddev_vbmat_img_up[v],gb_name,v,graddev_vbmat_img_up.size());
            }
        }
        if(graddev_vbmat_img_down.size())
        {
            std::string name = data_names[1];
            fs::path path(name);
            std::string basename= fs::path(path).filename().string();
            basename=basename.substr(0,basename.rfind(".nii"));
            std::string gb_name;
            if(output_gradnonlin_type=="vbmat")
                gb_name= final_folder + "/"+basename + "_TORTOISE_final_vbmat.nii";
            else
                gb_name= final_folder + "/"+basename + "_TORTOISE_final_graddev.nii";

            for(int v=0;v<graddev_vbmat_img_down.size();v++)
            {
                write_3D_image_to_4D_file<float>(graddev_vbmat_img_down[v],gb_name,v,graddev_vbmat_img_down.size());
            }
        }
    }
    else
    {
        std::string gb_name;
        if(output_gradnonlin_type=="vbmat")
            gb_name= this->output_name.substr(0,this->output_name.rfind(".nii")) + "_vbmat.nii";
        else
            gb_name= this->output_name.substr(0,this->output_name.rfind(".nii")) + "_graddev.nii";

        int Ntot = graddev_vbmat_img_up.size()+graddev_vbmat_img_down.size();

        for(int v=0;v<graddev_vbmat_img_up.size();v++)
        {
            write_3D_image_to_4D_file<float>(graddev_vbmat_img_up[v],gb_name,v,Ntot);
        }
        if(graddev_vbmat_img_down.size())
        {
            for(int v=0;v<graddev_vbmat_img_up.size();v++)
            {
                write_3D_image_to_4D_file<float>(graddev_vbmat_img_down[v],gb_name,v+graddev_vbmat_img_up.size(),Ntot);
            }
        }
    }
}


void FINALDATA::Generate()
{
    this->template_structural = GenerateStructurals();
    ReadOrigTransforms();

    if(this->gradwarp_field || this->s2v_transformations[0].size())
    {
        std::cout<<"Generating gradient nonlinearity information..."<<std::endl;
        GenerateGradNonlinOutput();
        std::cout<<"Done..."<<std::endl;
    }


    std::vector< std::vector<ImageType3D::Pointer> > trans_interp_DWIs = GenerateTransformedInterpolatedData();
    GenerateFinalData(trans_interp_DWIs);




}


#endif
