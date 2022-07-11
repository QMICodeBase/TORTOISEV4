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
#include "itkBSplineControlPointImageFunction.h"


#include <Eigen/Dense>
using namespace itkeigen;

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
        if(fs::exists(this->temp_folder + std::string("/b0_corrected_final.nii")))
            target_img = readImageD<ImageType3D>(this->temp_folder + std::string("/b0_corrected_final.nii"));
        else if(fs::exists(this->temp_folder + std::string("/blip_up_b0_corrected_JAC.nii")))
            target_img= readImageD<ImageType3D>(this->temp_folder + std::string("/blip_up_b0_corrected_JAC.nii"));
        else
            target_img= read_3D_volume_from_4D(this->data_names[0],0);
    }

    target_img= ReorientImage3D(target_img, "", output_orientation);

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

        new_sz[0]= (int)std::ceil(FOV[0]/new_spc[0]);
        new_sz[1]= (int)std::ceil(FOV[1]/new_spc[1]);
        new_sz[2]= (int)std::ceil(FOV[2]/new_spc[2]);


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
    if(fs::exists(this->temp_folder +"/b0_to_str_rigidtrans.hdf5"))
    {
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(this->temp_folder +"/b0_to_str_rigidtrans.hdf5" );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();
        this->b0_t0_str_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
    }
    if(fs::exists(this->temp_folder +"/deformation_FINV.nii.gz"))
    {
        this->epi_trans[0]=readImageD<DisplacementFieldType>(this->temp_folder +"/deformation_FINV.nii.gz");
    }
    if(fs::exists(this->temp_folder +"/deformation_MINV.nii.gz"))
    {
        this->epi_trans[1]=readImageD<DisplacementFieldType>(this->temp_folder +"/deformation_MINV.nii.gz");
    }

    std::string gradnonlin_field_name= RegistrationSettings::get().getValue<std::string>("grad_nonlin");
    std::string gradnonlin_name_inv = gradnonlin_field_name.substr(0,gradnonlin_field_name.rfind(".nii"))+ "_inv.nii";
    if(fs::exists(gradnonlin_name_inv))
    {
        this->gradwarp_field= readImageD<DisplacementFieldType>(gradnonlin_name_inv);
    }


    for(int d=0;d<2;d++)
    {
        std::string up_name = data_names[d];
        std::string bmtxt_name = up_name.substr(0,up_name.rfind(".nii"))+".bmtxt";
        vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);
        Nvols[d] = Bmatrix.rows();


        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));        
        std::string moteddy_trans_name= this->temp_folder + "/" + basename + "_moteddy_transformations.txt";

        this->dwi_transforms[d].resize(Nvols[d]);
        if(fs::exists(moteddy_trans_name))
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
                for(int p=0;p<NQUADPARAMS;p++)
                {
                    int npos = line.find(", ");
                    std::string curr_p_string = line.substr(0,npos);

                    double val = atof(curr_p_string.c_str());
                    params[p]=val;
                    line=line.substr(npos+2);
                }
                quad_trans->SetParameters(params);
                OkanQuadraticTransformType::ParametersType flags;
                flags.SetSize(NQUADPARAMS);
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
        if(fs::exists(s2v_trans_name))
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
                    for(int p=0;p<NQUADPARAMS;p++)
                    {
                        int npos = line.find(", ");
                        std::string curr_p_string = line.substr(0,npos);

                        double val = atof(curr_p_string.c_str());
                        params[p]=val;
                        line=line.substr(npos+2);
                    }
                    quad_trans->SetParameters(params);
                    OkanQuadraticTransformType::ParametersType flags;
                    flags.SetSize(NQUADPARAMS);
                    flags.Fill(0);
                    flags[0]=flags[1]=flags[2]=flags[3]=flags[4]=flags[5]=1;
                    quad_trans->SetParametersForOptimizationFlags(flags);
                    s2v_transformations[d][vol][k]    = quad_trans;

                }
            }
            s2v_text_file.close();
        }

        std::string native_inc_name= this->temp_folder + "/" + basename + "_native_inclusion.nii";
        if(fs::exists(native_inc_name))
        {
            this->native_inclusion_img[d].resize(Nvols[d]);
            for(int v=0;v<Nvols[d];v++)
                this->native_inclusion_img[d][v]= read_3D_volume_from_4DBool(native_inc_name,v);
        }


        std::string drift_name= this->temp_folder + "/" + basename + "_moteddy_drift.txt";
        if(fs::exists(drift_name))
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
    /*********************************************************************************
     We are doing this for several reasons.
     1:  We have to operate on Read/Phase/Slice coordinate system, not x/y/z like ITK.
         So we set the image direction to identity.
         Yes this causes an inconsistency between images and transformations so
         we have to be really careful everytime we register/transform an image.
     2:  Eddy currents do not affect the scanner isocenter.
         If the image header is correct and the image coordinate (0,0,0) is indeed the scanner isocenter
         we should use that one.
         But if the header is wrong, we can use the closest thing which is the center voxel of the image.
    **************************************************************************************/

    std::string center_string= RegistrationSettings::get().getValue<std::string>(std::string("rot_eddy_center"));

    // do not want to touch the original image so we duplicate it

    typename ImageType::Pointer nimg= ImageType::New();
    nimg->SetRegions(img->GetLargestPossibleRegion());
    nimg->SetSpacing(img->GetSpacing());

    typename ImageType::DirectionType orig_dir = img->GetDirection();
    typename ImageType::PointType orig_org = img->GetOrigin();

    typename ImageType::DirectionType id_dir;
    id_dir.SetIdentity();
    typename ImageType::PointType id_org;
    id_org.Fill(0);

    //Make the rotation and eddy center the image center voxel.
    id_org[0]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[0])-1)/2. * img->GetSpacing()[0];
    id_org[1]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[1])-1)/2. * img->GetSpacing()[1];
    id_org[2]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[2])-1)/2. * img->GetSpacing()[2];

    nimg->SetOrigin(id_org);
    nimg->SetDirection(id_dir);


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

            auto vec= pt_trans- pt;
            auto vec2= ref_img->GetDirection() * vec;
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
    if(this->epi_trans[PE])
    {

        OkanQuadraticTransformType::Pointer curr_mot_eddy_trans=this->dwi_transforms[PE][vol];
        auto rotmat1= curr_mot_eddy_trans->GetMatrix().GetVnlMatrix();

        vnl_matrix<double> dirmat=ref_img->GetDirection().GetVnlMatrix();
        auto rotmat2= dirmat.transpose() * rotmat1 *dirmat;
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

std::vector< std::vector<ImageType3D::Pointer> > FINALDATA::Generate()
{
    ImageType3D::Pointer ref_img = GenerateStructurals();
    ReadOrigTransforms();

    std::vector< std::vector<ImageType3D::Pointer> > final_imgs_to_return;
    final_imgs_to_return.resize(2);

    for(int PE=0;PE<2;PE++)
    {
        int nvols= Nvols[PE];

        std::string up_name = data_names[PE];
        if(up_name=="")
            continue;

        std::string bmtxt_name = up_name.substr(0,up_name.rfind(".nii"))+".bmtxt";
        vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);

        std::vector<ImageType3D::Pointer> raw_data,final_data;
        std::vector<ImageType3DBool::Pointer> final_inclusion_imgs;
        raw_data.resize(nvols);
        final_data.resize(nvols);
        if(this->native_inclusion_img[PE].size())
            final_inclusion_imgs.resize(nvols);
        for(int vol=0;vol<nvols;vol++)
            raw_data[vol]= read_3D_volume_from_4D(data_names[PE],vol);


        (*stream)<<"Transforming Volume done: "<<std::flush;



        #pragma omp parallel for
        for(int vol=0;vol<nvols;vol++)
        {
            TORTOISE::EnableOMPThread();

            ImageType3D::Pointer final_img=ImageType3D::New();
            final_img->SetRegions(ref_img->GetLargestPossibleRegion());
            final_img->Allocate();
            final_img->SetSpacing(ref_img->GetSpacing());
            final_img->SetDirection(ref_img->GetDirection());
            final_img->SetOrigin(ref_img->GetOrigin());
            final_img->FillBuffer(0.);

            //Transformation + interpolation
            {
                using MeasurementVectorType = itk::Vector<float, 3>;
                using SampleType = itk::Statistics::ListSample<MeasurementVectorType>;
                using TreeGeneratorType = itk::Statistics::KdTreeGenerator<SampleType>;
                using TreeType = TreeGeneratorType::KdTreeType;

                std::vector<float> values;
                SampleType::Pointer sample = SampleType::New();
                sample->SetMeasurementVectorSize(3);
                TreeType::Pointer tree =nullptr;

                if(this->s2v_transformations[PE].size()!=0)
                {
                    ImageType3D::SizeType sz= raw_data[0]->GetLargestPossibleRegion().GetSize();

                    itk::ImageRegionIteratorWithIndex<ImageType3D> it(raw_data[vol],raw_data[vol]->GetLargestPossibleRegion());
                    for(it.GoToBegin();!it.IsAtEnd();++it)
                    {
                        ImageType3D::IndexType ind3=it.GetIndex();
                        if(  (this->native_inclusion_img[PE].size() && this->native_inclusion_img[PE][vol]->GetPixel(ind3)) || this->native_inclusion_img[PE].size()==0 )
                        {
                            ImageType3D::PointType pt,pt_trans;

                            raw_data[vol]->TransformIndexToPhysicalPoint(ind3,pt);
                            pt_trans=this->s2v_transformations[PE][vol][ind3[2]]->TransformPoint(pt);

                            itk::ContinuousIndex<double,3> ind3_t;
                            raw_data[vol]->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_t);
                            MeasurementVectorType tt;
                            tt[0]=ind3_t[0];
                            tt[1]=ind3_t[1];
                            tt[2]=ind3_t[2];

                            sample->PushBack(tt);
                            values.push_back(raw_data[vol]->GetPixel(ind3));
                        }
                    }
                    TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
                    treeGenerator->SetSample(sample);
                    treeGenerator->SetBucketSize(16);
                    treeGenerator->Update();
                    tree = treeGenerator->GetOutput();
                } //if s2v

                ImageType3D::SizeType orig_sz= raw_data[vol]->GetLargestPossibleRegion().GetSize();
                ImageType3D::SizeType final_sz= final_img->GetLargestPossibleRegion().GetSize();

                using BSInterpolatorType = itk::BSplineInterpolateImageFunction<ImageType3D, double>;
                BSInterpolatorType::Pointer BSinterpolator = BSInterpolatorType::New();
                BSinterpolator->SetSplineOrder(3);
                BSinterpolator->SetInputImage(raw_data[vol]);

                using NNInterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType3DBool, double>;
                NNInterpolatorType::Pointer NNinterpolator = NNInterpolatorType::New();
                if(this->native_inclusion_img[PE].size())
                {
                    NNinterpolator->SetInputImage(this->native_inclusion_img[PE][vol]);

                    final_inclusion_imgs[vol]=ImageType3DBool::New();
                    final_inclusion_imgs[vol]->SetRegions(ref_img->GetLargestPossibleRegion());
                    final_inclusion_imgs[vol]->Allocate();
                    final_inclusion_imgs[vol]->SetSpacing(ref_img->GetSpacing());
                    final_inclusion_imgs[vol]->SetDirection(ref_img->GetDirection());
                    final_inclusion_imgs[vol]->SetOrigin(ref_img->GetOrigin());
                    final_inclusion_imgs[vol]->FillBuffer(0);
                }

                // The following transform includes all the transformations EXCEPT s2v
                // We are handling s2v separately.
                CompositeTransformType::Pointer all_trans=GenerateCompositeTransformForVolume(raw_data[0],PE, vol);

                itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_img,final_img->GetLargestPossibleRegion());
                for(it.GoToBegin();!it.IsAtEnd();++it)
                {
                    ImageType3D::IndexType ind3=it.GetIndex();
                    ImageType3D::PointType pt,pt_trans;
                    final_img->TransformIndexToPhysicalPoint(ind3,pt);
                    pt_trans= all_trans->TransformPoint(pt);

                    if(this->native_inclusion_img[PE].size())
                    {
                        if(NNinterpolator->IsInsideBuffer(pt_trans))
                        {
                            ImageType3DBool::PixelType val = NNinterpolator->Evaluate(pt_trans);
                            final_inclusion_imgs[vol]->SetPixel(ind3,val);
                        }
                    }
                    if(this->s2v_transformations[PE].size()==0)
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
                        if(  (final_inclusion_imgs.size() && final_inclusion_imgs[vol]->GetPixel(ind3)) || final_inclusion_imgs.size()==0)
                        {
                            if(values.size()>0)
                            {
                                itk::ContinuousIndex<double,3> ind3_t;
                                raw_data[vol]->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_t);

                                if(ind3_t[0]<0 || ind3_t[0]>orig_sz[0]-1 || ind3_t[1]<0 || ind3_t[1]>orig_sz[1]-1 || ind3_t[2]<0 || ind3_t[2]>orig_sz[2]-1 )
                                {
                                    it.Set(0);
                                    continue;
                                }

                                //Everything up to this point was backward transformation
                                //Now Forward transformation for s2v
                                //First find nearest nighbors from kd-tree
                                MeasurementVectorType queryPoint;
                                queryPoint[0]=ind3_t[0];
                                queryPoint[1]=ind3_t[1];
                                queryPoint[2]=ind3_t[2];

                                //unsigned int                           numberOfNeighbors = 16;
                                unsigned int                           numberOfNeighbors = 32;
                                TreeType::InstanceIdentifierVectorType neighbors;
                                std::vector<double> dists;
                                tree->Search(queryPoint, numberOfNeighbors, neighbors,dists);

                                std::vector<double>::iterator mini = std::min_element(dists.begin(), dists.end());
                                double mn= *mini;
                                int mn_id=std::distance(dists.begin(), mini);



                                using RealType = float;
                                using ScalarType = itk::Vector<RealType, 1>;
                                using PointSetType = itk::PointSet<ScalarType, 3>;
                                using ScalarImageType = itk::Image<ScalarType, 3>;
                                using PointSetPointer = typename PointSetType::Pointer;

                                PointSetPointer fieldPoints = PointSetType::New();
                                fieldPoints->Initialize();
                                auto & pointSTLContainer = fieldPoints->GetPoints()->CastToSTLContainer();
                                pointSTLContainer.reserve(numberOfNeighbors);
                                auto & pointDataSTLContainer = fieldPoints->GetPointData()->CastToSTLContainer();
                                pointDataSTLContainer.reserve(numberOfNeighbors);



                                ImageType3D::PointType minp; minp.Fill(1E10);
                                ImageType3D::PointType maxp; maxp.Fill(-1E10);

                                for(int n=0;n<numberOfNeighbors;n++)
                                {
                                    int neighbor= neighbors[n];
                                    MeasurementVectorType  aa= tree->GetMeasurementVector(neighbor) ;
                                    itk::ContinuousIndex<double,3> nind3;
                                    nind3[0]=aa[0];
                                    nind3[1]=aa[1];
                                    nind3[2]=aa[2];

                                    ImageType3D::PointType point;
                                    raw_data[vol]->TransformContinuousIndexToPhysicalPoint(nind3,point);
                                    ScalarType scalar;
                                    scalar[0] = values[neighbor];
                                    pointDataSTLContainer.push_back(scalar);
                                    pointSTLContainer.push_back(point);
                                    for(int d=0;d<3;d++)
                                    {
                                        if(point[d]<minp[d])
                                            minp[d]=point[d];
                                        if(point[d]>maxp[d])
                                            maxp[d]=point[d];
                                    }
                                }

                                bool no_good=false;
                                for(int d=0;d<3;d++)
                                {
                                    if(pt_trans[d]<minp[d])
                                    {
                                        no_good=true;
                                        break;
                                    }
                                    if(pt_trans[d]>maxp[d])
                                    {
                                        no_good=true;
                                        break;
                                    }
                                }
                                if(no_good)
                                {
                                    it.Set(0);
                                    continue;
                                }



                                using BSplineFilterType = itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, ScalarImageType>;
                                using BiasFieldControlPointLatticeType = typename BSplineFilterType::PointDataImageType;

                                int BSplineOrder=3;
                                auto splineFilter = BSplineFilterType::New();
                                typename BSplineFilterType::ArrayType numberOfControlPoints;
                                typename BSplineFilterType::ArrayType numberOfFittingLevels;
                                numberOfFittingLevels.Fill(1);



                                ImageType3D::PointType parametricDomainOrigin=minp;
                                ImageType3D::SpacingType parametricDomainSpacing;
                                int Nx=5;
                                parametricDomainSpacing[0]= (maxp[0]-minp[0])/(Nx-1);
                                parametricDomainSpacing[1]= (maxp[1]-minp[1])/(Nx-1);
                                parametricDomainSpacing[2]= (maxp[2]-minp[2])/(Nx-1);


                                ImageType3D::SizeType parametricDomainSize;
                                parametricDomainSize[0] = Nx;
                                parametricDomainSize[1] = Nx;
                                parametricDomainSize[2] = Nx;
                                numberOfControlPoints[0]=parametricDomainSize[0];
                                numberOfControlPoints[1]=parametricDomainSize[1];
                                numberOfControlPoints[2]=parametricDomainSize[2];

                                splineFilter->SetGenerateOutputImage(true); // the only reason to turn this off is if one only wants to use the
                                                                             // control point lattice for further processing
                                splineFilter->SetInput(fieldPoints);
                                splineFilter->SetSplineOrder(BSplineOrder);
                                splineFilter->SetNumberOfControlPoints(numberOfControlPoints);
                                splineFilter->SetNumberOfLevels(numberOfFittingLevels);
                                splineFilter->SetSize(parametricDomainSize);
                                splineFilter->SetSpacing(parametricDomainSpacing);
                                splineFilter->SetOrigin(parametricDomainOrigin);
                                splineFilter->SetNumberOfWorkUnits(1);
                                splineFilter->Update();

                                using BsplineFunctionType = itk::BSplineControlPointImageFunction< BiasFieldControlPointLatticeType, double>;
                                BsplineFunctionType::Pointer bspliner=BsplineFunctionType::New();
                                bspliner->SetOrigin(parametricDomainOrigin);
                                bspliner->SetSize(parametricDomainSize);
                                bspliner->SetSpacing(parametricDomainSpacing);
                                bspliner->SetSplineOrder(BSplineOrder);
                                bspliner->SetInputImage(splineFilter->GetPhiLattice());
                                ScalarType val = bspliner->EvaluateAtParametricPoint(pt_trans);
                                it.Set(val[0]);





                                /*

                                const double DIST_POW=6;
                                //If REALLY close nearest neighbor just set value to that point.
                                if(mn<0.1)
                                {
                                    float val= values[neighbors[mn_id]];
                                    it.Set(val);
                                }
                                else
                                {
                                    // If neigbor exists within one voxel distance
                                    // do inverse powered distance weighted interpolation
                                    // power is 8 to make images sharper.
                                    if(mn<1.)
                                    {
                                        double sm_weight=0;
                                        double sm_val=0;
                                        for(int n=0;n<numberOfNeighbors;n++)
                                        {
                                            int neighbor= neighbors[n];
                                            float dist = dists[n];

                                            double dist2= 1./pow(dist,DIST_POW);
                                            if(dist2>1E-50)
                                            {
                                                sm_val+= values[neighbor] *dist2;
                                                sm_weight+= dist2;
                                            }
                                        }
                                        if(sm_weight==0)
                                            it.Set(0);
                                        else
                                        {
                                            it.Set(sm_val/sm_weight);
                                        }
                                    }
                                    else
                                    {
                                        // Not a single value within a voxel distance
                                        // Not ideal.
                                        //So first check if we have neighbors on ALL sides of the cube for RBF interpolation.
                                        int sm_x=0,gt_x=0, sm_y=0,gt_y=0, sm_z=0,gt_z=0;
                                        for (unsigned long neighbor : neighbors)
                                        {
                                            MeasurementVectorType  aa= tree->GetMeasurementVector(neighbor) ;
                                            if(aa[0]<=ind3_t[0])
                                                sm_x++;
                                            if(aa[0]>ind3_t[0])
                                                gt_x++;
                                            if(aa[1]<=ind3_t[1])
                                                sm_y++;
                                            if(aa[1]>ind3_t[1])
                                                gt_y++;
                                            if(aa[2]<=ind3_t[2])
                                                sm_z++;
                                            if(aa[2]>ind3_t[2])
                                                gt_z++;
                                        }
                                        if(sm_x<2 || sm_y<2 || sm_z<2 || gt_x<2 || gt_y<2 || gt_z<2)
                                        {
                                            // We do not cover all sides of the cube with neighbors.
                                            // So default back to powered inverse distance weighting.


                                            //
                                            if(ind3[2]>1 && ind3[2]<final_sz[2]-2)
                                                if(final_inclusion_imgs.size())
                                                    final_inclusion_imgs[vol]->SetPixel(ind3,0);
                                            else
                                            {
                                                double sm_weight=0;
                                                double sm_val=0;
                                                for(int n=0;n<numberOfNeighbors;n++)
                                                {
                                                    int neighbor= neighbors[n];
                                                    float dist = dists[n];

                                                    double dist2= 1./pow(dist,DIST_POW);
                                                    if(dist2>1E-50)
                                                    {
                                                        sm_val+= values[neighbor] *dist2;
                                                        sm_weight+= dist2;
                                                    }
                                                }
                                                if(sm_weight==0)
                                                    it.Set(0);
                                                else
                                                {
                                                    it.Set(sm_val/sm_weight);
                                                }
                                            }
                                        }
                                        else
                                        {
                                            //RBF interpolation

                                            MatrixXd RBF = MatrixXd::Identity(numberOfNeighbors,numberOfNeighbors);
                                            VectorXd f(numberOfNeighbors);
                                            VectorXd curr_p(numberOfNeighbors);

                                            double eps=0.3;

                                            for(int r=0;r<numberOfNeighbors;r++)
                                            {
                                                MeasurementVectorType  aa1= tree->GetMeasurementVector(neighbors[r]) ;
                                                f[r]= values[neighbors[r]];
                                                double r2= (aa1[0]-ind3_t[0])*(aa1[0]-ind3_t[0])+(aa1[1]-ind3_t[1])*(aa1[1]-ind3_t[1])+(aa1[2]-ind3_t[2])*(aa1[2]-ind3_t[2]);
                                                curr_p[r]= exp(-eps*eps*r2);

                                                for(int c=r+1;c<numberOfNeighbors;c++)
                                                {
                                                    MeasurementVectorType  aa2= tree->GetMeasurementVector(neighbors[c]) ;

                                                    double rm= (aa1[0]-aa2[0])*(aa1[0]-aa2[0])+(aa1[1]-aa2[1])*(aa1[1]-aa2[1])+(aa1[2]-aa2[2])*(aa1[2]-aa2[2]);

                                                    double val = exp(-eps*eps *rm);
                                                    RBF(r,c)=val;
                                                    RBF(c,r)=val;
                                                }
                                            }

                                            JacobiSVD<MatrixXd> svd(RBF);
                                            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

                                            //Condition number of RBF interpolator not good
                                            //revert back to inverse distance interpolation
                                            if(cond>1E2)
                                            {
                                                if(ind3[2]>1 && ind3[2]<final_sz[2]-2)
                                                    if(final_inclusion_imgs.size())
                                                        final_inclusion_imgs[vol]->SetPixel(ind3,0);
                                                else
                                                {
                                                    double sm_weight=0;
                                                    double sm_val=0;
                                                    for(int n=0;n<numberOfNeighbors;n++)
                                                    {
                                                        int neighbor= neighbors[n];
                                                        float dist = dists[n];

                                                        double dist2= 1./pow(dist,DIST_POW);
                                                        if(dist2>1E-50)
                                                        {
                                                            sm_val+= values[neighbor] *dist2;
                                                            sm_weight+= dist2;
                                                        }
                                                    }
                                                    if(sm_weight==0)
                                                        it.Set(0);
                                                    else
                                                    {
                                                        it.Set(sm_val/sm_weight);
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                VectorXd w = RBF.colPivHouseholderQr().solve(f);
                                                double valm = w.dot(curr_p);
                                                it.Set(valm);
                                            } //if cond
                                        } //if RBF

                                    } //if mn<1

                                } //if mn<0.1

                                */

                            } //if values.size
                         } //if final_inc_img->GetPixel
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
                write_3D_image_to_4D_file<float>(final_img,new_nii_name,vol,Nvols[PE]);
            }

            TORTOISE::DisableOMPThread();
        } //for vol        



        vnl_matrix<double> rot_Bmat;
        {
            //Compute rotated Bmatrix, overall version
            vnl_matrix_fixed<double,3,3> id_trans; id_trans.set_identity();
            rot_Bmat= RotateBMatrix(Bmatrix,this->dwi_transforms[PE],id_trans);
            if(this->b0_t0_str_trans)
                rot_Bmat= RotateBMatrix(rot_Bmat,this->b0_t0_str_trans->GetMatrix().GetVnlMatrix(),raw_data[0]->GetDirection().GetVnlMatrix());
            vnl_matrix<double> flip_mat= ref_img->GetDirection().GetVnlMatrix() * raw_data[0]->GetDirection().GetVnlMatrix().transpose();

            for(int v=0;v<rot_Bmat.rows();v++)
            {
                vnl_vector<double> rot_Bmat_vec= rot_Bmat.get_row(v);
                vnl_matrix_fixed<double,3,3> rot_Bmat_vec_mat;
                rot_Bmat_vec_mat(0,0)=rot_Bmat_vec[0];rot_Bmat_vec_mat(0,1)=rot_Bmat_vec[1]/2; rot_Bmat_vec_mat(0,2)=rot_Bmat_vec[2]/2;
                rot_Bmat_vec_mat(1,0)=rot_Bmat_vec[1]/2;rot_Bmat_vec_mat(1,1)=rot_Bmat_vec[3]; rot_Bmat_vec_mat(1,2)=rot_Bmat_vec[4]/2;
                rot_Bmat_vec_mat(2,0)=rot_Bmat_vec[2]/2;rot_Bmat_vec_mat(2,1)=rot_Bmat_vec[4]/2; rot_Bmat_vec_mat(2,2)=rot_Bmat_vec[5];

                rot_Bmat_vec_mat= flip_mat * rot_Bmat_vec_mat * flip_mat.transpose();
                rot_Bmat_vec[0]=rot_Bmat_vec_mat(0,0);
                rot_Bmat_vec[1]=rot_Bmat_vec_mat(0,1)*2;
                rot_Bmat_vec[2]=rot_Bmat_vec_mat(0,2)*2;
                rot_Bmat_vec[3]=rot_Bmat_vec_mat(1,1);
                rot_Bmat_vec[4]=rot_Bmat_vec_mat(1,2)*2;
                rot_Bmat_vec[5]=rot_Bmat_vec_mat(2,2);
                rot_Bmat.set_row(v,rot_Bmat_vec);
            }

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
            (*stream)<<"Replacing final outliers..."<<std::endl;

            ImageType3D::Pointer b0_img = final_data[0];
            ImageType3D::Pointer final_mask= create_mask(b0_img);
            ImageType3D::Pointer orig_mask= create_mask(raw_data[0]);

            #pragma omp parallel for
            for(int vol=0;vol<Nvols[PE];vol++)
            {
                TORTOISE::EnableOMPThread();

                //Generate a final mask that includes both the brain mask and outlier mask
                using FilterType = itk::MultiplyImageFilter<ImageType3DBool, ImageType3D, ImageType3D>;
                FilterType::Pointer filter = FilterType::New();
                filter->SetInput2(orig_mask);
                filter->SetInput1(this->native_inclusion_img[PE][vol]);
                filter->Update();
                ImageType3D::Pointer weight_img= filter->GetOutput();

                //Check if we have a valid voxel
                bool allzeros=true;
                itk::ImageRegionIterator<ImageType3D> it(weight_img,weight_img->GetLargestPossibleRegion());
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
                if(allzeros)
                    final_inclusion_imgs[vol]->FillBuffer(0);

                fs::path up_path(up_name);
                std::string basename= fs::path(up_path).filename().string();
                basename=basename.substr(0,basename.rfind(".nii"));
                std::string new_inc_name= this->temp_folder + "/" + basename + "_final_temp_inc.nii";
                #pragma omp critical
                {
                    write_3D_image_to_4D_file<char>(final_inclusion_imgs[vol], new_inc_name,vol,Nvols[PE]);
                }
            }



            // FIRST, resynthesize images in the final space using the final_data            
            float dti_bval_cutoff= RegistrationSettings::get().getValue<float>(std::string("dti_bval"));
            float mapmri_bval_cutoff= RegistrationSettings::get().getValue<float>(std::string("hardi_bval"));

            vnl_vector<double> bvals = Bmatrix.get_column(0) + Bmatrix.get_column(3)+ Bmatrix.get_column(5);
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

            {
                std::vector<ImageType4D::Pointer> dummyv;
                std::vector<int> dummy;
                DTIModel dti_estimator;
                dti_estimator.SetBmatrix(rot_Bmat);
                dti_estimator.SetDWIData(final_data);
                dti_estimator.SetInclusionImage(final_inclusion_imgs);
                dti_estimator.SetVoxelwiseBmatrix(dummyv);
                dti_estimator.SetMaskImage(nullptr);
                dti_estimator.SetVolIndicesForFitting(low_DT_indices);
                dti_estimator.SetFittingMode("WLLS");
                dti_estimator.PerformFitting();


                // MAPMRI FITTING
                const unsigned int FINAL_STAGE_MAPMRI_DEGREE=6;
                MAPMRIModel mapmri_estimator;
                if(MAPMRI_indices.size()>0)
                {
                    double max_bval= bvals.max_value();
                    float small_delta,big_delta;

                    if(this->jsons[PE]["SmallDelta"]==json::value_t::null || this->jsons[PE]["BigDelta"]==json::value_t::null)
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
                        this->jsons[PE]["BigDelta"]= big_delta;
                        this->jsons[PE]["SmallDelta"]= small_delta;
                    }
                    else
                    {
                        big_delta=this->jsons[PE]["BigDelta"];
                        small_delta=this->jsons[PE]["SmallDelta"];
                    }

                    mapmri_estimator.SetMAPMRIDegree(FINAL_STAGE_MAPMRI_DEGREE);
                    mapmri_estimator.SetDTImg(dti_estimator.GetOutput());
                    mapmri_estimator.SetA0Image(dti_estimator.GetA0Image());
                    mapmri_estimator.SetBmatrix(rot_Bmat);
                    mapmri_estimator.SetDWIData(final_data);
                    mapmri_estimator.SetInclusionImage(final_inclusion_imgs);
                    mapmri_estimator.SetVoxelwiseBmatrix(dummyv);
                    mapmri_estimator.SetMaskImage(final_mask);
                    mapmri_estimator.SetVolIndicesForFitting(dummy);
                    mapmri_estimator.SetSmallDelta(small_delta);
                    mapmri_estimator.SetBigDelta(big_delta);
                    mapmri_estimator.PerformFitting();
                }



                #pragma omp parallel for
                for(int vol=0;vol<Nvols[PE];vol++)
                {
                    TORTOISE::EnableOMPThread();

                    if(MAPMRI_indices.size()>0)
                        synth_imgs[vol] = mapmri_estimator.SynthesizeDWI( rot_Bmat.get_row(vol));
                    else
                        synth_imgs[vol]= dti_estimator.SynthesizeDWI( rot_Bmat.get_row(vol) );

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
                using FilterType = itk::MultiplyImageFilter<ImageType3DBool, ImageType3D, ImageType3D>;
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



                        //Fit a quadratic spatial scaling function in both masks to minimize ||synth-real||^2
                        bool entered=false;

                        /*
                        MatrixXd AA = MatrixXd::Zero(6,6);
                        VectorXd f = VectorXd::Zero(6);

                        for(int j=0;j<sz[1];j++)
                        {
                            ind3[1]=j;
                            for(int i=0;i<sz[0];i++)
                            {
                                ind3[0]=i;

                                if(final_mask->GetPixel(ind3) && (final_inclusion_imgs[vol]->GetPixel(ind3)!=0))
                                {
                                    entered=true;

                                    long coords[]= {ind3[0]*ind3[0], ind3[1]*ind3[1], ind3[0]*ind3[1], ind3[0],ind3[1],1};
                                    for(int r=0;r<6;r++)
                                    {
                                        f[r]+= final_data2->GetPixel(ind3)*  synth_img->GetPixel(ind3) *coords[r];
                                        AA(r,0)+= ind3[0]*ind3[0]* coords[r] *synth_img->GetPixel(ind3) *synth_img->GetPixel(ind3);
                                        AA(r,1)+= ind3[1]*ind3[1]* coords[r] *synth_img->GetPixel(ind3) *synth_img->GetPixel(ind3);
                                        AA(r,2)+= ind3[0]*ind3[1]* coords[r] *synth_img->GetPixel(ind3) *synth_img->GetPixel(ind3);
                                        AA(r,3)+= ind3[0]*         coords[r] *synth_img->GetPixel(ind3) *synth_img->GetPixel(ind3);
                                        AA(r,4)+= ind3[1]*         coords[r] *synth_img->GetPixel(ind3) *synth_img->GetPixel(ind3);
                                        AA(r,5)+=                  coords[r] *synth_img->GetPixel(ind3) *synth_img->GetPixel(ind3);
                                    }
                                }
                            }
                        }
                        VectorXd w =VectorXd::Zero(6);
                        if(entered)
                            w=AA.colPivHouseholderQr().solve(f);
                        else
                            w[5]=1;
                        */
                        VectorXd w =VectorXd::Zero(6);
                        w[5]=1;

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
                                        //foreground region

                                         val=synth_imgs[vol]->GetPixel(ind3);
                                         double fact=w[0]*ind3[0]*ind3[0]+ w[1]*ind3[1]*ind3[1]+ w[2]*ind3[0]*ind3[1]+ w[3]*ind3[0]+ w[4]*ind3[1]+ w[5];
                                         val*= fact;
                                    }
                                    final_data[vol]->SetPixel(ind3,val);
                                }
                                else
                                {
                                    // On the intersection line (still in outlier region), average the two values
                                    float val=final_data2->GetPixel(ind3);
                                    if(sl_img_erode->GetPixel(ind2)==0 && final_mask->GetPixel(ind3) && synth_img->GetPixel(ind3)>0)
                                          val= 0.5* (val + synth_img->GetPixel(ind3));
                                    final_data[vol]->SetPixel(ind3,val);
                                }

                            } //for i
                        } //for j
                    } //for k


                } //if not allzeros










                TORTOISE::DisableOMPThread();
            } //for vol
            (*stream)<<std::endl;

        } //if repol


        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string new_nii_name=  this->temp_folder + "/" + basename + "_final_temp.nii";        
        for(int v=0;v<Nvols[PE];v++)
        {
            write_3D_image_to_4D_file<float>(final_data[v],new_nii_name,v,Nvols[PE]);
        }

        final_imgs_to_return[PE]=final_data;

    } //for PE

    return final_imgs_to_return;


}


#endif

