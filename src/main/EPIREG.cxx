#ifndef _EPIREG_CXX
#define _EPIREG_CXX



#include "EPIREG.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "registration_settings.h"

#include "../tools/EstimateTensor/estimate_tensor_wlls.h"
#include "create_mask.h"
//#include "../tools/ComputeFAMap/compute_fa_map.h"
//#include "../tools/RotateBMatrix/rotate_bmatrix.h"

#include "rigid_register_images.h"

#include "itkNearestNeighborInterpolateImageFunction.h"
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


EPIREG::EPIREG(std::string uname,std::vector<std::string> str_names,json mjson)
{
    this->up_nii_name=uname;

    this->structural_names=str_names;
    my_json=mjson;

    std::string json_PE= my_json["PhaseEncodingDirection"];      //get phase encoding direction
    if(json_PE.find("j")!=std::string::npos)
        PE_string="vertical";
    else
        if(json_PE.find("i")!=std::string::npos)
            PE_string="horizontal";
        else
            PE_string="slice";

    this->proc_folder = fs::path(up_nii_name).parent_path().string();

    this->stream= TORTOISE::stream;
    (*stream)<<"Starting EPIREG Processing..."<<std::endl;
}


void EPIREG::Process()
{
    Step0_CreateImages();
    Step1_RigidRegistration();
    Step2_DiffeoRegistration();
    Step3_WriteOutput();
}

void EPIREG::Step0_CreateImages()
{
    ImageType3D::Pointer dummy;
    CreateCorrectionImage(this->up_nii_name,this->b0_up,dummy);

    std::string gradnonlin_field_name= RegistrationSettings::get().getValue<std::string>("grad_nonlin");
    if(gradnonlin_field_name!="")
    {
        std::string gradnonlin_name_inv = gradnonlin_field_name.substr(0,gradnonlin_field_name.rfind(".nii"))+ "_inv.nii";
        DisplacementFieldType::Pointer field= readImageD<DisplacementFieldType>(gradnonlin_name_inv);

        DisplacementFieldTransformType::Pointer gradwarp_trans=DisplacementFieldTransformType::New();
        gradwarp_trans->SetDisplacementField(field);

        using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
        {
            ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
            resampleFilter->SetOutputParametersFromImage(b0_up);
            resampleFilter->SetInput(b0_up);
            resampleFilter->SetTransform(gradwarp_trans);
            resampleFilter->Update();
            this->b0_up=resampleFilter->GetOutput();
        }

    }

    writeImageD<ImageType3D>(this->b0_up,proc_folder+"/blip_up_b0.nii");
}




void EPIREG::Step1_RigidRegistration()
{
    //Create and write b0_up quad image
    CreateBlipUpQuadImage();
    writeImageD<ImageType3D>(this->b0_up_quad,proc_folder+"/blip_up_b0_quad.nii");



    // and finally rigid register all structural images to the kind of corrected b0 image
    int Nstr= parser->getNumberOfStructurals();

    for(int str=0;str<Nstr;str++)
    {
        (*stream)<<"Rigidly registering structural image id: " <<str<<" to b0_up quad..."<<std::endl;

        ImageType3D::Pointer str_img = readImageD<ImageType3D>(parser->getStructuralNames(str));
        RigidTransformType::Pointer rigid_trans= RigidRegisterImagesEuler( this->b0_up_quad,  str_img,parser->getRigidMetricType(),parser->getRigidLR());

        {
            using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
            ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
            resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);
            resampleFilter3->SetInput(str_img);
            resampleFilter3->SetTransform(rigid_trans);
            resampleFilter3->SetDefaultPixelValue(0);
            resampleFilter3->Update();
            ImageType3D::Pointer structural_used= resampleFilter3->GetOutput();
            itk::ImageRegionIterator<ImageType3D> it(structural_used,structural_used->GetLargestPossibleRegion());
            for(it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
                if(it.Get()<0)
                    it.Set(0);
            }
            structural_imgs.push_back(structural_used);

            char dummy_name[100]={0};
            if(Nstr>1)
                sprintf(dummy_name,"/structural_used_%d.nii",str);
            else
                sprintf(dummy_name,"/structural_used.nii");
            std::string new_str_name= this->proc_folder + std::string(dummy_name);
            writeImageD<ImageType3D>(structural_used,new_str_name);
        }
    }
}


void EPIREG::Step2_DiffeoRegistration()
{    
    this->b0_up_quad=readImageD<ImageType3D>(proc_folder+"/blip_up_b0_quad.nii");


    int Nstr= parser->getNumberOfStructurals();
    for(int str=0;str<Nstr;str++)
    {
        {
            char dummy_name[100]={0};
            if(Nstr>1)
                sprintf(dummy_name,"/structural_used_%d.nii",str);
            else
                sprintf(dummy_name,"/structural_used.nii");
            std::string new_str_name= this->proc_folder + std::string(dummy_name);

            ImageType3D::Pointer str_img =readImageD<ImageType3D>(new_str_name);
            structural_imgs.push_back(str_img);
        }
    }

    vnl_vector<double> phase_vector(3,0);
    if(this->PE_string=="vertical")
        phase_vector[1]=1;
    if(this->PE_string=="horizontal")
        phase_vector[0]=1;
    if(this->PE_string=="slice")
        phase_vector[2]=1;
    phase_vector= this->b0_up->GetDirection().GetVnlMatrix() * phase_vector;


    std::vector<DRBUDDIStageSettings> stages;
    stages.resize(6);
    {
        stages[0].niter=100;
        stages[0].img_smoothing_std=3.;
        stages[0].downsample_factor=8;
        stages[0].learning_rate=0.05;
        stages[0].update_gaussian_sigma=5.;
        stages[0].total_gaussian_sigma=0.05;
        stages[0].restrct=1;
        stages[0].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CCJacS);
        metric.weight=1;
        stages[0].metrics.push_back(metric);
    }
    {
        stages[1].niter=100;
        stages[1].img_smoothing_std=2.;
        stages[1].downsample_factor=6;
        stages[1].learning_rate=0.1;
        stages[1].update_gaussian_sigma=5.;
        stages[1].total_gaussian_sigma=0.05;
        stages[1].restrct=1;
        stages[1].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CCJacS);
        metric.weight=1;
        stages[1].metrics.push_back(metric);
    }
    {
        stages[2].niter=100;
        stages[2].img_smoothing_std=2.;
        stages[2].downsample_factor=4;
        stages[2].learning_rate=0.2;
        stages[2].update_gaussian_sigma=5.;
        stages[2].total_gaussian_sigma=0.05;
        stages[2].restrct=1;
        stages[2].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CCJacS);
        metric.weight=1;
        stages[2].metrics.push_back(metric);
    }
    {
        stages[3].niter=100;
        stages[3].img_smoothing_std=1.;
        stages[3].downsample_factor=2;
        stages[3].learning_rate=0.2;
        stages[3].update_gaussian_sigma=5.;
        stages[3].total_gaussian_sigma=0.05;
        stages[3].restrct=1;
        stages[3].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CCJacS);
        metric.weight=1;
        stages[3].metrics.push_back(metric);
    }
    {
        stages[4].niter=100;
        stages[4].img_smoothing_std=0.;
        stages[4].downsample_factor=1;
        stages[4].learning_rate=0.2;
        stages[4].update_gaussian_sigma=5.;
        stages[4].total_gaussian_sigma=0.05;
        stages[4].restrct=1;
        stages[4].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CCJacS);
        metric.weight=1;
        stages[4].metrics.push_back(metric);
    }
    {
        stages[5].niter=20;
        stages[5].img_smoothing_std=0.;
        stages[5].downsample_factor=1;
        stages[5].learning_rate=0.2;
        stages[5].update_gaussian_sigma=5.;
        stages[5].total_gaussian_sigma=0.05;
        stages[5].restrct=0;
        stages[5].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CCJacS);
        metric.weight=1;
        stages[5].metrics.push_back(metric);
    }



    DRBUDDI_Diffeo *myEPIREG_processor = new DRBUDDI_Diffeo;
    myEPIREG_processor->SetB0UpImage(this->b0_up_quad);
    myEPIREG_processor->SetB0DownImage(this->b0_up_quad);
    myEPIREG_processor->SetStructuralImages(structural_imgs);
    myEPIREG_processor->SetUpPEVector(phase_vector);
    myEPIREG_processor->SetDownPEVector(phase_vector);
    myEPIREG_processor->SetParser(parser);
    myEPIREG_processor->SetStagesFromExternal(stages);
    myEPIREG_processor->Process();

    this->def_FINV=myEPIREG_processor->getDefFINV();    

    delete myEPIREG_processor;
}

void EPIREG::Step3_WriteOutput()
{
    (*stream)<<"Writing EPIREG output files..."<<std::endl;


    writeImageD<DisplacementFieldType>(this->def_FINV,proc_folder+"/deformation_FINV.nii.gz");

    ImageType3D::Pointer b0_up_corrected=nullptr;

    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    {
        DisplacementFieldTransformType::Pointer trans = DisplacementFieldTransformType::New();
        trans->SetDisplacementField(this->def_FINV);
        ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
        resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);;
        resampleFilter3->SetInput(this->b0_up_quad);
        resampleFilter3->SetTransform(trans);
        resampleFilter3->Update();
        b0_up_corrected= resampleFilter3->GetOutput();
        writeImageD<ImageType3D>(b0_up_corrected,proc_folder+"/blip_up_b0_corrected.nii");
    }

    ImageType3D::Pointer b0_up_corrected_JAC= JacobianTransformImage(b0_up_quad,def_FINV, b0_up_quad);
    writeImageD<ImageType3D>(b0_up_corrected_JAC,proc_folder+"/blip_up_b0_corrected_JAC.nii");

}



#endif

