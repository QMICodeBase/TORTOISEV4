#ifndef _DRBUDDI_CXX
#define _DRBUDDI_CXX



#include "DRBUDDI.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "registration_settings.h"

#include "../tools/EstimateTensor/estimate_tensor_wlls.h"
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


DRBUDDI::DRBUDDI(std::string uname,std::string dname,std::vector<std::string> str_names,json mjson)
{
    this->up_nii_name=uname;
    this->down_nii_name=dname;
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
    if(this->proc_folder=="")
        this->proc_folder="./";

#ifdef DRBUDDIALONE
    this->stream= &(std::cout);
#else
    this->stream= TORTOISE::stream;
#endif

    (*stream)<<"Starting DRBUDDI Processing..."<<std::endl;
}


void DRBUDDI::Process()
{
    Step0_CreateImages();
    Step1_RigidRegistration();
    Step2_DiffeoRegistration();
    Step3_WriteOutput();
}

void DRBUDDI::Step0_CreateImages()
{

    CreateCorrectionImage(this->up_nii_name,this->b0_up,this->FA_up);
    CreateCorrectionImage(this->down_nii_name,this->b0_down,this->FA_down);

    std::string gradnonlin_field_name = parser->getGradNonlinInput();
    //std::string gradnonlin_field_name= RegistrationSettings::get().getValue<std::string>("grad_nonlin");
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
        {
            ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
            resampleFilter->SetOutputParametersFromImage(b0_down);
            resampleFilter->SetInput(b0_down);
            resampleFilter->SetTransform(gradwarp_trans);
            resampleFilter->Update();
            this->b0_down=resampleFilter->GetOutput();
        }
        if(this->FA_up)
        {
            ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
            resampleFilter->SetOutputParametersFromImage(b0_up);
            resampleFilter->SetInput(FA_up);
            resampleFilter->SetTransform(gradwarp_trans);
            resampleFilter->Update();
            this->FA_up=resampleFilter->GetOutput();
        }
        if(this->FA_down)
        {
            ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
            resampleFilter->SetOutputParametersFromImage(FA_down);
            resampleFilter->SetInput(FA_down);
            resampleFilter->SetTransform(gradwarp_trans);
            resampleFilter->Update();
            this->FA_down=resampleFilter->GetOutput();
        }
    }

    writeImageD<ImageType3D>(this->b0_up,proc_folder+"/blip_up_b0.nii");
    writeImageD<ImageType3D>(this->b0_down,proc_folder+"/blip_down_b0.nii");
    if(FA_up)
        writeImageD<ImageType3D>(this->FA_up,proc_folder+"/blip_up_FA.nii");
    if(FA_down)
        writeImageD<ImageType3D>(this->FA_down,proc_folder+"/blip_down_FA.nii");

}



std::vector<DRBUDDI::DisplacementFieldType::Pointer> DRBUDDI::DRBUDDI_Initial_Register_Up_Down(ImageType3D::Pointer b0_up_img,ImageType3D::Pointer b0_down_img, std::string phase)
{
    vnl_vector<double> phase_vector(3,0);
    if(phase=="vertical")
        phase_vector[1]=1;
    if(phase=="horizontal")
        phase_vector[0]=1;
    if(phase=="slice")
        phase_vector[2]=1;    


    std::vector<DRBUDDIStageSettings> stages;
    stages.resize(4);
    {
        stages[0].niter=100;
        stages[0].img_smoothing_std=3.;
        stages[0].downsample_factor=6;
        stages[0].learning_rate=0.2;
        stages[0].update_gaussian_sigma=4.5;
        stages[0].total_gaussian_sigma=0.05;
        stages[0].restrct=1;
        stages[0].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[0].metrics.push_back(metric);
    }
    {
        stages[1].niter=100;
        stages[1].img_smoothing_std=2.;
        stages[1].downsample_factor=4;
        stages[1].learning_rate=0.25;
        stages[1].update_gaussian_sigma=4.5;
        stages[1].total_gaussian_sigma=0.05;
        stages[1].restrct=1;
        stages[1].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[1].metrics.push_back(metric);
    }
    {
        stages[2].niter=10;
        stages[2].img_smoothing_std=1;
        stages[2].downsample_factor=2;
        stages[2].learning_rate=0.4;
        stages[2].update_gaussian_sigma=4.;
        stages[2].total_gaussian_sigma=0.05;
        stages[2].restrct=1;
        stages[2].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[2].metrics.push_back(metric);
    }
    {
        stages[3].niter=5;
        stages[3].img_smoothing_std=0.;
        stages[3].downsample_factor=1;
        stages[3].learning_rate=0.5;
        stages[3].update_gaussian_sigma=3.;
        stages[3].total_gaussian_sigma=0.05;
        stages[3].restrct=1;
        stages[3].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[3].metrics.push_back(metric);
    }

    DRBUDDI_Diffeo *myDRBUDDI_processor = new DRBUDDI_Diffeo;
    myDRBUDDI_processor->SetB0UpImage(b0_up_img);
    myDRBUDDI_processor->SetB0DownImage(b0_down_img);
    myDRBUDDI_processor->SetUpPEVector(phase_vector);
    myDRBUDDI_processor->SetDownPEVector(phase_vector);
    myDRBUDDI_processor->SetStagesFromExternal(stages);
    myDRBUDDI_processor->SetParser(parser);
    myDRBUDDI_processor->Process();

    DisplacementFieldType::Pointer ffield= myDRBUDDI_processor->getDefFINV();
    DisplacementFieldType::Pointer mfield= myDRBUDDI_processor->getDefMINV();

    std::vector<DisplacementFieldType::Pointer> fields;
    fields.push_back(ffield);
    fields.push_back(mfield);

    delete myDRBUDDI_processor;

    return fields;
}

DRBUDDI::RigidTransformType::Pointer DRBUDDI::RigidDiffeoRigidRegisterB0DownToB0Up(ImageType3D::Pointer b0_up_image, ImageType3D::Pointer b0_down_image, std::string mtype, ImageType3D::Pointer & initial_corrected_b0)
{
    initial_corrected_b0=nullptr;

    (*stream)<<"Starting initial rigid registration."<<std::endl;
    RigidTransformType::Pointer rigid1= nullptr,rigid2=nullptr;
    if(!parser->getStartWithDiffeo())
    {
        rigid1= RigidRegisterImagesEuler(b0_up_image,b0_down_image,mtype,parser->getRigidLR());
    }
    else
    {
        rigid1=RigidTransformType::New();
        rigid1->SetIdentity();
        rigid1->SetComputeZYX(true);
    }

    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(b0_up_image);
    resampleFilter2->SetInput(b0_down_image);
    resampleFilter2->SetTransform(rigid1);
    resampleFilter2->Update();
    ImageType3D::Pointer curr_down_quad_image= resampleFilter2->GetOutput();

    (*stream)<<"Starting initial diffeomorphic registration."<<std::endl;
    std::vector<DisplacementFieldType::Pointer> epi_fields= DRBUDDI_Initial_Register_Up_Down(b0_up_image, curr_down_quad_image, this->PE_string);

    ImageType3D::Pointer new_up_img=   JacobianTransformImage(b0_up_image,epi_fields[0],b0_up_image);
    ImageType3D::Pointer new_down_img= JacobianTransformImage(curr_down_quad_image,epi_fields[1],curr_down_quad_image);

    (*stream)<<"ReStarting initial rigid registration."<<std::endl;
    rigid2= RigidRegisterImagesEuler(new_up_img,new_down_img,mtype,parser->getRigidLR());


    ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
    resampleFilter3->SetOutputParametersFromImage(b0_up_image);
    resampleFilter3->SetInput(new_down_img);
    resampleFilter3->SetTransform(rigid2);
    resampleFilter3->Update();
    ImageType3D::Pointer down_quad_image= resampleFilter2->GetOutput();

    typedef itk::AddImageFilter<ImageType3D,ImageType3D,ImageType3D> AdderType;
    AdderType::Pointer adder= AdderType::New();
    adder->SetInput1(new_up_img);
    adder->SetInput2(down_quad_image);
    adder->Update();
    initial_corrected_b0= adder->GetOutput();


    RigidTransformType::Pointer total_rigid= RigidTransformType::New();
    total_rigid->SetIdentity();
    total_rigid->SetComputeZYX(true);
    total_rigid->Compose(rigid1,true);
    total_rigid->Compose(rigid2,true);

    return total_rigid;
}


void DRBUDDI::Step1_RigidRegistration()
{
    //Create and write b0_up quad image
    CreateBlipUpQuadImage();
    writeImageD<ImageType3D>(this->b0_up_quad,proc_folder+"/blip_up_b0_quad.nii");


    // Perform rigid + diffeo + rigid registration of b0_down to b0_up.
    // also generate the kind of corrected b0 image for further registration with the structural images
    RigidTransformType::Pointer down_to_up_rigid_trans=nullptr;
    ImageType3D::Pointer initial_corrected_b0=this->b0_up_quad;
    if(parser->getDisableInitRigid())
    {
        down_to_up_rigid_trans=RigidTransformType::New();
        down_to_up_rigid_trans->SetIdentity();
    }
    else
    {
        down_to_up_rigid_trans = RigidDiffeoRigidRegisterB0DownToB0Up(this->b0_up_quad,this->b0_down,"CC",initial_corrected_b0);
    }


    //Create and write b0_down quad image and the rigid transformation
    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(this->b0_up_quad);;
    resampleFilter2->SetInput(this->b0_down);
    resampleFilter2->SetTransform(down_to_up_rigid_trans);
    resampleFilter2->Update();
    this->b0_down_quad= resampleFilter2->GetOutput();
    itk::ImageRegionIterator<ImageType3D> it(this->b0_down_quad,this->b0_down_quad->GetLargestPossibleRegion());
    for(it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        if(it.Get()<0)
            it.Set(0);
    }
    writeImageD<ImageType3D>(this->b0_down_quad,proc_folder+"/blip_down_b0_quad.nii");

    std::string trans_name= proc_folder + std::string("/bdown_to_bup_rigidtrans.hdf5");
    typedef itk::TransformFileWriterTemplate< double > TransformWriterType;
    TransformWriterType::Pointer trwriter = TransformWriterType::New();
    trwriter->SetInput(down_to_up_rigid_trans);
    trwriter->SetFileName(trans_name);
    trwriter->Update();

    // if the FA images exist for both data, they will be used for actual correction
    // so transform both FA images to the quad space
    if(fs::exists(proc_folder+"/blip_up_FA.nii") && fs::exists(proc_folder+"/blip_down_FA.nii"))
    {
        {
            itk::IdentityTransform<double,3>::Pointer  id=itk::IdentityTransform<double,3>::New();
            id->SetIdentity();

            ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
            resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);;
            resampleFilter3->SetInput(this->FA_up);
            resampleFilter3->SetDefaultPixelValue(0);
            resampleFilter3->SetTransform(id);
            resampleFilter3->Update();
            this->FA_up_quad= resampleFilter3->GetOutput();
            itk::ImageRegionIterator<ImageType3D> it(this->FA_up_quad,this->FA_up_quad->GetLargestPossibleRegion());
            for(it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
                if(it.Get()<0)
                    it.Set(0);
            }
        }
        {
            ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
            resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);;
            resampleFilter3->SetInput(this->FA_down);
            resampleFilter3->SetDefaultPixelValue(0);
            resampleFilter3->SetTransform(down_to_up_rigid_trans);
            resampleFilter3->Update();
            this->FA_down_quad= resampleFilter3->GetOutput();

            itk::ImageRegionIterator<ImageType3D> it(this->FA_down_quad,this->FA_down_quad->GetLargestPossibleRegion());
            for(it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
                if(it.Get()<0)
                    it.Set(0);
            }


        }
    }

    // and finally rigid register all structural images to the kind of corrected b0 image

    int Nstr= parser->getNumberOfStructurals();

    for(int str=0;str<Nstr;str++)
    {
        (*stream)<<"Rigidly registering structural image id: " <<str<<" to b0_up quad..."<<std::endl;

        ImageType3D::Pointer str_img = readImageD<ImageType3D>(parser->getStructuralNames(str));
        RigidTransformType::Pointer rigid_trans= RigidRegisterImagesEuler( initial_corrected_b0,  str_img,parser->getRigidMetricType(),parser->getRigidLR());

        {
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



void DRBUDDI::Step2_DiffeoRegistration()
{

    this->b0_up=readImageD<ImageType3D>(proc_folder+"/blip_up_b0.nii");
    this->b0_down=readImageD<ImageType3D>(proc_folder+"/blip_down_b0.nii");
    this->b0_up_quad=readImageD<ImageType3D>(proc_folder+"/blip_up_b0_quad.nii");
    this->b0_down_quad=readImageD<ImageType3D>(proc_folder+"/blip_down_b0_quad.nii");

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
    //phase vector is in index space, not xyz space. The metric will do that conversion
    //so I commented the next line
    //phase_vector= this->b0_up_quad->GetDirection().GetVnlMatrix() * phase_vector;

    DRBUDDI_Diffeo *myDRBUDDI_processor = new DRBUDDI_Diffeo;
    myDRBUDDI_processor->SetB0UpImage(this->b0_up_quad);
    myDRBUDDI_processor->SetB0DownImage(this->b0_down_quad);
    if(this->FA_up_quad && this->FA_down_quad)
    {
        myDRBUDDI_processor->SetFAUpImage(this->FA_up_quad);
        myDRBUDDI_processor->SetFADownImage(this->FA_down_quad);
    }
    myDRBUDDI_processor->SetStructuralImages(structural_imgs);
    myDRBUDDI_processor->SetUpPEVector(phase_vector);
    myDRBUDDI_processor->SetDownPEVector(phase_vector);
    myDRBUDDI_processor->SetParser(parser);
    myDRBUDDI_processor->Process();

    this->def_FINV=myDRBUDDI_processor->getDefFINV();
    this->def_MINV=myDRBUDDI_processor->getDefMINV();

    delete myDRBUDDI_processor;
}

void DRBUDDI::Step3_WriteOutput()
{
    (*stream)<<"Writing DRBUDDI output files..."<<std::endl;


    writeImageD<DisplacementFieldType>(this->def_FINV,proc_folder+"/deformation_FINV.nii.gz");
    writeImageD<DisplacementFieldType>(this->def_MINV,proc_folder+"/deformation_MINV.nii.gz");



    ImageType3D::Pointer b0_up_corrected=nullptr, b0_down_corrected=nullptr;

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
    {
        DisplacementFieldTransformType::Pointer trans = DisplacementFieldTransformType::New();
        trans->SetDisplacementField(this->def_MINV);
        ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
        resampleFilter3->SetOutputParametersFromImage(this->b0_down_quad);;
        resampleFilter3->SetInput(this->b0_down_quad);
        resampleFilter3->SetTransform(trans);
        resampleFilter3->Update();
        b0_down_corrected= resampleFilter3->GetOutput();
        writeImageD<ImageType3D>(b0_down_corrected,proc_folder+"/blip_down_b0_corrected.nii");
    }

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(b0_up_corrected, b0_up_corrected->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        double val1= it.Get();
        double val2= b0_down_corrected->GetPixel(ind3);

        double val=0;
        if(val1+val2>1E-6)
        {
            val= 2*val1*val2/(val1+val2);
        }
        it.Set(val);
        ++it;
    }
    writeImageD<ImageType3D>(b0_up_corrected,proc_folder+"/b0_corrected_final.nii");


    ImageType3D::Pointer b0_up_corrected_JAC= JacobianTransformImage(b0_up_quad,def_FINV, b0_up_quad);
    writeImageD<ImageType3D>(b0_up_corrected_JAC,proc_folder+"/blip_up_b0_corrected_JAC.nii");
    ImageType3D::Pointer b0_down_corrected_JAC= JacobianTransformImage(b0_down_quad,def_MINV, b0_up_quad);
    writeImageD<ImageType3D>(b0_down_corrected_JAC,proc_folder+"/blip_down_b0_corrected_JAC.nii");


}



#endif

