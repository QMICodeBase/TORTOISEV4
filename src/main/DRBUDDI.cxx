#ifndef _DRBUDDI_CXX
#define _DRBUDDI_CXX



#include "DRBUDDI.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "registration_settings.h"
#include "create_mask.h"

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
#include "../tools/ResampleDWIs/resample_dwis.h"


DRBUDDI::DRBUDDI(std::string uname,std::string dname,std::vector<std::string> str_names,json mjson)
{
    this->up_nii_name=uname;
    this->down_nii_name=dname;
    this->structural_names=str_names;
    my_json=mjson;



#ifdef DRBUDDIALONE
    this->stream= &(std::cout);
#else
    this->stream= TORTOISE::stream;
#endif

    if(this->my_json["PhaseEncodingDirection"]==json::value_t::null)
    {
        if(this->my_json["PhaseEncodingAxis"]!=json::value_t::null)
        {
            this->my_json["PhaseEncodingDirection"]=this->my_json["PhaseEncodingAxis"];
        }
        else
        {
            if(this->my_json["InPlanePhaseEncodingDirectionDICOM"]!=json::value_t::null)
            {
                if(this->my_json["InPlanePhaseEncodingDirectionDICOM"]=="COL")
                {
                    this->my_json["PhaseEncodingDirection"]="j";
                }
                else
                {
                    this->my_json["PhaseEncodingDirection"]="i";
                }
            }
            else
            {
                (*stream)<<"Phase encoding information not present in JSON file. Create a new json file for the dataset..."<<std::endl;
                (*stream)<<"Exiting"<<std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

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



    (*stream)<<"Starting DRBUDDI Processing..."<<std::endl;
}


void DRBUDDI::Process()
{
    if(parser->getDRBUDDIStep()==0)
        Step0_CreateImages();
    if(parser->getDRBUDDIStep()<=1)
    {
        if(parser->getDRBUDDIStep()==1)
        {
            this->b0_up=readImageD<ImageType3D>(proc_folder+"/blip_up_b0.nii");
            this->b0_down=readImageD<ImageType3D>(proc_folder+"/blip_down_b0.nii");

            if(fs::exists(proc_folder+"/blip_up_FA.nii"))
                this->FA_up=readImageD<ImageType3D>(proc_folder+"/blip_up_FA.nii");
            if(fs::exists(proc_folder+"/blip_down_FA.nii"))
                this->FA_down=readImageD<ImageType3D>(proc_folder+"/blip_down_FA.nii");
        }


        Step1_RigidRegistration();
    }
    Step2_DiffeoRegistration();
    Step3_WriteOutput();
}

void DRBUDDI::Step0_CreateImages()
{

    CreateCorrectionImage(this->up_nii_name,this->b0_up,this->FA_up);
    CreateCorrectionImage(this->down_nii_name,this->b0_down,this->FA_down);

    std::string gradnonlin_field_name = parser->getGradNonlinInput();    
    if(gradnonlin_field_name!="" && parser->getNOGradWarp()==false)
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



std::vector<DRBUDDI::DisplacementFieldType::Pointer> DRBUDDI::DRBUDDI_Initial_Register_Up_Down(ImageType3D::Pointer b0_up_img,ImageType3D::Pointer b0_down_img, std::string phase,bool small)
{
    vnl_vector<double> phase_vector(3,0);
    if(phase=="vertical")
        phase_vector[1]=1;
    if(phase=="horizontal")
        phase_vector[0]=1;
    if(phase=="slice")
        phase_vector[2]=1;


    std::vector<DRBUDDIStageSettings> stages;

    if(small)
    {
        stages.resize(2);
        {
            stages[0].niter=30;
            stages[0].img_smoothing_std=2.;
            stages[0].downsample_factor=4;
            stages[0].learning_rate=0.2;
            stages[0].update_gaussian_sigma=6.;
            stages[0].total_gaussian_sigma=0.0;
            stages[0].restrct=1;
            stages[0].constrain=1;
            DRBUDDIMetric metric;
            metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric.weight=1;
            stages[0].metrics.push_back(metric);
        }
        {
            stages[1].niter=10;
            stages[1].img_smoothing_std=1.;
            stages[1].downsample_factor=2;
            stages[1].learning_rate=0.2;
            stages[1].update_gaussian_sigma=6.;
            stages[1].total_gaussian_sigma=0.0;
            stages[1].restrct=1;
            stages[1].constrain=1;
            DRBUDDIMetric metric;
            metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric.weight=1;
            stages[1].metrics.push_back(metric);
        }
    }
    else
    {
        stages.resize(4);
        {
            stages[0].niter=300;
            stages[0].img_smoothing_std=3.;
            stages[0].downsample_factor=6;
            stages[0].learning_rate=0.2;
            stages[0].update_gaussian_sigma=8.;
            stages[0].total_gaussian_sigma=0.0;
            stages[0].restrct=1;
            stages[0].constrain=1;
            DRBUDDIMetric metric;
            metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric.weight=1;
            stages[0].metrics.push_back(metric);
        }
        {
            stages[1].niter=300;
            stages[1].img_smoothing_std=2.;
            stages[1].downsample_factor=4;
            stages[1].learning_rate=0.25;
            stages[1].update_gaussian_sigma=7.;
            stages[1].total_gaussian_sigma=0.0;
            stages[1].restrct=1;
            stages[1].constrain=1;
            DRBUDDIMetric metric;
            metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric.weight=1;
            stages[1].metrics.push_back(metric);
        }
        {
            stages[2].niter=300;
            stages[2].img_smoothing_std=1;
            stages[2].downsample_factor=2;
            stages[2].learning_rate=0.25;
            stages[2].update_gaussian_sigma=6.;
            stages[2].total_gaussian_sigma=0.0;
            stages[2].restrct=1;
            stages[2].constrain=1;
            DRBUDDIMetric metric;
            metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric.weight=1;
            stages[2].metrics.push_back(metric);
        }
        {
            stages[3].niter=10;
            stages[3].img_smoothing_std=0.5;
            stages[3].downsample_factor=1;
            stages[3].learning_rate=0.3;
            stages[3].update_gaussian_sigma=5.;
            stages[3].total_gaussian_sigma=0.0;
            stages[3].restrct=1;
            stages[3].constrain=1;
            DRBUDDIMetric metric;
            metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
            metric.weight=1;
            stages[3].metrics.push_back(metric);
        }
    }



    /*
    std::vector<DRBUDDIStageSettings> stages;
    stages.resize(4);
    {
        stages[0].niter=300;
        stages[0].img_smoothing_std=3.;
        stages[0].downsample_factor=6;
        stages[0].learning_rate=0.2;
        stages[0].update_gaussian_sigma=6.5;
        stages[0].total_gaussian_sigma=0.0;
        stages[0].restrct=1;
        stages[0].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[0].metrics.push_back(metric);
    }
    {
        stages[1].niter=300;
        stages[1].img_smoothing_std=2.;
        stages[1].downsample_factor=4;
        stages[1].learning_rate=0.25;
        stages[1].update_gaussian_sigma=5.5;
        stages[1].total_gaussian_sigma=0.0;
        stages[1].restrct=1;
        stages[1].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[1].metrics.push_back(metric);
    }
    {
        stages[2].niter=300;
        stages[2].img_smoothing_std=1;
        stages[2].downsample_factor=2;
        stages[2].learning_rate=0.4;
        stages[2].update_gaussian_sigma=4.5;
        stages[2].total_gaussian_sigma=0.0;
        stages[2].restrct=1;
        stages[2].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[2].metrics.push_back(metric);
    }
    {
        stages[3].niter=10;
        stages[3].img_smoothing_std=0.5;
        stages[3].downsample_factor=1;
        stages[3].learning_rate=0.5;
        stages[3].update_gaussian_sigma=3.;
        stages[3].total_gaussian_sigma=0.0;
        stages[3].restrct=1;
        stages[3].constrain=1;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::MSJac);
        metric.weight=1;
        stages[3].metrics.push_back(metric);
    }
    */


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



DisplacementFieldType::Pointer DRBUDDI::CompositeToDispField(CompositeTransformType::Pointer comp_trans, ImageType3D::Pointer ref_img)
{
    DisplacementFieldType::Pointer disp_field= DisplacementFieldType::New();
    disp_field->SetRegions(ref_img->GetLargestPossibleRegion());
    disp_field->Allocate();
    disp_field->SetOrigin(ref_img->GetOrigin());
    disp_field->SetSpacing(ref_img->GetSpacing());
    disp_field->SetDirection(ref_img->GetDirection());
    DisplacementFieldType::PixelType zer; zer.Fill(0);
    disp_field->FillBuffer(zer);

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(disp_field,disp_field->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        ImageType3D::PointType pt,pt_trans;
        disp_field->TransformIndexToPhysicalPoint(ind3,pt);
        pt_trans = comp_trans->TransformPoint(pt);

        DisplacementFieldType::PixelType vec;
        vec[0]=pt_trans[0]-pt[0];
        vec[1]=pt_trans[1]-pt[1];
        vec[2]=pt_trans[2]-pt[2];

        it.Set(vec);
    }
    return disp_field;
}

DRBUDDI::RigidTransformType::Pointer DRBUDDI::RigidDiffeoRigidRegisterB0DownToB0Up(ImageType3D::Pointer b0_up_image, ImageType3D::Pointer b0_down_image, std::string mtype, ImageType3D::Pointer & initial_corrected_b0)
{
    (*stream)<<"Starting initial rigid registration."<<std::endl;

    double diff=1E10;
    int iter=0;

    itk::IdentityTransform<double,3>::Pointer  id=itk::IdentityTransform<double,3>::New();
    id->SetIdentity();

    CompositeTransformType::Pointer up_trans = CompositeTransformType::New();
    up_trans->AddTransform(id);
    CompositeTransformType::Pointer down_trans = CompositeTransformType::New();
    down_trans->AddTransform(id);

    RigidTransformType::Pointer total_trans=RigidTransformType::New();
    total_trans->SetIdentity();

    while(diff>5E-5 && iter<4)
    {
        DisplacementFieldType::Pointer up_field=CompositeToDispField(up_trans,b0_up_image);
        DisplacementFieldType::Pointer down_field=CompositeToDispField(down_trans,b0_up_image);

        ImageType3D::Pointer new_up_img=   JacobianTransformImage(b0_up_image,up_field,b0_up_image);
        ImageType3D::Pointer new_down_img=   JacobianTransformImage(b0_down_image,down_field,b0_up_image);

        RigidTransformType::Pointer rigid1= RigidRegisterImagesEulerSmall(new_up_img,new_down_img,mtype);
        down_trans->AddTransform(rigid1);
        total_trans->Compose(rigid1,true);

        using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
        ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
        resampleFilter2->SetOutputParametersFromImage(b0_up_image);
        resampleFilter2->SetInput(b0_down_image);
        resampleFilter2->SetTransform(down_trans);
        resampleFilter2->Update();
        ImageType3D::Pointer curr_down_quad_image= resampleFilter2->GetOutput();

        std::vector<DisplacementFieldType::Pointer> epi_fields= DRBUDDI_Initial_Register_Up_Down(new_up_img, curr_down_quad_image, this->PE_string,true);

        DisplacementFieldTransformType::Pointer up_diffeo=DisplacementFieldTransformType::New();
        up_diffeo->SetDisplacementField(epi_fields[0]);
        DisplacementFieldTransformType::Pointer down_diffeo=DisplacementFieldTransformType::New();
        down_diffeo->SetDisplacementField(epi_fields[1]);

        up_trans->AddTransform(up_diffeo);
        down_trans->AddTransform(down_diffeo);

        iter++;


        auto p1= rigid1->GetParameters();

        diff= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] +
                p1[3]*p1[3]/400. + p1[4]*p1[4]/400. + p1[5]*p1[5]/400. ;
    }

    std::cout<<"DIFF: " <<diff<<std::endl;

    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(b0_up_image);
    resampleFilter2->SetInput(b0_down_image);
    resampleFilter2->SetTransform(total_trans);
    resampleFilter2->Update();
    ImageType3D::Pointer curr_down_quad_image= resampleFilter2->GetOutput();

    std::vector<DisplacementFieldType::Pointer> epi_fields= DRBUDDI_Initial_Register_Up_Down(b0_up_image, curr_down_quad_image, this->PE_string,false);

    ImageType3D::Pointer new_up_img=   JacobianTransformImage(b0_up_image,epi_fields[0],b0_up_image);
    ImageType3D::Pointer new_down_img=   JacobianTransformImage(curr_down_quad_image,epi_fields[1],b0_up_image);

    typedef itk::AddImageFilter<ImageType3D,ImageType3D,ImageType3D> AdderType;
    AdderType::Pointer adder= AdderType::New();
    adder->SetInput1(new_up_img);
    adder->SetInput2(new_down_img);
    adder->Update();
    initial_corrected_b0= adder->GetOutput();



    return total_trans;




    /*
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
*/
}

#include "itkImageToHistogramFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkHistogramMatchingImageFilter.h"

ImageType3D::Pointer DRBUDDI::PreprocessImage(  ImageType3D::ConstPointer  inputImage,
                                                ImageType3D::PixelType lowerScaleValue,
                                                ImageType3D::PixelType upperScaleValue,
                                                float winsorizeLowerQuantile, float winsorizeUpperQuantile,
                                                ImageType3D::ConstPointer histogramMatchSourceImage )
{
    typedef itk::Statistics::ImageToHistogramFilter<ImageType3D>   HistogramFilterType;
    typedef  HistogramFilterType::InputBooleanObjectType InputBooleanObjectType;
    typedef  HistogramFilterType::HistogramSizeType      HistogramSizeType;
    typedef  HistogramFilterType::HistogramType          HistogramType;

    HistogramSizeType histogramSize( 1 );
    histogramSize[0] = 256;

    InputBooleanObjectType::Pointer autoMinMaxInputObject = InputBooleanObjectType::New();
    autoMinMaxInputObject->Set( true );

    HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();
    histogramFilter->SetInput( inputImage );
    histogramFilter->SetAutoMinimumMaximumInput( autoMinMaxInputObject );
    histogramFilter->SetHistogramSize( histogramSize );
    histogramFilter->SetMarginalScale( 10.0 );
    histogramFilter->Update();

    float lowerValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeLowerQuantile );
    float upperValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeUpperQuantile );

    typedef itk::IntensityWindowingImageFilter<ImageType3D, ImageType3D> IntensityWindowingImageFilterType;

    IntensityWindowingImageFilterType::Pointer windowingFilter = IntensityWindowingImageFilterType::New();
    windowingFilter->SetInput( inputImage );
    windowingFilter->SetWindowMinimum( lowerValue );
    windowingFilter->SetWindowMaximum( upperValue );
    windowingFilter->SetOutputMinimum( lowerScaleValue );
    windowingFilter->SetOutputMaximum( upperScaleValue );
    windowingFilter->Update();

    ImageType3D::Pointer outputImage = nullptr;
    if( histogramMatchSourceImage )
    {
        typedef itk::HistogramMatchingImageFilter<ImageType3D, ImageType3D> HistogramMatchingFilterType;
        HistogramMatchingFilterType::Pointer matchingFilter = HistogramMatchingFilterType::New();
        matchingFilter->SetSourceImage( windowingFilter->GetOutput() );
        matchingFilter->SetReferenceImage( histogramMatchSourceImage );
        matchingFilter->SetNumberOfHistogramLevels( 256 );
        matchingFilter->SetNumberOfMatchPoints( 12 );
        matchingFilter->ThresholdAtMeanIntensityOn();
        matchingFilter->Update();

        outputImage = matchingFilter->GetOutput();
        outputImage->Update();
        outputImage->DisconnectPipeline();
    }
    else
    {
        outputImage = windowingFilter->GetOutput();
        outputImage->Update();
        outputImage->DisconnectPipeline();
    }
    return outputImage;
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
        //down_to_up_rigid_trans = RigidDiffeoRigidRegisterB0DownToB0Up(this->b0_up_quad,this->b0_down,"CC",initial_corrected_b0);
        down_to_up_rigid_trans = RigidDiffeoRigidRegisterB0DownToB0Up(this->b0_up_quad,this->b0_down,"MI",initial_corrected_b0);
    }

    writeImageD<ImageType3D>(initial_corrected_b0,proc_folder+"/b0_str_registration_target.nii");

    using InterpolatorType= itk::BSplineInterpolateImageFunction<ImageType3D,double,double>;
    InterpolatorType::Pointer interp=InterpolatorType::New();
    interp->SetSplineOrder(3);


    //Create and write b0_down quad image and the rigid transformation
    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(this->b0_up_quad);;
    resampleFilter2->SetInput(this->b0_down);
    resampleFilter2->SetTransform(down_to_up_rigid_trans);
  //  resampleFilter2->SetInterpolator(interp);
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
            writeImageD<ImageType3D>(this->FA_up_quad,proc_folder+"/blip_up_FA_quad.nii");
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
            writeImageD<ImageType3D>(this->FA_down_quad,proc_folder+"/blip_down_FA_quad.nii");


        }
    }

    // and finally rigid register all structural images to the kind of corrected b0 image

    int Nstr= parser->getNumberOfStructurals();



    for(int str=0;str<Nstr;str++)
    {
        (*stream)<<"Rigidly registering structural image id: " <<str<<" to b0_up quad..."<<std::endl;

        ImageType3D::Pointer str_img_orig = readImageD<ImageType3D>(parser->getStructuralNames(str));
        ImageType3D::Pointer str_img= create_mask(str_img_orig);

        {
            itk::ImageRegionIteratorWithIndex<ImageType3D> it(str_img,str_img->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ImageType3D::IndexType ind3= it.GetIndex();
                it.Set(it.Get()* str_img_orig->GetPixel(ind3)*5 + str_img_orig->GetPixel(ind3) );
            }
        }

        str_img=PreprocessImage(str_img,0,1,0,1);
        initial_corrected_b0=PreprocessImage(initial_corrected_b0,0,1,0,1);

        RigidTransformType::Pointer rigid_trans1= RigidRegisterImagesEuler( initial_corrected_b0,  str_img, "CC",parser->getRigidLR());
        RigidTransformType::Pointer rigid_trans2= RigidRegisterImagesEuler( initial_corrected_b0,  str_img,"MI",parser->getRigidLR());

        auto params1= rigid_trans1->GetParameters();
        auto params2= rigid_trans2->GetParameters();
        auto p1=params1-params2;

        double diff=0;
        diff+= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] +
               p1[3]*p1[3]/400. + p1[4]*p1[4]/400. + p1[5]*p1[5]/400. ;

        RigidTransformType::Pointer rigid_trans=nullptr;
        (*stream)<<"R1: "<< params1<<std::endl;
        (*stream)<<"R2: "<< params2<<std::endl;
        (*stream)<<"MI vs CC diff: "<< diff<<std::endl;
        if(diff<0.005)
            rigid_trans=rigid_trans2;
        else
        {
            (*stream)<<"Could not compute the rigid transformation from the structural imageto b=0 image... Starting multistart.... This could take a while"<<std::endl;
            (*stream)<<"Better be safe than sorry, right?"<<std::endl;

            RigidTransformType::Pointer rigid_trans1a= RigidRegisterImagesEuler( str_img, initial_corrected_b0,  "CC",parser->getRigidLR(),false);
            RigidTransformType::ParametersType b1= rigid_trans1a->GetParameters();

            p1[0]= params1[0]+ b1[0];
            p1[1]= params1[1]+ b1[1];
            p1[2]= params1[2]+ b1[2];

            double diff1= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] ;
            RigidTransformType::Pointer rigid_trans2a= RigidRegisterImagesEuler( str_img, initial_corrected_b0,  "MI",parser->getRigidLR(),false);
            RigidTransformType::ParametersType b2= rigid_trans2a->GetParameters();

            std::cout<< "Trans CC F" << rigid_trans1->GetParameters()<<std::endl;
            std::cout<< "Trans CC B" << rigid_trans1a->GetParameters()<<std::endl;
            std::cout<< "Trans MI F" << rigid_trans2->GetParameters()<<std::endl;
            std::cout<< "Trans MI B" << rigid_trans2a->GetParameters()<<std::endl;


            p1[0]= params2[0]+ b2[0];
            p1[1]= params2[1]+ b2[1];
            p1[2]= params2[2]+ b2[2];

            double diff2= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] ;
            std::cout<< "diff1 "<<diff1 << " diff2 " <<diff2 <<std::endl;

            std::string new_metric_type="MI";
            if(diff1 < diff2)
            {
                (*stream)<< "CC was determined to be more robust than MI. Switching..."<<std::endl;
                new_metric_type="CC";
            }

            if(diff1<0.001)
            {
                b1[0]= (params1[0] - b1[0])/2.;
                b1[1]= (params1[1] - b1[1])/2.;
                b1[2]= (params1[2] - b1[2])/2.;
                b1[3]= (params1[3] );
                b1[4]= (params1[4] );
                b1[5]= (params1[5] );
                rigid_trans1->SetParameters(b1);

                rigid_trans= RigidRegisterImagesEuler( initial_corrected_b0,  str_img, "CC",parser->getRigidLR(),true, rigid_trans1);
            }
            else
            {
                if(diff2<0.001)
                {
                    b2[0]= (params2[0] - b2[0])/2.;
                    b2[1]= (params2[1] - b2[1])/2.;
                    b2[2]= (params2[2] - b2[2])/2.;
                    b2[3]= (params2[3] );
                    b2[4]= (params2[4] );
                    b2[5]= (params2[5] );
                    rigid_trans2->SetParameters(b2);

                    rigid_trans= RigidRegisterImagesEuler( initial_corrected_b0,  str_img, "MI",parser->getRigidLR(),true,rigid_trans2);
                }
                else
                {
                    std::vector<float> new_res; new_res.resize(3);
                    new_res[0]= initial_corrected_b0->GetSpacing()[0] * 2;
                    new_res[1]= initial_corrected_b0->GetSpacing()[1] * 2;
                    new_res[2]= initial_corrected_b0->GetSpacing()[2] * 2;
                    std::vector<float> dummy;
                    ImageType3D::Pointer b02= resample_3D_image(initial_corrected_b0,new_res,dummy,"Linear");
                    new_res[0]= str_img->GetSpacing()[0] * 2;
                    new_res[1]= str_img->GetSpacing()[1] * 2;
                    new_res[2]= str_img->GetSpacing()[2] * 2;
                    ImageType3D::Pointer str2= resample_3D_image(str_img,new_res,dummy,"Linear");

                    rigid_trans1=MultiStartRigidSearch(b02,  str2,new_metric_type);
                    rigid_trans= RigidRegisterImagesEuler( initial_corrected_b0,  str_img, new_metric_type,parser->getRigidLR(),rigid_trans1);
                }
            }

        }


        (*stream)<<"Rigid transformation: " << rigid_trans->GetParameters()<<std::endl;

        {
            ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
            resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);
            resampleFilter3->SetInput(str_img_orig);
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
    if(fs::exists(proc_folder+"/blip_up_FA_quad.nii"))
        this->FA_up_quad=readImageD<ImageType3D>(proc_folder+"/blip_up_FA_quad.nii");
    if(fs::exists(proc_folder+"/blip_down_FA_quad.nii"))
        this->FA_down_quad=readImageD<ImageType3D>(proc_folder+"/blip_down_FA_quad.nii");


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
        resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);
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

