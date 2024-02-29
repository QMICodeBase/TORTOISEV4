#ifndef _TORTOISE_CXX
#define _TORTOISE_CXX

#include <omp.h>

//#include "TORTOISE.h"

#include "registration_settings.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../tools/gradnonlin/mk_displacementMaps.h"
#include "../tools/DWIDenoise/dwi_denoise.h"
#include "../tools/UnRing/unring.h"
#include "DIFFPREP.h"
#include "DRBUDDI.h"
#include "EPIREG.h"
#include "rigid_register_images.h"
#include "../utilities/read_bmatrix_file.h"
#include "create_mask.h"
#include "FINALDATA.h"
#include "../utilities/math_utilities.h"



#include "itkTransformFileWriter.h"
#include "itkTransformFactory.h"
#include "itkInvertDisplacementFieldImageFilterOkan.h"




TORTOISE::TORTOISE(int argc, char *argv[])
{
    std::string TORTOISE_loc = executable_path(argv[0]);         //get the location of the called executable
    this->executable_folder= fs::path(TORTOISE_loc).parent_path().string(); //to access the settings folder


    //parse all the command line parameters
    this->parser = new TORTOISE_PARSER(argc,argv);

    //Set up the number of cpus to use
    std::string system_settings_file = this->executable_folder +std::string("/../settings/system_settings/default_system_settings.json");
    json system_settings_json;
    if(fs::exists(system_settings_file))
    {
        std::ifstream system_settings_stream(system_settings_file);
        system_settings_stream >> system_settings_json;
        system_settings_stream.close();
    }
    else
    {
        system_settings_json["PercentOfCpuCoresToUse"]=0.5;
    }

    //Set threading parameters
    {
        SetNMaxCores(getNCores());
        float perc=system_settings_json["PercentOfCpuCoresToUse"];

        /*
        if(perc<1)
        {
            std::string aa("ITK_GLOBAL_DEFAULT_THREADER=PLATFORM");
            putenv((char *)aa.c_str());
        }
        else
        {
            std::string aa("ITK_GLOBAL_DEFAULT_THREADER=POOL");
            putenv((char *)aa.c_str());
        }
        */

        int nc=(int)(GetNMaxCores()*perc)-1;
        if(nc==0)
            nc=1;
        SetNAvailableCores(nc);
        omp_set_num_threads(GetNAvailableCores());
        if(parser->getDisableITKThreads())
        {
            itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads( 1 );
        }


        std::vector<uint> thread_array; thread_array.resize(GetNAvailableCores());
        for(int t=0;t<thread_array.size();t++)
            thread_array[t]=0;
        TORTOISE::SetThreadArray(thread_array);
    }



    //check if the command line parameters are okay.
    bool aokay= CheckIfInputsOkay();
    if(!aokay)
    {
        std::cout<<"Exiting..."<<std::endl;
        exit(EXIT_FAILURE);
    }


    itk::TransformFactory<OkanQuadraticTransformType>::RegisterTransform();


    this->temp_proc_folder= this->parser->getTempProcFolder();  //get or create the folder where all the intermediate files will be located
    if(this->temp_proc_folder=="")
    {
        std::string up_name = this->parser->getUpInputName();
        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));

        fs::path up_folder_path = up_path.parent_path();
        this->temp_proc_folder= (up_folder_path / (basename + std::string("_temp_proc")) ).string();

        if (!fs::exists(this->temp_proc_folder))
          fs::create_directory(this->temp_proc_folder);
    }

    std::string log_dir= this->temp_proc_folder + std::string("/logs");  //Create the log_Stream that will output to both
    if (!fs::exists(log_dir))                                            // std::cout and the log file
      fs::create_directories(log_dir);
    std::string log_main=  log_dir + std::string("/log_main.txt");
    std::ofstream log_stream;
    log_stream.open(log_main);
    Tee tee( std::cout, log_stream );
    TORTOISE::stream = new TeeStream(tee);


    (*stream)<<"TORTOISE version: " <<GetTORTOISEVersion()<<std::endl;
    for(int i=0;i<argc;i++)                   // Print the entered command for logging
        (*stream)<<argv[i]<< " ";
    (*stream)<<std::endl<<std::endl;
    (*stream)<<"Using up to " << GetNAvailableCores()+1<< " CPU cores."<<std::endl;


    //Set up the name of the output data and create directories if needed
    if(parser->getOutputName()=="")
    {
        std::string up_name = this->parser->getUpInputName();

        boost::system::error_code ec;
        boost::filesystem::path p2(up_name );

        boost::filesystem::path up_path( boost::filesystem::canonical(
                p2, boost::filesystem::current_path(), ec));


        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));

        fs::path up_folder_path = up_path.parent_path();
        this->output_name= up_folder_path.string() + std::string("/") + basename + std::string("_TORTOISE_final.nii");
    }
    else
    {
        this->output_name=parser->getOutputName();
        if(this->output_name.find("/")!=std::string::npos)
        {
            fs::path output_name_path(this->output_name);
            if(!fs::exists(output_name_path.parent_path()))
            {
                fs::create_directories(output_name_path.parent_path());
            }
        }
    }


    this->proc_infos.resize(2);
    {
        this->proc_infos.resize(1);
        std::string up_name = this->parser->getUpInputName();
        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        this->proc_infos[0].nii_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc.nii");
        this->proc_infos[0].bmtxt_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc.bmtxt");
        this->proc_infos[0].json_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc.json");
    }

     if(parser->getDownInputName()=="")    //get the proc files names
     {
         std::string down_name = this->parser->getDownInputName();
         fs::path down_path(down_name);
         std::string basename= fs::path(down_path).filename().string();
         basename=basename.substr(0,basename.rfind(".nii"));
         this->proc_infos[1].nii_name= "";
         this->proc_infos[1].bmtxt_name= "";
         this->proc_infos[1].json_name= "";
     }
     else
     {
         std::string down_name = this->parser->getDownInputName();
         fs::path down_path(down_name);
         std::string basename= fs::path(down_path).filename().string();
         basename=basename.substr(0,basename.rfind(".nii"));
         this->proc_infos[1].nii_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc.nii");
         this->proc_infos[1].bmtxt_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc.bmtxt");
         this->proc_infos[1].json_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc.json");
     }


    LoadDefaultSettings();                  // Load the default settings from the files in the settings folder
    UpdateSettingsFromCommandLine();        // Update any setting that needs to be changed by command line
    WriteFinalSettings();                   // write the final settings in the proc folder for logging purposes


    this->entryPoint();

    FillReportJson();


}

void TORTOISE::FillReportJson()
{
    this->processing_report_json["TORTOISE_ver"]=GetTORTOISEVersion();
    if(fs::exists("/.dockerenv"))
    {
        this->processing_report_json["TORTOISE_package"] = std::string("Docker version");
    }
    else
    {
        this->processing_report_json["TORTOISE_package"] = std::string("Linux5_local");
    }

    std::string log_dir= this->temp_proc_folder + std::string("/logs");
    this->processing_report_json["log_file"]=  log_dir + std::string("/log_main.txt");


    std::string up_name = this->parser->getUpInputName();
    fs::path up_path(up_name);
    std::string basename= fs::path(up_path).filename().string();
    basename=basename.substr(0,basename.rfind(".nii"));
    this->processing_report_json["settings_file"]= this->temp_proc_folder + std::string("/logs/") + basename + std::string("_proc_settings.dmc");


    if(this->my_jsons[0]["EchoTime"]!=json::value_t::null )
    {
        float te= this->my_jsons[0]["EchoTime"];
        this->processing_report_json["te"]= te *1000.;
    }
    if(this->my_jsons[0]["RepetitionTime"]!=json::value_t::null )
    {
        float tr= this->my_jsons[0]["RepetitionTime"];
        this->processing_report_json["tr"]=tr*1000.;
    }


    std::vector<std::string> tags={"up","down"};
    for(int PE=0;PE<2;PE++)
    {
        if(this->proc_infos[PE].nii_name!="")
        {
            std::string nii_name=this->proc_infos[PE].nii_name;
            vnl_matrix<double> Bmatrix= read_bmatrix_file(nii_name.substr(0,nii_name.rfind(".nii"))+std::string(".bmtxt"));
            int Nvols = Bmatrix.rows();
            vnl_vector<double> bvals= Bmatrix.get_column(0)+Bmatrix.get_column(3)+Bmatrix.get_column(5);
            this->processing_report_json[tags[PE] + "_nvols"]=Nvols;

            std::vector<int> shells;
            for(int v=0;v<Nvols;v++)
            {
                float curr_b= bvals[v];
                bool found=false;
                for(int v2=0;v2<shells.size();v2++)
                {
                    if(fabs(shells[v2] -curr_b)<50)
                    {
                        found=true;
                        break;
                    }
                }
                if(!found)
                    shells.push_back(round50(curr_b));
            }
            this->processing_report_json[tags[PE] +"_nshells"]=shells.size();

            if(PE==0)
                this->processing_report_json["orig_data_" + tags[PE]] = parser->getUpInputName();
            else
                this->processing_report_json["orig_data_" + tags[PE]] = parser->getDownInputName();

            std::vector<std::string> str_names=parser->getStructuralNames();
            if(str_names.size()>0)
                this->processing_report_json["orig_structural"] =  str_names[0];


            this->processing_report_json["den_gibbs_data_" + tags[PE]] = nii_name;

            std::string nname= fs::path(nii_name).filename().string();
            fs::path proc_path2=fs::path(nii_name).parent_path();
            std::string basename = nname.substr(0, nname.find(".nii"));
            fs::path trans_text_path = proc_path2 / (basename + std::string("_moteddy_transformations.txt"));
            fs::path moteddy_nii_path = proc_path2 / (basename + std::string("_moteddy.nii"));
            fs::path s2v_text_path = proc_path2 / (basename + std::string("_s2v_transformations.txt"));
            fs::path slice_resids_path = proc_path2 / (basename + std::string("_slice_resids.txt"));
            fs::path slice_resids_Z_path = proc_path2 / (basename + std::string("_slice_resids_Z.txt"));
            fs::path native_inc_map_path = proc_path2 / (basename + std::string("_native_inclusion.nii"));
            fs::path drift_path = proc_path2 / (basename + std::string("_moteddy_drift.txt"));
            fs::path b0_path = proc_path2 / ( std::string("blip_")+ tags[PE] + "_b0_quad.nii");
            fs::path b0_corrected_path = proc_path2 / ( std::string("b0_corrected_final.nii") );
            fs::path gradfield_path = proc_path2 / ( std::string("field_inv.nii") );




            if(fs::exists(moteddy_nii_path))
                this->processing_report_json["mot_eddy_data_" + tags[PE]] = moteddy_nii_path.string();
            if(fs::exists(trans_text_path))
                this->processing_report_json["inter-volume_motion_dset_" + tags[PE]] = trans_text_path.string();
            if(fs::exists(s2v_text_path))
                this->processing_report_json["intra-volume_motion_dset_" + tags[PE]] = s2v_text_path.string();
            if(fs::exists(slice_resids_path))
                this->processing_report_json["slice_residuals_" + tags[PE]] = slice_resids_path.string();
            if(fs::exists(slice_resids_Z_path))
                this->processing_report_json["slice_residuals_Z_" + tags[PE]] = slice_resids_Z_path.string();
            if(fs::exists(native_inc_map_path))
                this->processing_report_json["native_inclusion_map_" + tags[PE]] = native_inc_map_path.string();
            if(fs::exists(drift_path))
                this->processing_report_json["signal_drift_" + tags[PE]] = drift_path.string();
            if(fs::exists(b0_path))
                this->processing_report_json["b0_epi_" + tags[PE]] = b0_path.string();
            if(fs::exists(b0_corrected_path))
                this->processing_report_json["b0_epi_corrected"] = b0_corrected_path.string();
            if(fs::exists(gradfield_path))
                this->processing_report_json["gradient_nonlinearity_field"] = gradfield_path.string();
            if(PE==0)
            {
                fs::path fname= proc_path2 / ( std::string("deformation_FINV.nii.gz") );
                if(fs::exists(fname))
                    this->processing_report_json["epi_field_" + tags[PE]] = fname.string();
            }
            else
            {
                fs::path fname= proc_path2 / ( std::string("deformation_MINV.nii.gz") );
                if(fs::exists(fname))
                    this->processing_report_json["epi_field_" + tags[PE]] = fname.string();
            }


            std::vector<std::string>  structural_names = RegistrationSettings::get().getVectorValue<std::string>("structural");
            std::string output_folder= fs::path(output_name).parent_path().string();
            if(output_folder=="")
                output_folder="./";
            if(structural_names.size())
            {
                std::string nm = output_folder + "/structural_0.nii";
                this->processing_report_json["final_structural"] = nm;

            }

            this->processing_report_json["final_data"] = this->output_name;

            fs::path new_inc_path = proc_path2 / (basename + std::string("_final_temp_inc.nii"));
            if(fs::exists(new_inc_path))
            {
                this->processing_report_json["final_inclusion_map_" + tags[PE]] = new_inc_path.string();
            }

        }
    }

    std::string output_folder= fs::path(output_name).parent_path().string();
    if(output_folder=="")
        output_folder="./";


    std::ofstream out_json(output_folder + "/processing_report.json");
    out_json << std::setw(4) << this->processing_report_json<< std::endl;
    out_json.close();



}



void TORTOISE::LoadDefaultSettings()
{
    std::string registration_settings_filename= this->executable_folder + std::string("/../settings/defaults.dmc");

    if(!fs::exists(registration_settings_filename))
    {
        (*stream)<<"The registration settings file "<<registration_settings_filename << " does not exist.. Exiting"<<std::endl;
        exit(EXIT_FAILURE);
    }
    RegistrationSettings::get().parseFile(registration_settings_filename);
}


void TORTOISE::UpdateSettingsFromCommandLine()
{

    RegistrationSettings::get().setValue<std::string>("up_data", parser->getUpInputName());
    if(parser->getDownInputName()!="")
        RegistrationSettings::get().setValue<std::string>("down_data", parser->getDownInputName());
    RegistrationSettings::get().setValue<std::string>("temp_folder", this->temp_proc_folder);
    RegistrationSettings::get().setVectorValue<std::string>("structural", parser->getStructuralNames());
    if(parser->getReorientationName()!="")
        RegistrationSettings::get().setValue<std::string>("reorientation", parser->getReorientationName());
    if(parser->getGradNonlinInput()!="")
    {
        RegistrationSettings::get().setValue<std::string>("grad_nonlin", parser->getGradNonlinInput());
        RegistrationSettings::get().setValue<bool>("grad_nonlin_isGE", parser->getGradNonlinIsGE());
        RegistrationSettings::get().setValue<std::string>("grad_nonlin_warpD", parser->getGradNonlinGradWarpDim());
    }
    RegistrationSettings::get().setValue<int>("flipX", parser->getFlipX());
    RegistrationSettings::get().setValue<int>("flipY", parser->getFlipY());
    RegistrationSettings::get().setValue<int>("flipZ", parser->getFlipZ());

    RegistrationSettings::get().setValue<float>("big_delta", parser->getBigDelta());
    RegistrationSettings::get().setValue<float>("small_delta", parser->getSmallDelta());


    RegistrationSettings::get().setValue<std::string>("step",parser->getStartStep());
    RegistrationSettings::get().setValue<bool>("do_QC",parser->getDoQC());
    RegistrationSettings::get().setValue<bool>("remove_temp",parser->getRemoveTempFolder());
    RegistrationSettings::get().setValue<std::string>("b0_mask_img",parser->getB0MaskName());


    RegistrationSettings::get().setValue<std::string>("denoising",parser->getDenoising());
    RegistrationSettings::get().setValue<int>("denoising_kernel_size",parser->getDenoisingKernelSize());

    RegistrationSettings::get().setValue<bool>("gibbs",parser->getGibbs());
    RegistrationSettings::get().setValue<float>("gibbs_kspace_coverage",parser->getGibbsKSpace());
    RegistrationSettings::get().setValue<int>("gibbs_nsh",parser->getGibbsNsh());
    RegistrationSettings::get().setValue<int>("gibbs_minW",parser->getGibbsMinW());
    RegistrationSettings::get().setValue<int>("gibbs_maxW",parser->getGibbsMaxW());


    RegistrationSettings::get().setValue<int>("b0_id",parser->getB0Id());
    RegistrationSettings::get().setValue<bool>("is_human_brain",parser->getIsHuman());
    RegistrationSettings::get().setValue<std::string>("rot_eddy_center",parser->getRotEddyCenter());
    RegistrationSettings::get().setValue<bool>("center_of_mass",parser->getCenterOfMass());
    RegistrationSettings::get().setValue<std::string>("correction_mode",parser->getCorrectionMode());
    RegistrationSettings::get().setValue<bool>("s2v",parser->getS2V());
    RegistrationSettings::get().setValue<bool>("repol",parser->getRepol());
    RegistrationSettings::get().setValue<float>("outlier_frac",parser->getOutlierFrac());
    RegistrationSettings::get().setValue<float>("outlier_prob",parser->getOutlierProbabilityThreshold());
    RegistrationSettings::get().setValue<int>("outlier_EM_clusters",parser->getOutlierNumberOfResidualClusters());
    RegistrationSettings::get().setValue<bool>("outlier_replacement_mode",parser->getOutlierReplacementModeAggessive());

    RegistrationSettings::get().setValue<int>("is_human_brain",parser->getIsHuman());
    RegistrationSettings::get().setValue<int>("niter",parser->getNiter());
    RegistrationSettings::get().setValue<int>("dti_bval",parser->getDTIBval());
    RegistrationSettings::get().setValue<int>("hardi_bval",parser->getHARDIBval());


    RegistrationSettings::get().setValue<std::string>("drift",parser->getDrift());

    RegistrationSettings::get().setValue<std::string>("epi",parser->getEPI());
    RegistrationSettings::get().setValue<bool>("DRBUDDI_disable_initial_rigid",parser->getDisableInitRigid());
    RegistrationSettings::get().setValue<bool>("DRBUDDI_start_with_diffeomorphic_for_rigid_reg",parser->getStartWithDiffeo());
    RegistrationSettings::get().setValue<std::string>("DRBUDDI_rigid_metric_type",parser->getRigidMetricType());
    RegistrationSettings::get().setValue<float>("DRBUDDI_rigid_metric_learning_rate",parser->getRigidLR());
    RegistrationSettings::get().setValue<int>("DRBUDDI_DWI_bval_tensor_fitting",parser->getDWIBvalue());

    RegistrationSettings::get().setValue<std::string>("output",parser->getOutputName());
    RegistrationSettings::get().setValue<std::string>("output_orientation",parser->getOutputOrientation());
    RegistrationSettings::get().setVectorValue<int>("output_voxels",parser->GetOutputNVoxels());
    RegistrationSettings::get().setVectorValue<float>("output_res",parser->GetOutputRes());
    RegistrationSettings::get().setValue<std::string>("output_gradnonlin_Bmtxt_type",parser->getOutputGradientNonlinearityType());
    RegistrationSettings::get().setValue<int>("interp_POW",parser->getPOW());

}


void TORTOISE::WriteFinalSettings()
{
    std::string up_name = this->parser->getUpInputName();
    fs::path up_path(up_name);
    std::string basename= fs::path(up_path).filename().string();
    basename=basename.substr(0,basename.rfind(".nii"));
    std::string settings_file= this->temp_proc_folder + std::string("/logs/") + basename + std::string("_proc_settings.dmc");


   // I decided to not the include the help comments. Cluttering the file. So the following line is commented out.
   // std::map<std::string, std::string> comments=RegistrationSettings::get().GetComments();

    std::ofstream file(settings_file.c_str());
    std::map<std::string, std::string>::iterator it;
    std::map<std::string, std::string> settings =RegistrationSettings::get().GetSettings();
    for (it = settings.begin(); it != settings.end(); it++)
    {
        std::string key= it->first ;
        std::string value = it->second ;

    //    file<< comments[key];
        file<< "<" << key << ">" << value << "</" << key << ">" << std::endl;
    }
    file.close();
}

void TORTOISE::entryPoint()
{
    this->Process();
}

void TORTOISE::Process()
{   
    if(ConvertStringToStep(parser->getStartStep())== STEPS::Import)  //these are self explanatory :)
    {
        (*stream)<<"Importing and copying data..."<<std::endl;
        CheckAndCopyInputData();
    }






    // Take care of little things normally done in the import step
    // in case we start from a later one.

    my_jsons.resize(2);                          //read JSONs
    for(int PE=0;PE<2;PE++)
    {
        if(this->proc_infos[PE].json_name!="")
        {
            std::ifstream json_file(this->proc_infos[PE].json_name);
            json_file >> this->my_jsons[PE];
            json_file.close();

            if(this->my_jsons[PE]["PhaseEncodingDirection"]==json::value_t::null)
            {
                if(this->my_jsons[PE]["PhaseEncodingAxis"]!=json::value_t::null)
                {
                    this->my_jsons[PE]["PhaseEncodingDirection"]=this->my_jsons[PE]["PhaseEncodingAxis"];
                }
                else
                {
                    if(this->my_jsons[PE]["InPlanePhaseEncodingDirectionDICOM"]!=json::value_t::null)
                    {
                        if(this->my_jsons[PE]["InPlanePhaseEncodingDirectionDICOM"]=="COL")
                        {
                            this->my_jsons[PE]["PhaseEncodingDirection"]="j";
                        }
                        else
                        {
                            this->my_jsons[PE]["PhaseEncodingDirection"]="i";
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

            if(this->my_jsons[PE]["PartialFourier"]==json::value_t::null)
            {
                if(this->my_jsons[PE]["PercentSampling"]!=json::value_t::null)
                {
                    int ps= this->my_jsons[PE]["PercentSampling"];
                    float psf= 1.*ps/100;

                    this->my_jsons[PE]["PartialFourier"]=psf;
                }
                else
                    this->my_jsons[PE]["PartialFourier"]=1;
            }
        }
    }

    //Check if the up and down Bmatrices are identical and set the output combination setting accordingly
    //we could not do this beforehand where it belongs before Bmatrices were not created yet.

    if(parser->getDownInputName()=="")
    {
        RegistrationSettings::get().setValue<std::string>("output_data_combination","");
    }
    else
    {
        bool do_jacobian=false;
        if(parser->getOutputDataCombination()=="Merge")
        {
           vnl_matrix<double> up_Bmatrix= read_bmatrix_file(proc_infos[0].bmtxt_name);
           vnl_matrix<double> down_Bmatrix= read_bmatrix_file(proc_infos[1].bmtxt_name);

           if(up_Bmatrix.rows()!=down_Bmatrix.rows())
               do_jacobian=true;
           else
           {
               vnl_matrix<double> Bmat_diff= up_Bmatrix- down_Bmatrix;
               vnl_vector<double> bmat_diff_norm(Bmat_diff.rows());

               vnl_vector<double> bvals= up_Bmatrix.get_column(0) + up_Bmatrix.get_column(3)+ up_Bmatrix.get_column(5);
               double max_bval= bvals.max_value();

               for(int i=0;i<up_Bmatrix.rows();i++)
               {
                   bmat_diff_norm[i] =  sqrt( Bmat_diff(i,0)*Bmat_diff(i,0) +  Bmat_diff(i,1)*Bmat_diff(i,1) +  Bmat_diff(i,2)*Bmat_diff(i,2) + Bmat_diff(i,3)*Bmat_diff(i,3)+  Bmat_diff(i,4)*Bmat_diff(i,4) + Bmat_diff(i,5)*Bmat_diff(i,5));
               }
               double diff_norm = bmat_diff_norm.mean();
               if(diff_norm > 0.05*max_bval)
                   do_jacobian=true;
           }
           if(!do_jacobian)
           {
               RegistrationSettings::get().setValue<std::string>("output_data_combination",parser->getOutputDataCombination());
           }
           else
           {
               if(down_Bmatrix.rows()<=2)
               {
                   (*stream)<<"UP AND DOWN DATA MERGE REQUESTED. HOWEVER, THEY DO NOT HAVE IDENTICAL BMATRICES. REVERTING TO JACSEP."<<std::endl;
                   RegistrationSettings::get().setValue<std::string>("output_data_combination","JacSep");
               }
               else
               {
                   (*stream)<<"UP AND DOWN DATA MERGE REQUESTED. HOWEVER, THEY DO NOT HAVE IDENTICAL BMATRICES. REVERTING TO JACCONCAT."<<std::endl;
                   RegistrationSettings::get().setValue<std::string>("output_data_combination","JacConcat");
               }
           }
        }
        else
        {
            RegistrationSettings::get().setValue<std::string>("output_data_combination",parser->getOutputDataCombination());
        }
    }





    if(parser->getGradNonlinInput()!="")    //update the gradient nonlinearity file to the adjusted one.
    {

        std::string nnm=parser->getGradNonlinInput();
        if(nnm.find(".nii")==std::string::npos)
        {
            RegistrationSettings::get().setValue<std::string>("grad_nonlin_coeffs", parser->getGradNonlinInput());
        }
        else
        {
            RegistrationSettings::get().setValue<std::string>("grad_nonlin_coeffs", "");
        }

        std::string up_name = this->parser->getUpInputName();
        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string gradnonlin_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc_gradnonlin_field.nii");
        parser->setGradNonlinInput(gradnonlin_name);
        RegistrationSettings::get().setValue<std::string>("grad_nonlin", parser->getGradNonlinInput());
    }


    if(ConvertStringToStep(parser->getStartStep())<= STEPS::Denoising) //these are self explanatory :))))
    {
        for(int PE=0;PE<2;PE++)
        {
            double b0_noise_mean,b0_noise_std;

            DenoiseData(this->proc_infos[PE].nii_name,b0_noise_mean,b0_noise_std);

            this->my_jsons[PE]["b0_noise_mean"]=b0_noise_mean;
            this->my_jsons[PE]["b0_noise_std"]=b0_noise_std;
            std::ofstream out_json(this->proc_infos[PE].json_name);
            out_json << std::setw(4) << this->my_jsons[PE] << std::endl;
            out_json.close();
        }
    }

    if(ConvertStringToStep(parser->getStartStep())<= STEPS::Gibbs)
    {
        for(int PE=0;PE<2;PE++)
        {

            if(this->proc_infos[PE].nii_name!="")
                GibbsUnringData(this->proc_infos[PE].nii_name,this->my_jsons[PE]["PartialFourier"],this->my_jsons[PE]["PhaseEncodingDirection"] );
        }
    }

    if(ConvertStringToStep(parser->getStartStep())<= STEPS::MotionEddy)
    {        
        for(int PE=0;PE<2;PE++)
        {
            if(this->proc_infos[PE].nii_name!="")
            {
                (*stream)<<"Starting DIFFPREP correction for data:"<<PE<<std::endl;
                DIFFPREP myDIFFPREP(this->proc_infos[PE].nii_name,this->my_jsons[PE]);
            }
        }
    }

    if(ConvertStringToStep(parser->getStartStep())<= STEPS::Drift)
    {
        std::string drift_option= RegistrationSettings::get().getValue<std::string>("drift");

        if(drift_option!="off")
        {
            for(int PE=0;PE<2;PE++)
            {
                (*stream)<<"Drift correcting dataset: "<<PE<<std::endl;
                if(this->proc_infos[PE].nii_name!="")
                {
                    std::string orig_name=this->proc_infos[PE].nii_name;
                    std::string DP_name = orig_name.substr(0,orig_name.rfind(".nii")) + "_moteddy.nii";

                    DriftCorrect(DP_name);
                }
            }
        }
    }
    //re read jsons after DIFFPREP changes it.
    for(int PE=0;PE<2;PE++)
    {
        if(this->proc_infos[PE].json_name!="")
        {
            std::ifstream json_file(this->proc_infos[PE].json_name);
            json_file >> this->my_jsons[PE];
            json_file.close();

            if(this->my_jsons[PE]["PhaseEncodingDirection"]==json::value_t::null)
            {
                if(this->my_jsons[PE]["PhaseEncodingAxis"]!=json::value_t::null)
                {
                    this->my_jsons[PE]["PhaseEncodingDirection"]=this->my_jsons[PE]["PhaseEncodingAxis"];
                }
                else
                {
                    if(this->my_jsons[PE]["InPlanePhaseEncodingDirectionDICOM"]!=json::value_t::null)
                    {
                        if(this->my_jsons[PE]["InPlanePhaseEncodingDirectionDICOM"]=="COL")
                        {
                            this->my_jsons[PE]["PhaseEncodingDirection"]="j";
                        }
                        else
                        {
                            this->my_jsons[PE]["PhaseEncodingDirection"]="i";
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

            if(this->my_jsons[PE]["PartialFourier"]==json::value_t::null)
            {
                if(this->my_jsons[PE]["PercentSampling"]!=json::value_t::null)
                {
                    int ps= this->my_jsons[PE]["PercentSampling"];
                    float psf= 1.*ps/100;

                    this->my_jsons[PE]["PartialFourier"]=psf;
                }
                else
                    this->my_jsons[PE]["PartialFourier"]=1;
            }
        }
    }

    if(ConvertStringToStep(parser->getStartStep())<= STEPS::EPI)
    {
        (*stream)<<"Starting EPI distortion correction..."<<std::endl;
        EPICorrectData();
    }

    if(ConvertStringToStep(parser->getStartStep())<= STEPS::StructuralAlignment)
    {
        (*stream)<<"Starting b=0 to structural registration..."<<std::endl;
        AlignB0ToReorientation();
    }


    if(ConvertStringToStep(parser->getStartStep())<= STEPS::FinalData)
    {
        std::string denoising_option= RegistrationSettings::get().getValue<std::string>("denoising");
        bool gibbs_option= RegistrationSettings::get().getValue<bool>("gibbs");

        if(denoising_option=="for_reg" && gibbs_option)
        {
            (*stream)<<"Denoising option was set for only correction, not final data. Copying the original data one last time."<<std::endl;
            CheckAndCopyInputData();
            for(int PE=0;PE<2;PE++)
            {
                if(this->proc_infos[PE].nii_name!="")
                {
                    GibbsUnringData(this->proc_infos[PE].nii_name,this->my_jsons[PE]["PartialFourier"],this->my_jsons[PE]["PhaseEncodingDirection"]);
                }
            }
        }

        (*stream)<<"Writing final outputs..."<<std::endl;
        FINALDATA my_final_data_generator(this->proc_infos[0].nii_name,this->my_jsons[0],this->proc_infos[1].nii_name,this->my_jsons[1]);
        my_final_data_generator.SetTempFolder(this->temp_proc_folder);
        my_final_data_generator.SetParser(this->parser);
        my_final_data_generator.SetOutputName(this->output_name);
        my_final_data_generator.Generate();
    }

    if(parser->getRemoveTempFolder())
    {
        fs::remove_all(this->temp_proc_folder);
    }

    (*stream)<<"Done with TORTOISE processing...Congratulations...."<<std::endl;
    (*stream)<<"Your final data is: "<<this->output_name <<std::endl;

}


void TORTOISE::DriftCorrect(std::string nii_name)
{
    std::string drift_option= RegistrationSettings::get().getValue<std::string>("drift");

    std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii")) + ".bmtxt";

    vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxt_name);
    vnl_vector<double> bvals= Bmatrix.get_column(0) +  Bmatrix.get_column(3) + Bmatrix.get_column(5);

    int Nvols= bvals.size();

    float b0_bval = bvals.min_value();
    std::vector<int> b0_ids;
    for(int v=0;v<Nvols;v++)
    {
        if(fabs(bvals[v]-b0_bval) <10)
            b0_ids.push_back(v);
    }

    if(Nvols<30)
    {
        (*stream)<<"Less than 30 volumes for the data. Skipping signal drift correction.."<<std::endl;
        return;
    }

    if(b0_ids.size()<4)
    {
        (*stream)<<"Less than five b=0 volumes for the data. Skipping signal drift correction.."<<std::endl;
        return;
    }


    // Check if b0s are spread nicely
    int max_adjacent_b0_separation=-1;
    for(int v=1;v<b0_ids.size();v++)
    {
        int dist = b0_ids[v] - b0_ids[v-1];
        if(dist>max_adjacent_b0_separation)
            max_adjacent_b0_separation=dist;
    }
    max_adjacent_b0_separation*=1.2;

    std::vector<bool> covered;
    covered.resize(Nvols);
    for(int v=0;v<Nvols;v++)
        covered[v]=0;

    for(int v=0;v<b0_ids.size();v++)
    {
        int vol_id = b0_ids[v];
        int start= std::max(0,vol_id-max_adjacent_b0_separation);
        int end= std::min(Nvols,vol_id+max_adjacent_b0_separation);
        for(int v2=start;v2<end;v2++)
            covered[v2]=1;
    }
    bool all_covered=true;
    for(int v2=0;v2<Nvols;v2++)
    {
        if(!covered[v2])
        {
            all_covered=false;
            break;
        }
    }

    if(!all_covered)
    {
        (*stream)<<"The spread of b0s within the data not ideal. Skipping signal drift correction.."<<std::endl;
        return;
    }


    //All tests passed. Doing signal drift estimation

    ImageType3D::Pointer b0_img = read_3D_volume_from_4D(nii_name,b0_ids[0]);
    ImageType3D::Pointer mask_img= nullptr;
    std::string b0_mask_img_fname = RegistrationSettings::get().getValue<std::string>("b0_mask_img");
    if(b0_mask_img_fname=="")
    {
        mask_img= create_mask(b0_img,nullptr);
    }
    else
    {
        mask_img=readImageD<ImageType3D>(b0_mask_img_fname);
    }


    std::string inc_name = nii_name.substr(0,nii_name.rfind(".nii")) + "_inc.nii";


    vnl_vector<double> mean_signals(b0_ids.size(),0);
    for(int v=0;v<b0_ids.size();v++)
    {
        ImageType3D::Pointer b0_img = read_3D_volume_from_4D(nii_name,b0_ids[v]);
        ImageType3DBool::Pointer inc_img = nullptr;
        if(fs::exists(inc_name))
            inc_img=read_3D_volume_from_4DBool(inc_name,b0_ids[v]);

        itk::ImageRegionIteratorWithIndex<ImageType3D> it(b0_img,b0_img->GetLargestPossibleRegion());
        it.GoToBegin();
        long N=0;
        double sm=0;
        while(!it.IsAtEnd())
        {
            ImageType3D::IndexType ind3= it.GetIndex();
            if(mask_img->GetPixel(ind3))
            {
                if((inc_img && inc_img->GetPixel(ind3)) || !inc_img )
                {
                    sm+=it.Get();
                    N++;
                }
            }
            ++it;
        }
        if(N>0)
            mean_signals[v]=sm/N;
        else
            mean_signals[v]=0;
    }


    std::string drift_name = nii_name.substr(0,nii_name.rfind(".nii")) + "_drift.txt";

    //Least squares
    if(drift_option=="linear")
    {
        int first_good_b0_id=0;
        for(int v=0;v<b0_ids.size();v++)
        {
            if(mean_signals[v]!=0)
            {
                first_good_b0_id=v;
                break;
            }
        }


        double nom=0,denom=0;
        for(int v=0;v<b0_ids.size();v++)
        {
            if(mean_signals[v]!=0)
            {
                nom+= (mean_signals[v] - mean_signals[first_good_b0_id])*(b0_ids[v] -b0_ids[first_good_b0_id]) ;
                denom+= (b0_ids[v] -b0_ids[first_good_b0_id])*(b0_ids[v] -b0_ids[first_good_b0_id]);
            }
        }
        double slope = nom/denom;


        std::ofstream outfile(drift_name);
        outfile<<"linear"<<std::endl;
        if(mean_signals[0]!=0)
            outfile<<mean_signals[0] << " " << slope;
        else
        {
            outfile<<mean_signals[first_good_b0_id]-slope*b0_ids[first_good_b0_id] << " " << slope;
        }

        //outfile<<"S_n^c = S_n "<<mean_signals[0]<<"/("<<slope<<    " vol_id + "<< mean_signals[0]<< ")";
        outfile.close();
    }
    else
    {
        vnl_matrix<double> Dmat(2,2,0);
        vnl_vector<double> f(2,0);

        for(int v=0;v<b0_ids.size();v++)
        {
            Dmat(0,0)+= b0_ids[v]*b0_ids[v]*b0_ids[v]*b0_ids[v];
            Dmat(0,1)+= b0_ids[v]*b0_ids[v]*b0_ids[v];
            Dmat(1,0)+= b0_ids[v]*b0_ids[v]*b0_ids[v];
            Dmat(1,1)+= b0_ids[v]*b0_ids[v];

            f[0]+= (mean_signals[v] - mean_signals[0])*b0_ids[v]*b0_ids[v];
            f[1]+= (mean_signals[v] - mean_signals[0])*b0_ids[v];
        }

        vnl_vector<double> coeffs =vnl_matrix_inverse<double>(Dmat) *f;

        std::ofstream outfile(drift_name);
        outfile<<"quadratic"<<std::endl;
        outfile<<mean_signals[0] << " " << coeffs[0]<< " "<< coeffs[1];
       // outfile<<"S_n^c = S_n "<<mean_signals[0]<<"/("<< coeffs[0] <<"vol_id^2 + " << coeffs[1] << "vol_id + "<< mean_signals[0]<< ")";
        outfile.close();

    }
}



void TORTOISE::AlignB0ToReorientation()
{
    std::vector<std::string>  structural_names = RegistrationSettings::get().getVectorValue<std::string>("structural");
    std::string reorientation_name= RegistrationSettings::get().getValue<std::string>("reorientation");


    RigidTransformType::Pointer b0_to_str_trans=nullptr;
    ImageType3D::Pointer target_img=nullptr;


    if(reorientation_name!="")
        target_img = readImageD<ImageType3D>(reorientation_name);
    else
    {
        if(structural_names.size())
            target_img = readImageD<ImageType3D>(structural_names[0]);
    }

    if(target_img)
    {
        ImageType3D::Pointer b0_img=nullptr;

        if(fs::exists(this->temp_proc_folder + std::string("/b0_corrected_final.nii")))
            b0_img = readImageD<ImageType3D>(this->temp_proc_folder + std::string("/b0_corrected_final.nii"));
        else if(fs::exists(this->temp_proc_folder + std::string("/blip_up_b0_corrected_JAC.nii")))
            b0_img= readImageD<ImageType3D>(this->temp_proc_folder + std::string("/blip_up_b0_corrected_JAC.nii"));
        else
            b0_img= read_3D_volume_from_4D(this->proc_infos[0].nii_name,0);


        RigidTransformType::Pointer rigid_trans1= RigidRegisterImagesEuler( target_img,  b0_img, "CC",parser->getRigidLR());
        RigidTransformType::Pointer rigid_trans2= RigidRegisterImagesEuler( target_img,  b0_img, "MI",parser->getRigidLR());

        auto params1= rigid_trans1->GetParameters();
        auto params2= rigid_trans2->GetParameters();
        auto p1=params1-params2;

        double diff=0;
        diff+= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] +
               p1[3]*p1[3]/400. + p1[4]*p1[4]/400. + p1[5]*p1[5]/400. ;

        RigidTransformType::Pointer rigid_trans=nullptr;
        std::cout<<"R1: "<< params1<<std::endl;
        std::cout<<"R2: "<< params2<<std::endl;
        std::cout<<"MI vs CC diff: "<< diff<<std::endl;


        if(diff<0.005)
            b0_to_str_trans=rigid_trans2;
        else
        {
            std::cout<<"Could not compute the rigid transformation from the structural imageto b=0 image... Starting multistart.... This could take a while"<<std::endl;
            std::cout<<"Better be safe than sorry, right?"<<std::endl;


            std::vector<float> new_res; new_res.resize(3);
            new_res[0]= b0_img->GetSpacing()[0] * 2;
            new_res[1]= b0_img->GetSpacing()[1] * 2;
            new_res[2]= b0_img->GetSpacing()[2] * 2;
            std::vector<float> dummy;
            ImageType3D::Pointer b02= resample_3D_image(b0_img,new_res,dummy,"Linear");
            new_res[0]= target_img->GetSpacing()[0] * 2;
            new_res[1]= target_img->GetSpacing()[1] * 2;
            new_res[2]= target_img->GetSpacing()[2] * 2;
            ImageType3D::Pointer str2= resample_3D_image(target_img,new_res,dummy,"Linear");

            rigid_trans1=MultiStartRigidSearch(str2,b02);
            b0_to_str_trans= RigidRegisterImagesEuler( target_img, b0_img,   "MI",parser->getRigidLR()/2.,rigid_trans1);
        }
        (*stream)<<"Final transformation: " << b0_to_str_trans->GetParameters()<<std::endl;

    }
    else
    {
        b0_to_str_trans = RigidTransformType::New();
        b0_to_str_trans->SetIdentity();
    }

    std::string trans_name= this->temp_proc_folder + std::string("/b0_to_str_rigidtrans.hdf5");
    typedef itk::TransformFileWriterTemplate< double > TransformWriterType;
    TransformWriterType::Pointer trwriter = TransformWriterType::New();
    trwriter->SetInput(b0_to_str_trans);
    trwriter->SetFileName(trans_name);
    trwriter->Update();
}


void TORTOISE::EPICorrectData()
{
    std::string epi_option= RegistrationSettings::get().getValue<std::string>("epi");
    if(epi_option=="off")
        return;

    std::vector<std::string> structural_names = RegistrationSettings::get().getVectorValue<std::string>("structural");

    std::string up_name;
    {
        std::string nname= fs::path(this->proc_infos[0].nii_name).filename().string();
        std::string basename = nname.substr(0, nname.find(".nii"));
        up_name = this->temp_proc_folder  + "/"  +basename + std::string("_moteddy.nii");
    }

    if(epi_option=="T2Wreg")
    {
        EPIREG myEPIREG(up_name,structural_names,this->my_jsons[0]);
        myEPIREG.SetParser(this->parser);
        myEPIREG.Process();
    }

    if(epi_option=="DRBUDDI")
    {
        std::string nname= fs::path(this->proc_infos[1].nii_name).filename().string();
        std::string basename = nname.substr(0, nname.find(".nii"));
        std::string down_name = this->temp_proc_folder  + "/"  +basename + std::string("_moteddy.nii");

        DRBUDDI myDRBUDDI(up_name,down_name,structural_names,this->my_jsons[0]);
        myDRBUDDI.SetParser(this->parser);
        if(parser->getB0MaskName()!="")
        {
            ImageType3D::Pointer mask_img= readImageD<ImageType3D>(parser->getB0MaskName());
            myDRBUDDI.SetMaskImg(mask_img);
        }


        myDRBUDDI.Process();
    }
}

void TORTOISE::GibbsUnringData(std::string input_name, float PF,std::string json_PE)
{
    bool gibbs_option= RegistrationSettings::get().getValue<bool>("gibbs");
    float gibbs_kspace_coverage= RegistrationSettings::get().getValue<float>("gibbs_kspace_coverage");
    int gibbs_nsh= RegistrationSettings::get().getValue<int>("gibbs_nsh");
    int gibbs_minW= RegistrationSettings::get().getValue<int>("gibbs_minW");
    int gibbs_maxW= RegistrationSettings::get().getValue<int>("gibbs_maxW");


    short phase=0;
    if(json_PE.find("j")!=std::string::npos)
       phase=1;
    else
        if(json_PE.find("i")!=std::string::npos)
            phase=0;
        else
            phase=2;


    ImageType4D::DirectionType orig_dir;
    ImageType4D::SpacingType orig_spc;
    ImageType4D::RegionType orig_reg;
    ImageType4D::PointType orig_or;

    if(input_name!="" && gibbs_option)
    {
        float ks_cov= PF;
        if(gibbs_kspace_coverage!=0)             //use the command line enforced coverage instead of
            ks_cov = gibbs_kspace_coverage;      // what is reported in the json file.

        ImageType4D::Pointer dwis= readImageD<ImageType4D>(input_name);
        if(phase==0)
        {

            orig_dir=dwis->GetDirection();
            orig_spc=dwis->GetSpacing();
            orig_reg=dwis->GetLargestPossibleRegion();
            orig_or =dwis->GetOrigin();


            ImageType4D::SizeType new_size;
            new_size[0]=dwis->GetLargestPossibleRegion().GetSize()[1];
            new_size[1]=dwis->GetLargestPossibleRegion().GetSize()[0];
            new_size[2]=dwis->GetLargestPossibleRegion().GetSize()[2];
            new_size[3]=dwis->GetLargestPossibleRegion().GetSize()[3];

            ImageType4D::IndexType start; start.Fill(0);
            ImageType4D::RegionType reg(start,new_size);

            ImageType4D::SpacingType new_spc;

            ImageType4D::Pointer dwis2= ImageType4D::New();
            dwis2->SetRegions(reg);
            dwis2->Allocate();            
            dwis2->FillBuffer(0);

            itk::ImageRegionIteratorWithIndex<ImageType4D> it(dwis2,dwis2->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ImageType4D::IndexType ind4= it.GetIndex();
                ImageType4D::IndexType ind4_old=ind4;
                ind4_old[0]=ind4[1];
                ind4_old[1]=ind4[0];
                it.Set(dwis->GetPixel(ind4_old));
            }
            dwis=dwis2;
        }


        if(ks_cov>=0.9375)
        {
            (*stream)<<"Gibbs correction with full k-space coverage"<<std::endl;
            dwis=UnRingFull(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
        }
        else
        {
            if(ks_cov<0.9375 && ks_cov >=0.8125)
            {
                (*stream)<<"Gibbs correction with 7/8 k-space coverage"<<std::endl;
                dwis=UnRing78(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
            }
            else
            {
                if(ks_cov>0.65)
                {
                    (*stream)<<"Gibbs correction with 6/8 k-space coverage"<<std::endl;
                    dwis=UnRing68(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
                }
                else
                {
                    (*stream)<<"K-space coverage in the data is less than 65\%. Skipping Gibbs ringing correction. "<<std::endl;
                }
            }
        }
        (*stream)<<std::endl;
        if(phase==0)
        {            
            ImageType4D::Pointer dwis2= ImageType4D::New();
            dwis2->SetRegions(orig_reg);
            dwis2->SetSpacing(orig_spc);
            dwis2->SetDirection(orig_dir);
            dwis2->SetOrigin(orig_or);
            dwis2->Allocate();
            dwis2->FillBuffer(0);

            itk::ImageRegionIteratorWithIndex<ImageType4D> it(dwis2,dwis2->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ImageType4D::IndexType ind4= it.GetIndex();
                ImageType4D::IndexType ind4_old=ind4;
                ind4_old[0]=ind4[1];
                ind4_old[1]=ind4[0];
                it.Set(dwis->GetPixel(ind4_old));
            }
            dwis=dwis2;
        }



        writeImageD<ImageType4D>(dwis,input_name);
    }
}


void TORTOISE::DenoiseData(std::string input_name,double &b0_noise_mean, double &b0_noise_std)
{
    std::string denoising_option= RegistrationSettings::get().getValue<std::string>("denoising");
    int kernel_size = RegistrationSettings::get().getValue<int>("denoising_kernel_size");

    if(input_name!="")
    {
        ImageType3D::Pointer noise_img=nullptr;
        std::string noise_filename= input_name.substr(0,input_name.find(".nii"))+ std::string("_noise.nii");

        if(denoising_option=="off")
        {
            (*stream)<<"Estimating noise levels..."<<std::endl;
            ImageType4D::Pointer dwis= readImageD<ImageType4D>(input_name);

            double noise_mean=0;
            ImageType4D::Pointer denoised_img= DWIDenoise(dwis, noise_img,noise_mean,false,kernel_size);
            b0_noise_std=noise_mean;
            b0_noise_mean= 1.2532*noise_mean;
            writeImageD<ImageType3D>(noise_img,noise_filename);
        }
        else
        {
            (*stream)<<"Denoising DWIs for use in registration.."<<std::endl;

            ImageType4D::Pointer dwis= readImageD<ImageType4D>(input_name);
            double noise_mean=0;
            ImageType4D::Pointer denoised_img= DWIDenoise(dwis, noise_img,noise_mean,true,kernel_size);
            b0_noise_std=noise_mean;
            b0_noise_mean= 1.2532*noise_mean;
            writeImageD<ImageType4D>(denoised_img,input_name);
            writeImageD<ImageType3D>(noise_img,noise_filename);
        }
    }

}

void TORTOISE::CheckAndCopyInputData()
{
    int flipX=parser->getFlipX();
    int flipY=parser->getFlipY();
    int flipZ=parser->getFlipZ();

    std::vector<std::string> input_names={parser->getUpInputName(),parser->getDownInputName()};
    std::vector<std::string> bval_names={parser->getUpBvalName(),parser->getDownBvalName()};
    std::vector<std::string> bvec_names={parser->getUpBvecName(),parser->getDownBvecName()};

    for(int PE=0;PE<2;PE++)  //for both up and down data, do the following
    {
        std::string input_name=input_names[PE];
        if(input_name!="")
        {
            // First copy the nifti files to the proc folder

            itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
            myio->SetFileName(input_name);
            myio->ReadImageInformation();
            bool is3D= (myio->GetNumberOfDimensions()==3);
            if(is3D)
            {
                ImageType3D::Pointer fvol= readImageD<ImageType3D>(input_name);
                ImageType4D::Pointer two_vols= ImageType4D::New();
                ImageType4D::DirectionType dir;
                dir.SetIdentity();
                dir(0,0)= fvol->GetDirection()(0,0);dir(0,1)= fvol->GetDirection()(0,1);dir(0,2)= fvol->GetDirection()(0,2);
                dir(1,0)= fvol->GetDirection()(1,0);dir(1,1)= fvol->GetDirection()(1,1);dir(1,2)= fvol->GetDirection()(1,2);
                dir(2,0)= fvol->GetDirection()(2,0);dir(2,1)= fvol->GetDirection()(2,1);dir(2,2)= fvol->GetDirection()(2,2);
                two_vols->SetDirection(dir);

                ImageType4D::IndexType start; start.Fill(0);
                ImageType4D::SizeType sz4;
                sz4[0]=fvol->GetLargestPossibleRegion().GetSize()[0];
                sz4[1]=fvol->GetLargestPossibleRegion().GetSize()[1];
                sz4[2]=fvol->GetLargestPossibleRegion().GetSize()[2];
                sz4[3]=2;
                ImageType4D::RegionType reg4(start,sz4);
                two_vols->SetRegions(reg4);
                two_vols->Allocate();
                ImageType4D::SpacingType spc4;
                spc4[0]= fvol->GetSpacing()[0];
                spc4[1]= fvol->GetSpacing()[1];
                spc4[2]= fvol->GetSpacing()[2];
                spc4[3]=1;
                two_vols->SetSpacing(spc4);
                ImageType4D::PointType orig4;
                orig4[0]= fvol->GetOrigin()[0];
                orig4[1]= fvol->GetOrigin()[1];
                orig4[2]= fvol->GetOrigin()[2];
                orig4[3]=0;
                two_vols->SetOrigin(orig4);
                itk::ImageRegionIteratorWithIndex<ImageType3D> it3(fvol,fvol->GetLargestPossibleRegion());
                it3.GoToBegin();
                while(!it3.IsAtEnd())
                {
                    ImageType3D::IndexType ind3 = it3.GetIndex();
                    ImageType4D::IndexType ind4;
                    ind4[0]=ind3[0];
                    ind4[1]=ind3[1];
                    ind4[2]=ind3[2];
                    ind4[3]=0;
                    two_vols->SetPixel(ind4,it3.Get());
                    ind4[3]=1;
                    two_vols->SetPixel(ind4,it3.Get());

                    ++it3;
                }
                writeImageD<ImageType4D>(two_vols,input_name);


                vnl_matrix<double> newbvec(3,2,0);
                vnl_matrix<double> newbval(1,2,0);               
                {
                    std::string nm= input_name.substr(0,input_name.rfind(".nii"))+".bvec";
                    std::ofstream outfile(nm.c_str());
                    outfile<<newbvec;
                    outfile.close();
                }
                {
                    std::string nm= input_name.substr(0,input_name.rfind(".nii"))+".bval";
                    std::ofstream outfile(nm.c_str());
                    outfile<<newbval;
                    outfile.close();
                }
            }

            //and finally copy JSON file
            std::string input_basename = input_name.substr(0,input_name.rfind(".nii"));
            std::string json_name=input_basename + std::string(".json");
            bool json_exists =  fs::exists(json_name);
            if(json_exists)
            {
                fs::copy_file(input_basename+std::string(".json"), this->proc_infos[PE].json_name, fs::copy_option::overwrite_if_exists );
            }
            else
            {
                json_name=input_basename + std::string(".JSON");
                fs::copy_file(json_name, this->proc_infos[PE].json_name, fs::copy_option::overwrite_if_exists );
            }


            //Copy the data
            {
                ImageType4D::Pointer in_img= readImageD<ImageType4D>(input_name);

                json temp_json;
                std::ifstream json_file(json_name);
                json_file >> temp_json;
                json_file.close();
                if(temp_json["SliceTiming"]!=json::value_t::null)
                {
                    std::vector<float> slice_timing= temp_json["SliceTiming"];
                    int Nslices_from_json= slice_timing.size();
                    int Nslices_image= in_img->GetLargestPossibleRegion().GetSize()[2];

                    if(Nslices_image!=Nslices_from_json)
                    {
                        (*stream)<<"WARNING: Number of slices in the image (" <<Nslices_image << ") does NOT match the json file (" <<Nslices_from_json<<")."<<std::endl;
                        (*stream)<<"ARE YOU SURE EVERYTHING IS CORRECT?"<<std::endl;
                        (*stream)<<"Padding/cropping the image to match the json file. If this is not desired, please manually fix the issue."<<std::endl;


                        ImageType4D::SizeType sz= in_img->GetLargestPossibleRegion().GetSize();
                        sz[2]=Nslices_from_json;

                        ImageType4D::Pointer in_img2= ImageType4D::New();
                        ImageType4D::IndexType start; start.Fill(0);
                        ImageType4D::RegionType reg(start,sz);
                        in_img2->SetRegions(reg);
                        in_img2->Allocate();
                        in_img2->SetSpacing(in_img->GetSpacing());
                        in_img2->SetDirection(in_img->GetDirection());
                        in_img2->SetOrigin(in_img->GetOrigin());
                        in_img2->FillBuffer(0);

                        if(Nslices_from_json>Nslices_image)
                            sz[2]=Nslices_image;
                        reg.SetSize(sz);

                        itk::ImageRegionIteratorWithIndex<ImageType4D> it(in_img,reg);
                        for(it.GoToBegin();!it.IsAtEnd();++it)
                        {
                            ImageType4D::IndexType ind4= it.GetIndex();
                            in_img2->SetPixel(ind4,it.Get());
                        }
                        in_img= in_img2;
                    }
                }

                writeImageD<ImageType4D>(in_img,this->proc_infos[PE].nii_name);
            }






            // Then deal with BMTXT or bvecs/bvals


            bool bmtxt_exists = (fs::exists(input_basename + std::string(".bmtxt")));
            if(bmtxt_exists)
            {
                vnl_matrix<double> Bmatrix= read_bmatrix_file(input_basename+std::string(".bmtxt"));
                for(int v=0;v< Bmatrix.rows();v++)
                {
                    Bmatrix(v,1)*= flipX*flipY;
                    Bmatrix(v,2)*= flipX*flipZ;
                    Bmatrix(v,4)*= flipY*flipZ;
                }
                std::ofstream bmat_stream(this->proc_infos[PE].bmtxt_name);
                bmat_stream<<Bmatrix;
                bmat_stream.close();

                //fs::copy_file(input_basename+std::string(".bmtxt"), this->proc_infos[PE].bmtxt_name, fs::copy_option::overwrite_if_exists );
            }
            else
            {

                itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
                myio->SetFileName(input_name);
                myio->ReadImageInformation();
                int Nvols= myio->GetDimensions(3);

                vnl_matrix<double> bvecs(3,Nvols);
                vnl_matrix<double> bvecs_transposed(Nvols,3);
                bool use_transposed_bvecs=false;
                vnl_vector<double> bvals(Nvols);
                vnl_matrix<double> Bmatrix(Nvols,6);

                std::string bvecs_file, bvals_file;
                if(bval_names[PE]!="" && bvec_names[PE]!="")
                {
                    bvecs_file=bvec_names[PE];
                    bvals_file=bval_names[PE];
                }
                else
                {
                    bool bvecs_exists = (fs::exists(input_basename + std::string(".bvecs")));
                    bool bvals_exists = (fs::exists(input_basename + std::string(".bvals")));
                    if(bvecs_exists && bvals_exists)
                    {
                        bvecs_file=input_basename + std::string(".bvecs");
                        bvals_file=input_basename + std::string(".bvals");
                    }
                    else
                    {
                        bool bvec_exists = (fs::exists(input_basename + std::string(".bvec")));
                        bool bval_exists = (fs::exists(input_basename + std::string(".bval")));
                        if(bvec_exists && bval_exists)
                        {
                            bvecs_file=input_basename + std::string(".bvec");
                            bvals_file=input_basename + std::string(".bval");
                        }
                        else
                        {
                            bool BVECS_exists = (fs::exists(input_basename + std::string(".BVECS")));
                            bool BVALS_exists = (fs::exists(input_basename + std::string(".BVALS")));
                            if(BVECS_exists && BVALS_exists)
                            {
                                bvecs_file=input_basename + std::string(".BVECS");
                                bvals_file=input_basename + std::string(".BVALS");
                            }
                            else
                            {
                                bvecs_file=input_basename + std::string(".BVEC");
                                bvals_file=input_basename + std::string(".BVAL");
                            }
                        }
                    }
                }
                std::ifstream infileb(bvals_file.c_str());
                infileb>>bvals;
                infileb.close();

                std::string line;
                std::ifstream infile(bvecs_file.c_str());
                int nlines=0;
                while (std::getline(infile, line))
                {
                    line.erase(remove(line.begin(), line.end(), ' '), line.end());
                    if(line.length()>1)
                        nlines++;
                }
                infile.close();
                if(nlines>3)
                    use_transposed_bvecs=true;

                std::ifstream infile2(bvecs_file.c_str());
                if(use_transposed_bvecs)
                {
                    infile2>>bvecs_transposed;
                    bvecs= bvecs_transposed.transpose();
                }
                else
                    infile2>>bvecs;
                infile2.close();

                for(int i=0;i<Nvols;i++)
                {
                    vnl_matrix<double> vec= bvecs.get_n_columns(i,1);
                    double nrm= sqrt(vec(0,0)*vec(0,0) + vec(1,0)*vec(1,0) + vec(2,0)*vec(2,0) );
                    if(nrm > 1E-3)
                    {
                        vec(0,0)/=nrm;
                        vec(1,0)/=nrm;
                        vec(2,0)/=nrm;
                    }

                    vec(0,0)*= flipX;
                    vec(1,0)*= flipY;
                    vec(2,0)*= flipZ;

                    vnl_matrix<double> mat = bvals[i] * vec * vec.transpose();
                    Bmatrix(i,0)=mat(0,0);
                    Bmatrix(i,1)=2*mat(0,1);
                    Bmatrix(i,2)=2*mat(0,2);
                    Bmatrix(i,3)=mat(1,1);
                    Bmatrix(i,4)=2*mat(1,2);
                    Bmatrix(i,5)=mat(2,2);
                }

                std::ofstream outfile(this->proc_infos[PE].bmtxt_name.c_str());
                outfile<<Bmatrix;
                outfile.close();

           } // if !bmtxt

        } //if input exists
    } //for PE


    // Copy the gradient nonlinearity file in ITK displacement field format and make necessary changes.
    if(parser->getGradNonlinInput()!="")
    {
        //Gradpwarp field is already in ITK displacement field format
        // so just copy it into the temp folder

        std::string up_name = this->parser->getUpInputName();
        fs::path up_path(up_name);
        std::string basename= fs::path(up_path).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string gradnonlin_inv_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc_gradnonlin_field_inv.nii");
        std::string gradnonlin_name= this->temp_proc_folder + std::string("/") + basename + std::string("_proc_gradnonlin_field.nii");

        DisplacementFieldType::Pointer gradwarp_field_inv=nullptr;
        if(parser->getGradNonlinInput().find(".nii")!=std::string::npos)
        {
            gradwarp_field_inv = readImageD<DisplacementFieldType>(parser->getGradNonlinInput());
            writeImageD<DisplacementFieldType>(gradwarp_field_inv,gradnonlin_inv_name);

            RegistrationSettings::get().setValue<std::string>("grad_nonlin_coeffs", "");
        }
        else
        {
            RegistrationSettings::get().setValue<std::string>("grad_nonlin_coeffs", parser->getGradNonlinInput());

            //Gradpwarp field is in coefficients format. convert it to a gradwarp_field.
            ImageType3D::Pointer ref_img = read_3D_volume_from_4D(this->proc_infos[0].nii_name,0);

            bool is_GE=parser->getGradNonlinIsGE();
            gradwarp_field_inv= mk_displacement(parser->getGradNonlinInput(),ref_img,is_GE);
            writeImageD<DisplacementFieldType>(gradwarp_field_inv,gradnonlin_inv_name);
        }

        bool is_GE=parser->getGradNonlinIsGE();
        if(is_GE)
        {
            ImageType3D::Pointer fvol = read_3D_volume_from_4D(this->proc_infos[0].nii_name,0);
            itk::ContinuousIndex<double,3> center_ind;
            center_ind[0]= ((int)(fvol->GetLargestPossibleRegion().GetSize()[0])-1)/2.;
            center_ind[1]= ((int)(fvol->GetLargestPossibleRegion().GetSize()[1])-1)/2.;
            center_ind[2]= ((int)(fvol->GetLargestPossibleRegion().GetSize()[2])-1)/2.;

            ImageType3D::PointType center_pt;
            fvol->TransformContinuousIndexToPhysicalPoint(center_ind,center_pt);

            ImageType3D::PointType new_orig=gradwarp_field_inv->GetOrigin();
            new_orig[2]+= center_pt[2];
            gradwarp_field_inv->SetOrigin(new_orig);
            writeImageD<DisplacementFieldType>(gradwarp_field_inv,gradnonlin_inv_name);
        }
        std::string warpD= parser->getGradNonlinGradWarpDim();
        if(warpD=="2D" || warpD=="2d" || warpD=="1D" || warpD=="1d")
        {
            itk::ImageRegionIterator<DisplacementFieldType> it(gradwarp_field_inv,gradwarp_field_inv->GetLargestPossibleRegion());
            it.GoToBegin();
            while(!it.IsAtEnd())
            {
                DisplacementFieldType::PixelType vec= it.Get();
                if(warpD=="2D" || warpD=="2d")
                    vec[2]=0;
                else
                {
                    vec[0]=0;
                    vec[1]=0;
                }
                it.Set(vec);
                ++it;
            }
            writeImageD<DisplacementFieldType>(gradwarp_field_inv,gradnonlin_inv_name);
        }

        typedef itk::InvertDisplacementFieldImageFilterOkan<DisplacementFieldType> InverterType;
        InverterType::Pointer inverter = InverterType::New();
        inverter->SetInput( gradwarp_field_inv );
        inverter->SetMaximumNumberOfIterations( 50 );
        inverter->SetMeanErrorToleranceThreshold( 0.0003 );
        inverter->SetMaxErrorToleranceThreshold( 0.03 );
        inverter->Update();
        DisplacementFieldType::Pointer gradwarp_field = inverter->GetOutput();
        writeImageD<DisplacementFieldType>(gradwarp_field,gradnonlin_name);
    }
}



bool TORTOISE::CheckIfInputsOkay()
{
    if(parser->getUpInputName()=="")
    {
        std::cout<<"Up data input must be entered."<<std::endl;
        return 0;
    }

    {
        fs::path up_path(parser->getUpInputName());
        if(!fs::exists(up_path))
        {
            std::cout<<"The up NIFTI file "<< parser->getUpInputName() << " does not exist."<<std::endl;
            return 0;
        }
    }

    if(parser->getUpBvalName()!="")
    {
        if(!fs::exists(parser->getUpBvalName()))
        {
            std::cout<<"The entered up bval file does not exist.."<<std::endl;
            return 0;
        }
    }
    if(parser->getUpBvecName()!="")
    {
        if(!fs::exists(parser->getUpBvecName()))
        {
            std::cout<<"The entered up bvec file does not exist.."<<std::endl;
            return 0;
        }
    }
    if(parser->getDownBvalName()!="")
    {
        if(!fs::exists(parser->getDownBvalName()))
        {
            std::cout<<"The entered down bval file does not exist.."<<std::endl;
            return 0;
        }
    }
    if(parser->getDownBvecName()!="")
    {
        if(!fs::exists(parser->getDownBvecName()))
        {
            std::cout<<"The entered down bvec file does not exist.."<<std::endl;
            return 0;
        }
    }


    if(parser->getUpBvalName()==""  && parser->getUpBvecName()=="")
    {
        std::string input_name=parser->getUpInputName();
        if(input_name.rfind(".nii")==std::string::npos)
        {
            std::cout<<"The entered up data must be a NIFTI file.."<<std::endl;
            return 0;
        }

        std::string input_basename = input_name.substr(0,input_name.rfind(".nii"));

        bool bmtxt_exists = (fs::exists(input_basename + std::string(".bmtxt"))) || (fs::exists(input_basename + std::string(".BMTXT")));
        bool bvecs_exists = (fs::exists(input_basename + std::string(".bvecs"))) || (fs::exists(input_basename + std::string(".BVECS"))) ||
                            (fs::exists(input_basename + std::string(".bvec"))) || (fs::exists(input_basename + std::string(".BVEC")));
        bool bvals_exists = (fs::exists(input_basename + std::string(".bvals"))) || (fs::exists(input_basename + std::string(".BVALS"))) ||
                            (fs::exists(input_basename + std::string(".bval"))) || (fs::exists(input_basename + std::string(".BVAL")));
        bool json_exists =  (fs::exists(input_basename + std::string(".json"))) || (fs::exists(input_basename + std::string(".JSON")));

        if(!json_exists)
        {
            std::cout<<"JSON file should exist in the same folder as the input up data file and should have the same basename..."<<std::endl;
            return 0;
        }
        if(!bmtxt_exists)
        {
            if(!bvecs_exists || !bvals_exists)
            {
                std::cout<<"Either the .bmtxt file or the .bvecs/.bvals file should be present in the same folder as the up data file and should have the same basename."<<std::endl;
                return 0;
            }
        }
    }


    if(parser->getDownBvalName()==""  && parser->getDownBvecName()=="")
    {
        std::string input_name=parser->getDownInputName();
        if(input_name!="")
        {
            if(input_name.rfind(".nii")==std::string::npos)
            {
                std::cout<<"The entered down data must be a NIFTI file.."<<std::endl;
                return 0;
            }

            std::string input_basename = input_name.substr(0,input_name.rfind(".nii"));

            bool bmtxt_exists = (fs::exists(input_basename + std::string(".bmtxt"))) || (fs::exists(input_basename + std::string(".BMTXT")));
            bool bvecs_exists = (fs::exists(input_basename + std::string(".bvecs"))) || (fs::exists(input_basename + std::string(".BVECS"))) ||
                                (fs::exists(input_basename + std::string(".bvec"))) || (fs::exists(input_basename + std::string(".BVEC")));
            bool bvals_exists = (fs::exists(input_basename + std::string(".bvals"))) || (fs::exists(input_basename + std::string(".BVALS"))) ||
                                (fs::exists(input_basename + std::string(".bval"))) || (fs::exists(input_basename + std::string(".BVAL")));
            bool json_exists =  (fs::exists(input_basename + std::string(".json"))) || (fs::exists(input_basename + std::string(".JSON")));

            if(!json_exists)
            {
                std::cout<<"JSON file should exist in the same folder as the input down data file and should have the same basename..."<<std::endl;
                return 0;
            }
            if(!bmtxt_exists)
            {
                if(!bvecs_exists || !bvals_exists)
                {
                    std::cout<<"Either the .bmtxt file or the .bvecs/.bvals file should be present in the same folder as the up data file and should have the same basename."<<std::endl;
                    return 0;
                }
            }
        }
    }

    std::vector<std::string> structural_names = parser->getStructuralNames();
    if(structural_names.size())
    {

        for(int s=0;s<structural_names.size();s++)
        {
            std::string nm = structural_names[s];
            if(!fs::exists(nm))
            {
                std::cout<<"Structural image file "<<nm << " does not exist..."<<std::endl;
                return 0;
            }
        }
    }
    if(parser->getReorientationName()!="")
    {
        std::string nm= parser->getReorientationName();
        if(!fs::exists(nm))
        {
            std::cout<<"Reorientation image file "<<nm << " does not exist..."<<std::endl;
            return 0;
        }
    }

    if(ConvertStringToStep(parser->getStartStep())== STEPS::Unknown)
    {
        std::cout<<"Start Step not valid.."<<std::endl;
        return 0;
    }

    if(parser->getGradNonlinInput()!="")
    {
        std::string nm= parser->getGradNonlinInput();
        if(!fs::exists(nm))
        {
            std::cout<<"Gradient nonlinearity file "<<nm << " does not exist..."<<std::endl;
            return 0;
        }
    }
    if(!parser->getIsHuman())
    {
        if(parser->getDTIBval()==1000 && parser->getHARDIBval()==2000)
            std::cout<<"Data is NON-human but DTI-bval and HARDI-bval not entered. Default values of DTI=1000s/mm2 and high_b=2000s/mm2 will be used";
    }

    {
        float ks_cov = parser->getGibbsKSpace();
        if(ks_cov!=0 && ks_cov!=1 && ks_cov!=0.875 && ks_cov!=0.75)
        {
            std::cout<<"K-space coverage for Gibbs ringing correction can be one of: 0(use json file) , 0.75, 0.875, 1. "<<std::endl;
            return 0;
        }
    }

    if(parser->getEPI()=="T2Wreg")
    {
        std::vector<std::string> structural_names = parser->getStructuralNames();
        if(structural_names.size()==0)
        {
            std::cout<<"EPI distortion correction by registering to a T2W structural image is selected. However, no T2W structural image provided. Skipping this processing."<<std::endl;
            parser->setEPI("off");
        }
    }
    if(parser->getEPI()=="DRBUDDI")
    {
        if(parser->getDownInputName()=="")
        {
            std::cout<<"EPI distortion correction with DRBUDDI is selected. However, no DOWN data provided. Skipping this processing."<<std::endl;
            parser->setEPI("off");
        }
    }

    if(parser->getDownInputName()!="")
    {
        if(parser->getEPI()=="T2Wreg")
        {
            std::cout<<"Down data provided but EPI distortion correction is set to T2Wreg. Changing it to DRBUDDI."<<std::endl;
            parser->setEPI("DRBUDDI");
        }
    }



    return 1;
}




#endif
