#include "TORTOISE_parser.h"
#include <algorithm>
#include <ctype.h>





TORTOISE_PARSER::TORTOISE_PARSER( int argc , char * argv[] )  :DRBUDDI_PARSERBASE(argc,argv,0)
{
    CreateParserandFillText(argc,argv);    
    this->Parse(argc,argv);
    
           
    if( argc == 1 )
    {
        std::cout<<"Simple Usage:"<<std::endl;
        std::cout<<"TORTOISEProcess -u full_path_to_your_imported_UP_nifti"<<std::endl;
        std::cout<<"OR"<<std::endl;
        std::cout<<"TORTOISEProcess -u full_path_to_your_imported_UP_nifti -d full_path_to_your_imported_DOWN_nifti   -s full_path_to_your_structural_nifti"<<std::endl;

        std::cout<<std::endl;
        std::cout<<"If you want specific settings, you can specify them from the command line or alter the settings files in TORTOISE/settings folder"<<std::endl;
        std::cout<<std::endl;

        std::vector<std::string> module_strings;
        module_strings.push_back("******************************** INPUT/OUTPUT FILES/FOLDERS  ********************************");
        module_strings.push_back("******************************** RUN SETTINGS  ********************************");
        module_strings.push_back("******************************** DENOISING SETTINGS ********************************");
        module_strings.push_back("******************************** GIBBS RINGING SETTINGS ********************************");
        module_strings.push_back("******************************** MOTION & EDDY SETTINGS ********************************");
        module_strings.push_back("******************************** SIGNAL DRIFT SETTINGS ********************************");
        module_strings.push_back("******************************** SUSCEPTIBILITY DISTORTION CORRECTION SETTINGS ********************************");
        module_strings.push_back("******************************** FINAL OUTPUT SETTINGS ********************************");


        for(int m=0;m<module_strings.size();m++)
        {
            std::cout<<module_strings[m]<<std::endl<<std::endl;
            this->PrintMenu( std::cout, 5, m,false );
            std::cout<<std::endl<<std::endl;
        }
        exit(EXIT_FAILURE);
    }
    
} 


TORTOISE_PARSER::~TORTOISE_PARSER()
{       
}


void TORTOISE_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "The main TORTOISE program. It takes in an imported NIFTI file (with corresponding bmtxt-bvecs/bvals and .json file present ). It performs entire DWI processing including denosiing, gibbs ringing correction, inter-volume and intra-volume motion, eddy-currents distortion, outlier-replacement susceptibility distortion correction and signal drift correction. It reorients the DWIs onto the desired space  defined by the provided structural image with Bmatrix reorientation." );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        

    if(argc <2)
    {
        itk::Indent indent(5);
        std::cout << std::endl;
        std::cout << "COMMAND: " << std::endl;
        std::cout << indent << argv[0] << std::endl;

        std::stringstream ss1;
        ss1 << indent << indent;

        std::stringstream ss2;
        ss2 << commandDescription ;
        std::string description = this->BreakUpStringIntoNewLines(ss2.str(), ss1.str(), 120 );
        std::cout<< indent << indent << description << std::endl;
        std::cout << std::endl;
    }
}


void TORTOISE_PARSER::InitializeCommandLineOptions()
{
  //  Superclass::InitializeCommandLineOptions();
    typedef itk::ants::CommandLineParser::OptionType OptionType;

    {
        std::string description = std::string( "Full path to the input UP bval file. (OPTIONAL. If not present, NIFTI file's folder is searched.)" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "ub");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the input UP bvec file. (OPTIONAL. If not present, NIFTI file's folder is searched.)" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "uv");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Full path to the input DOWN bval file. (OPTIONAL. If not present, NIFTI file's folder is searched.)" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "db");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the input DOWN bvec file. (OPTIONAL. If not present, NIFTI file's folder is searched.)" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "dv");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Temporary processing folder (string). If not provided, a temp folder will be created in the subfolder of the UP data." );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 't');
        option->SetLongName( "temp_folder");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the structural/anatomical image file for final reorientation. Can have any contrast. If not provided, the first image in the structural image list will be used.   " );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'r');
        option->SetLongName( "reorientation");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Flip X gradient? Boolean (0/1). Optional. Default:0  " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "flipX");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Flip Y gradient? Boolean (0/1). Optional. Default:0  " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "flipY");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Flip Z gradient? Boolean (0/1). Optional. Default:0  " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "flipZ");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }




    {
        std::string description = std::string( "The start step for the processing. Can be (increasing order): Import Denoising Gibbs MotionEddy Drift EPI StructuralAlignment  FinalData.  Default:Import " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "step");
        option->SetUsageOption(0, "import: Starts from big-bang, i.e. very beginning" );
        option->SetUsageOption(1, "denoising: Starts from denoising, skips data check and copy." );
        option->SetUsageOption(2, "gibbs: Starts from gibbs ringing correction, skips data check/copy and denoising." );
        option->SetUsageOption(3, "motioneddy: Starts from motion&eddy correction. Assumes all previous steps are done. " );
        option->SetUsageOption(4, "drift: Starts with signal drift correction. Assumes all previous steps are done.  " );
        option->SetUsageOption(5, "epi: Starts from susceptibility distortion correction. Assumes all previous steps are done.  " );
        option->SetUsageOption(6, "StructuralAlignment: Starts with aligning DWIs to the structural image. Assumes all previous steps are done.  " );
        option->SetUsageOption(7, "finaldata: Writes the final data assuming all the previous steps are already performed.  " );
        option->SetDescription( description );
        option->SetModule(1);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Perform quality control steps and generate reports? (boolean). Default: 1.  " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "do_QC");
        option->SetDescription( description );
        option->SetModule(1);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Remove the temp folder after processing? (boolean).  Default: 0 so the folder is kept." );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "remove_temp");
        option->SetDescription( description );
        option->SetModule(1);
        this->AddOption( option );
    }



    {
        std::string description = std::string("DWI denoising application (string). (J. Veraart, E. Fieremans, and D.S. Novikov Diffusion MRI noise mapping using random matrix theory. Magn. Res. Med., 2016). Default:for_reg"  );

        OptionType::Pointer option = OptionType::New();        
        option->SetLongName( "denoising");
        option->SetUsageOption(0, "off: DWI denoising not performed. However, noise variance still be estimated with this method." );
        option->SetUsageOption(1, "for_reg: DEFAULT. Denoised DWIs are used for registration but the final outputs are the transformed versions of the original data." );
        option->SetUsageOption(2, "for_final: Final output is also based on denoised DWIs." );
        option->SetDescription( description );
        option->SetModule(2);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Denoising kernel diameter (int). If 0 or not provided, the kernel diameter is automatically estimated from the data based on its size. Default:0 " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "denoising_kernel_size");
        option->SetDescription( description );
        option->SetModule(2);
        this->AddOption( option );
    }





    {
        std::string description = std::string("Gibbs ringing correction of DWIs (boolean). Kellner, Dhital, Kiselev and Resiert, MRM 2016, 76:1574-1581 (if full k-space). Lee, Novikov and Fieremans, ISMRM 2021       (if partial k-space). Default:1 ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "gibbs");
        option->SetDescription( description );
        option->SetModule(3);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Gibbs_kspace_coverage (float).  To overwrite what is read from the JSON file, in case something is wrong. Possible values: 0,  1 , 0.875 , 0.75. Default:0, which means do not use this tag ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "gibbs_kspace_coverage");
        option->SetDescription( description );
        option->SetModule(3);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Parameter for gibbs correction (int). Default:25 ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "gibbs_nsh");
        option->SetDescription( description );
        option->SetModule(3);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Parameter for gibbs correction (int). Default:1 ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "gibbs_minW");
        option->SetDescription( description );
        option->SetModule(3);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Parameter for gibbs correction (int). Default:3 ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "gibbs_maxW");
        option->SetDescription( description );
        option->SetModule(3);
        this->AddOption( option );
    }


    {
        std::string description = std::string("Among possibly many b=0 s/mm2 images, the index of the b=0 image (in terms of volume id starting from 0) to be used as template  (int). Default:-1 ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "b0_id");
        option->SetUsageOption(0, "-1:  will automatically select the best b=0 image." );
        option->SetUsageOption(1, " 0:  will use the first volume in the data." );
        option->SetUsageOption(2, " vol_id:  Will used the volume with id vol_id. It is the user's responsability to make sure this volume is a b=0 image" );
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Is it an in-vivo human brain? (boolean). Specialized processing is performed if human brain. Default:1 ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "is_human_brain");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Rotation and eddy-currents center (string). Default:isocenter ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "rot_eddy_center");
        option->SetUsageOption(0, "isocenter: the (0,0,0) coordinate from the NIFTI header (ideally the magnet isocenter) will be used as the center" );
        option->SetUsageOption(1, "center_voxel:  the very center voxel of the image will be used as the isocenter." );
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Place the center of mass of the image to the center pixel (boolean). Affects only processing not the final data." );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "center_of_mass");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Motion & eddy-currents distortions correction mode (string). Specifies which undesirable effects will be corrected.Predefined motion & eddy distortion correction optimization settings. Each setting points to a file in the softwares settings/mecc_settings folder. Default:quadratic" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'c');
        option->SetLongName( "correction_mode");
        option->SetUsageOption(0, "off: no motion or eddy currents distortion correction" );
        option->SetUsageOption(1, "motion: Corrects only motion with a rigid transformation" );
        option->SetUsageOption(2, "eddy_only: Corrects only eddy-currents distortions and no motion. Ideal for phantoms." );
        option->SetUsageOption(3, "quadratic: Motion&eddy. Eddy currents are modeled with upto quadratic Laplace bases. Quadratic model is sufficient 99% of the time." );
        option->SetUsageOption(4, "cubic: Motion&eddy.  Eddy currents are modeled with upto-including cubic Laplace bases" );
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Slice to volume or in other-words slice-by-slice correction (boolean). Significantly increases processing time but no other penalties in data quality. Default:0  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "s2v");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Outlier detection and replacement (boolean). Replaces the automatically detected outlier slices with estimates from the MAPMRI model.Significantly increases processing time but no other penalties in data quality. Default:0  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "repol");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Outlier fraction ratio. If percentace of outlier slices is larger than this threshold, the entire volume is considered troublesome and all the values replaced with the predicted ones. Default:0.5  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "outlier_frac");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Outlier probability threshold. If the probability of a slice RMS is lower than this value, that slice is labeled artifactual. Default:0.025  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "outlier_prob");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Number of RMS clusters for EM outlier detection. Ideally there should be 2 clusters: inliers, outliers. However, life is not perfect. Clusters will afterwards be combined till they reach 75\% inclusion.  Default:4  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "outlier_EM_clusters");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("String to determine whether to label more voxels as outliers or less. Default:middle  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "outlier_replacement_mode");        
        option->SetUsageOption(0, "conservative: less voxels labeled as outliers" );
        option->SetUsageOption(1, "middle: somewhere between conservative and aggressive" );
        option->SetUsageOption(2, "aggressive: more voxels labeled as outliers" );
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }


    {
        std::string description = std::string("Number of iterations for high_bvalue / s2v / repol correction. Has no effect for dti regime data with s2v and repol disabled. Default:4 . Zero disables all iterative correction such as high-b, s2v or repol. ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "niter");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("DTI bval (int). In case non in-vivo human brain data, or to overwrite the default value of 1000 s/mm2, what is the bval for DTI regime? Default:1000  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "dti_bval");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }
    {
        std::string description = std::string("HARDI bval (int). In case non in-vivo human brain data, or to overwrite the default value of 2000 s/mm2, what is the bval for higher order regime? Default:2000  ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "hardi_bval");
        option->SetDescription( description );
        option->SetModule(4);
        this->AddOption( option );
    }

    {
        std::string description = std::string("Signal drift correction method. Data will be checked to automatically determine if this correction can be applied. ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "drift");
        option->SetUsageOption(0, "off: No signal drift correction. Default." );
        option->SetUsageOption(1, "linear: Linear signal drift over time" );
        option->SetUsageOption(2, "quadratic:  Quadratic signal drift over time." );
        option->SetDescription( description );
        option->SetModule(5);
        this->AddOption( option );
    }




    {
        std::string description = std::string("EPI Distortion correction method. ")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "epi");
        option->SetUsageOption(0, "off: no EPI distortion correction performed." );
        option->SetUsageOption(1, "T2Wreg: Diffeomorphically register the b=0 image to the provided T2W structural image. " );
        option->SetUsageOption(2, "DRBUDDI:  Perform blip-up blip-down correction with DRBUDDI." );
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }


    {
        std::string description = std::string("Output name of the final NIFTI file" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output");
        option->SetDescription( description );
        option->SetModule(7);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Output orientation of the data. 3 characters, eg: LPS, RAI, ILA. First letter for the anatomical direction that is from left of the image TOWARDS right (The letter if for the ending point     of the direction not the beginning). Second letter from the top of the image to bottom. Third letter from the first slice to the last slice. Default: LPS. " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "output_orientation");
        option->SetDescription( description );
        option->SetModule(7);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Resolution of the final output: res_x res_y res_z separated by space. Default: the original resolution  " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "output_res");
        option->SetDescription( description );
        option->SetModule(7);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Number of voxels in the final image: Nx Ny Nz separated by space. Image might be padded/cropped to match this. Default: computed from structural's FoV and the output resolution. " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "output_voxels");
        option->SetDescription( description );
        option->SetModule(7);
        this->AddOption( option );
    }

    {
        std::string description = std::string("Output data combination method:    " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "output_data_combination");
        option->SetUsageOption(0, "Merge: If up and down data have the same Bmatrix, the corresponding DWIs are geometrically averaged. Default if this is the case." );
        option->SetUsageOption(1, "JacConcat: Up and down DWIs' signals are manipulated by the Jacobian. The two data are concatenated into a single one. Default if upBmtxt != downBmtxt" );
        option->SetUsageOption(2, "JacSep: Up and down data are Jacobian manipulated and saved separately. The \"output\" tag has no effect. ");
        option->SetDescription( description );
        option->SetModule(7);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Format of the gradient nonlinearity output information. If vbmat is selected, s2v effects will also be considered even if no gradient nonlinearity information is present.   " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "output_gradnonlin_Bmtxt_type");
        option->SetUsageOption(0, "grad_dev: A single gradient deviation tensor image is written in HCP style to be applied to ALL volumes." );
        option->SetUsageOption(1, "vbmat: A voxelwise Bmatrix image is written that also includes the effect of motion. LARGE SIZE. Default." );
        option->SetDescription( description );
        option->SetModule(7);
        this->AddOption( option );
    }




}

std::string TORTOISE_PARSER::getUpBvalName()
{
    OptionType::Pointer option = this->GetOption( "ub");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}
std::string TORTOISE_PARSER::getUpBvecName()
{
    OptionType::Pointer option = this->GetOption( "uv");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}



std::string TORTOISE_PARSER::getDownBvalName()
{
    OptionType::Pointer option = this->GetOption( "db");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}
std::string TORTOISE_PARSER::getDownBvecName()
{
    OptionType::Pointer option = this->GetOption( "dv");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}



std::string TORTOISE_PARSER::getTempProcFolder()
{
    OptionType::Pointer option = this->GetOption( "temp_folder");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}



std::string TORTOISE_PARSER::getReorientationName()
{
    OptionType::Pointer option = this->GetOption( "reorientation");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}







int TORTOISE_PARSER::getFlipX()
{
    OptionType::Pointer option = this->GetOption( "flipX");
    if(option->GetNumberOfFunctions())
    {
        bool flip= (bool)( atoi(option->GetFunction(0)->GetName().c_str()));
        if(flip)
            return -1;
    }
    return 1;
}
int TORTOISE_PARSER::getFlipY()
{
    OptionType::Pointer option = this->GetOption( "flipY");
    if(option->GetNumberOfFunctions())
    {
        bool flip= (bool)( atoi(option->GetFunction(0)->GetName().c_str()));
        if(flip)
            return -1;
    }
    return 1;
}
int TORTOISE_PARSER::getFlipZ()
{
    OptionType::Pointer option = this->GetOption( "flipZ");
    if(option->GetNumberOfFunctions())
    {
        bool flip= (bool)( atoi(option->GetFunction(0)->GetName().c_str()));
        if(flip)
            return -1;
    }
    return 1;
}




std::string TORTOISE_PARSER::getStartStep()
{
    OptionType::Pointer option = this->GetOption( "step");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("import");
}
bool TORTOISE_PARSER::getDoQC()
{
    OptionType::Pointer option = this->GetOption( "do_QC");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}
bool TORTOISE_PARSER::getRemoveTempFolder()
{
    OptionType::Pointer option = this->GetOption( "remove_temp");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}



std::string TORTOISE_PARSER::getDenoising()
{
    OptionType::Pointer option = this->GetOption( "denoising");
    if(option->GetNumberOfFunctions())
        return  option->GetFunction(0)->GetName();
    else
       return std::string("for_reg");
}
int TORTOISE_PARSER::getDenoisingKernelSize()
{
    OptionType::Pointer option = this->GetOption( "denoising_kernel_size");
    if(option->GetNumberOfFunctions())
        return  (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}



bool TORTOISE_PARSER::getGibbs()
{
    OptionType::Pointer option = this->GetOption( "gibbs");
    if(option->GetNumberOfFunctions())
        return  (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}
float TORTOISE_PARSER::getGibbsKSpace()
{
    OptionType::Pointer option = this->GetOption( "gibbs_kspace_coverage");
    if(option->GetNumberOfFunctions())
        return  atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.;
}
int TORTOISE_PARSER::getGibbsNsh()
{
    OptionType::Pointer option = this->GetOption( "gibbs_nsh");
    if(option->GetNumberOfFunctions())
        return  atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 25;
}
int TORTOISE_PARSER::getGibbsMinW()
{
    OptionType::Pointer option = this->GetOption( "gibbs_minW");
    if(option->GetNumberOfFunctions())
        return  atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 1;
}
int TORTOISE_PARSER::getGibbsMaxW()
{
    OptionType::Pointer option = this->GetOption( "gibbs_maxW");
    if(option->GetNumberOfFunctions())
        return  atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 3;
}






int TORTOISE_PARSER::getB0Id()
{
    OptionType::Pointer option = this->GetOption( "b0_id");
    if(option->GetNumberOfFunctions())
        return  atoi(option->GetFunction(0)->GetName().c_str());
    else
       return -1;
}
bool TORTOISE_PARSER::getIsHuman()
{
    OptionType::Pointer option = this->GetOption( "is_human_brain");
    if(option->GetNumberOfFunctions())
        return  (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}
std::string TORTOISE_PARSER::getRotEddyCenter()
{
    OptionType::Pointer option = this->GetOption( "rot_eddy_center");
    if(option->GetNumberOfFunctions())
        return  option->GetFunction(0)->GetName();
    else
       return std::string("isocenter");
}
bool TORTOISE_PARSER::getCenterOfMass()
{
    OptionType::Pointer option = this->GetOption( "center_of_mass");
    if(option->GetNumberOfFunctions())
        return  (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}
std::string TORTOISE_PARSER::getCorrectionMode()
{
    OptionType::Pointer option = this->GetOption( "correction_mode");
    if(option->GetNumberOfFunctions())
        return  option->GetFunction(0)->GetName();
    else
       return std::string("quadratic");
}
bool TORTOISE_PARSER::getS2V()
{
    OptionType::Pointer option = this->GetOption( "s2v");
    if(option->GetNumberOfFunctions())
        return  (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}
bool TORTOISE_PARSER::getRepol()
{
    OptionType::Pointer option = this->GetOption( "repol");
    if(option->GetNumberOfFunctions())
        return  (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}
float TORTOISE_PARSER::getOutlierFrac()
{
    OptionType::Pointer option = this->GetOption( "outlier_frac");
    if(option->GetNumberOfFunctions())
        return  atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.5;
}


float TORTOISE_PARSER::getOutlierProbabilityThreshold()
{
    OptionType::Pointer option = this->GetOption( "outlier_prob");
    if(option->GetNumberOfFunctions())
        return  atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.025;
}




int TORTOISE_PARSER::getOutlierNumberOfResidualClusters()
{
    OptionType::Pointer option = this->GetOption( "outlier_EM_clusters");
    if(option->GetNumberOfFunctions())
        return  atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 4;
}


int TORTOISE_PARSER::getOutlierReplacementModeAggessive()
{
    OptionType::Pointer option = this->GetOption( "outlier_replacement_mode");
    if(option->GetNumberOfFunctions())
    {
        if(option->GetFunction(0)->GetName()=="conservative")
        {
            return 0;
        }
        if(option->GetFunction(0)->GetName()=="middle")
        {
             return 1;
        }
        if(option->GetFunction(0)->GetName()=="aggressive")
        {
            return 2;
        }
    }


    return 1;
}


int TORTOISE_PARSER::getNiter()
{
    OptionType::Pointer option = this->GetOption( "niter");
    if(option->GetNumberOfFunctions())
        return  (atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 3;
}
int TORTOISE_PARSER::getDTIBval()
{
    OptionType::Pointer option = this->GetOption( "dti_bval");
    if(option->GetNumberOfFunctions())
        return  (atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1000;
}
int TORTOISE_PARSER::getHARDIBval()
{
    OptionType::Pointer option = this->GetOption( "hardi_bval");
    if(option->GetNumberOfFunctions())
        return  (atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 2000;
}



std::string TORTOISE_PARSER::getDrift()
{
    OptionType::Pointer option = this->GetOption( "drift");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("off");
}



std::string TORTOISE_PARSER::getEPI()
{
    OptionType::Pointer option = this->GetOption( "epi");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("DRBUDDI");
}

void TORTOISE_PARSER::setEPI(std::string nepi)
{
    OptionType::Pointer option = this->GetOption( "epi");
    option->m_OptionFunctions.clear();
    option->AddFunction(nepi);
}



std::string TORTOISE_PARSER::getOutputName()
{
    OptionType::Pointer option = this->GetOption( "output");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}
std::string TORTOISE_PARSER::getOutputOrientation()
{
    OptionType::Pointer option = this->GetOption( "output_orientation");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");

}
std::vector<int> TORTOISE_PARSER::GetOutputNVoxels()
{
    OptionType::Pointer option = this->GetOption( "output_voxels");
    if(option->GetNumberOfFunctions())
    {
        if(option->GetNumberOfFunctions()!=3)
        {
            std::cout<<"3 numbers have to be entered for the final matrix size... Exiting..."<<std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<int> nvoxels;
        nvoxels.resize(3);
        nvoxels[0]= atoi(option->GetFunction(2)->GetName().c_str());
        nvoxels[1]= atoi(option->GetFunction(1)->GetName().c_str());
        nvoxels[2]= atoi(option->GetFunction(0)->GetName().c_str());
        return nvoxels;
    }
    else
       return std::vector<int>();

}
std::vector<float> TORTOISE_PARSER::GetOutputRes()
{
    OptionType::Pointer option = this->GetOption( "output_res");
    if(option->GetNumberOfFunctions())
    {
        if(option->GetNumberOfFunctions()!=3)
        {
            std::cout<<"3 numbers have to be entered for final resolution... Exiting..."<<std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<float> FoV;
        FoV.resize(3);
        FoV[0]= atof(option->GetFunction(2)->GetName().c_str());
        FoV[1]= atof(option->GetFunction(1)->GetName().c_str());
        FoV[2]= atof(option->GetFunction(0)->GetName().c_str());
        return FoV;
    }
    else
       return std::vector<float>();
}

std::string TORTOISE_PARSER::getOutputDataCombination()
{
    OptionType::Pointer option = this->GetOption( "output_data_combination");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("Merge");

}

std::string TORTOISE_PARSER::getOutputGradientNonlinearityType()
{
    OptionType::Pointer option = this->GetOption( "output_gradnonlin_Bmtxt_type");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("grad_dev");
}
