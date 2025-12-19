#include "DRBUDDI_parserBase.h"
#include <algorithm>
#include <ctype.h>





DRBUDDI_PARSERBASE::DRBUDDI_PARSERBASE( int argc , char * argv[] , bool print=0)
{
    CreateParserandFillText(argc,argv);    

    if(print)
        this->Parse(argc,argv);
    
           
    if( argc == 1 && print)
    {
        std::cout<<"Simple Usage:"<<std::endl;
        std::cout<<"DRBUDDI -u full_path_to_your_imported_UP_nifti -d full_path_to_your_imported_DOWN_nifti   -s full_path_to_your_structural_nifti"<<std::endl;


        std::vector<std::string> module_strings;
        module_strings.push_back("");
        module_strings.push_back("");
        module_strings.push_back("");
        module_strings.push_back("");
        module_strings.push_back("");
        module_strings.push_back("");
        module_strings.push_back("******************************** SUSCEPTIBILITY DISTORTION CORRECTION SETTINGS ********************************");



        for(int m=0;m<module_strings.size();m++)
        {
            std::cout<<module_strings[m]<<std::endl<<std::endl;
            this->PrintMenu( std::cout, 5, m,false );
            std::cout<<std::endl<<std::endl;
        }
        exit(EXIT_FAILURE);
    }
    
} 


DRBUDDI_PARSERBASE::~DRBUDDI_PARSERBASE()
{       
}


void DRBUDDI_PARSERBASE::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "The main DRBUDDI program. It takes in imported up and down DWI NIFTI files (with corresponding bmtxt and .json file present ) and anatomical structural NIFTI files. It performs susceptibility distortion correction" );
    
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


void DRBUDDI_PARSERBASE::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;
   
    {
        std::string description = std::string( "Full path to the input UP NIFTI file to be corrected. (REQUIRED. The only required parameteter.)" );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'u');
        option->SetLongName( "up_data");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the JSON file for up or down data. REQUIRED for DRBUDDI, optional for TORTOISEProcess. Phase encoding information will be read from this.)" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "up_json");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the input DOWN NIFTI file to be corrected." );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'd');
        option->SetLongName( "down_data");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Full path to the structural/anatomical image files. Can provide more than one. These will be used for EPI distortion correction. SO NO T1W images here.    " );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 's');
        option->SetLongName( "structural");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Gradient Nonlinearity information file. Can be in ITK displacement field format, TORTOISE coefficients .gc format, GE coefficients gw_coils format or Siemens coefficients .coeffs format. If it is GE, it should be specified in brackets. If 1D or 2D gradwarp is desired, it should be specified. Default:3D   " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "grad_nonlin");
        option->SetDescription( description );
        option->SetUsageOption( 0, "gradnonlin_file" );
        option->SetUsageOption( 1, "gradnonlin_file\[is_GE,warp_dim\]" );
        option->SetUsageOption( 2, "example1: coeffs.grad" );
        option->SetUsageOption( 2, "example2: coeffs.grad\[0,3D\]" );
        option->SetUsageOption( 3, "example3: field.nii\[1,2D\]" );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Disable Gradwarp field correction.  Sometimes DWIs are NOT gradwarped at the scanner but sometimes they are. In case they are, the user might want to turn off gradwarp to prevent double correction but just want a voxelwise Bmatrix at the end. Boolean. Default: 0" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "NO_gradwarp");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }



    {
        std::string description = std::string( "DRBUDDI transformation output folder." );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_output");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "DRBUDDI start step. 0: beginning. Creates b=0 and FA images. 1: Assumes b=0 and FA images are present with correct name and starts with rigid registration. 2: Assumes all images and rigid transformations are present and starts with diffeomorphic distortion correction. " );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_step");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }


    {
        std::string description = std::string("Initial transform field for the up data.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_initial_fixed_transform");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Initial transform field for the down data.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_initial_moving_transform");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("DRBUDDI performs an initial registration between the up and down data. This registration starts with rigid, followed by a quick diffeomorphic and finalized by another rigid. This parameter, when set to 1 disables all these registrations. Default: 0")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_disable_initial_rigid");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("DRBUDDI performs an initial registration between the up and down data. This registration starts with rigid, followed by a quick diffeomorphic and finalized by another rigid. This parameter, when set to 1 disables the very initial rigid registration and starts with the quick diffemorphic. This is helpful with VERY DISTORTED data, for which the initial rigid registration is problematic. Default: 0")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_start_with_diffeomorphic_for_rigid_reg");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Similarity Metric to be used in rigid registration. Options: MI or CC. Default: MI")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_rigid_metric_type");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Rigid metric learning rate: Default:0.25")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_rigid_learning_rate");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Up to which b-value should be used for DRBUDDI's tensor fitting. Default: 0 , meaning use all b-values")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_DWI_bval_tensor_fitting");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Flag to estimate learning rate at every iteration. Makes DRBUDDI slower but better results. Boolean. Default:0")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_estimate_LR_per_iteration");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string( "DRBUDDI runs many registration stages during correction. This tag sets all the parameters for a given stage. Each stage is executed in the order provided on the command line. Available metrics are:  MSJac, CC, CCSK. MSJac uses the b=0 images. CC uses FA images.  CCSK uses b=0 and the structural images. Which structural image to be used with CCSK is given with an index as:  CCSK\{str_id=1\}." ) ;

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_stage" );
        option->SetUsageOption(0, "\[learning_rate=\{learning_rate\},cfs=\{Niterations:downsampling_factor:image_smoothing_stdev\},field_smoothing=\{update_field_smoothing_stdev:total_field_smoothing_stdev\},metrics=\{metric1:metric2:...metricn\},restrict_constrain=\{restrict_to_phaseencoding:enforce_up_down_deformation_symmetry\}\]" );
        option->SetUsageOption(1, "\[learning_rate=\{0.5\},cfs=\{100:1:0\},field_smoothing=\{3.:0.1\},metrics=\{MSJac:CC:CCSK\{str_id=0\}:CCSK\{str_id=1\}\},restrict_constrain=\{1:1\}\]" );
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Flag to enforce DRBUDDI to enforce  blip-up blip-down antisymmetry and phasen-encoding restriction when using the default settings. Boolean. Default:0")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "enforce_full_symmetry");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Flag to enforce DRBUDDI to enforce  blip-up blip-down antisymmetry and phasen-encoding restriction when using the default settings. Boolean. Default:0")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_disable_last_stage");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Multiplicative factor for metrics that use the structural image. Might want to reduce it if the structural's contrast is significantly different than the b=0. Float. Default:1")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRBUDDI_structural_weight");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }


    {
        std::string description = std::string("The last DRBUDDI stage heavily favors the structural image. If this image is not ideal (not a good contrast), it might be more robust to disable this stage. Boolean. Default:0")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "disable_itk_threads");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Registration transformation type. Options: SyN or TVVF. Default: SyN.  TVVF only works in CUDA version.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "transformation_type");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Number of cores to use in the CPU version of DRBUDDI. ONLY applies to the DRBUDDI executable and not TORTOISEProcess. The default is 50\% of system cores.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "ncores");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
}


bool DRBUDDI_PARSERBASE::getNOGradWarp()
{
    OptionType::Pointer option = this->GetOption( "NO_gradwarp");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return false;
}

float DRBUDDI_PARSERBASE::getStructuralWeight()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_structural_weight");
    if(option->GetNumberOfFunctions())
        return (atof(option->GetFunction(0)->GetName().c_str()));
    else
       return 1.;
}

bool DRBUDDI_PARSERBASE::getDisableLastStage()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_disable_last_stage");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return false;
}

bool DRBUDDI_PARSERBASE::getEnforceFullAntiSymmetry()
{
    OptionType::Pointer option = this->GetOption( "enforce_full_symmetry");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return false;
}

int DRBUDDI_PARSERBASE::getNumberOfCores()
{
    OptionType::Pointer option = this->GetOption( "ncores");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}

bool DRBUDDI_PARSERBASE::getDisableITKThreads()
{
    OptionType::Pointer option = this->GetOption( "disable_itk_threads");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}

std::string DRBUDDI_PARSERBASE::getUpInputName()
{
    OptionType::Pointer option = this->GetOption( "up_data");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string DRBUDDI_PARSERBASE::getUpJSonName()
{
    OptionType::Pointer option = this->GetOption( "up_json");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string DRBUDDI_PARSERBASE::getDownInputName()
{
    OptionType::Pointer option = this->GetOption( "down_data");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::vector<std::string> DRBUDDI_PARSERBASE::getStructuralNames()
{
    std::vector<std::string> names;

    OptionType::Pointer option = this->GetOption( "structural");
    int nstr= option->GetNumberOfFunctions();
    if(nstr>0)
    {
        for(int str_id=0;str_id<nstr;str_id++)
        {
            std::string nm= option->GetFunction(nstr-1-str_id)->GetName();
            names.push_back(nm);
        }
    }
    return names;
}

std::string DRBUDDI_PARSERBASE::getDRBUDDIOutput()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_output");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


int DRBUDDI_PARSERBASE::getDRBUDDIStep()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_step");
    if(option->GetNumberOfFunctions())
        return atoi(option->GetFunction(0)->GetName().c_str());

    return 0;
}

std::string DRBUDDI_PARSERBASE::getGradNonlinInput()
{
    OptionType::Pointer option = this->GetOption( "grad_nonlin");
    if(option->GetNumberOfFunctions())
    {
        std::string nm = option->GetFunction(0)->GetName();
        std::string ext= nm.substr(nm.rfind("."));
        if(ext!=".grad" && ext!=".dat" && ext!=".gc")
        {
            std::cout<<"WARNING! Gradient nonlinearity file format not recognized. Check the file extension..."<<std::endl;
            std::cout<<"Disabling gradient nonlinearity based processing..."<<std::endl;
            return "";
        }
        return option->GetFunction(0)->GetName();
    }
    else
       return std::string("");
}
/*
void DRBUDDI_PARSERBASE::setGradNonlinInput(std::string fname)
{
    OptionType::Pointer option = this->GetOption( "grad_nonlin");
    option->m_OptionFunctions.clear();
    option->AddFunction(fname);
}
*/
bool DRBUDDI_PARSERBASE::getGradNonlinIsGE()
{
    OptionType::Pointer option = this->GetOption( "grad_nonlin");
    if(option->GetNumberOfFunctions())
    {
        int Nparams= option->GetFunction(0)->GetNumberOfParameters();
        if(Nparams>0)
        {
            bool isGE= (bool)(atoi(option->GetFunction( 0 )->GetParameter( 0 ).c_str())) ;
            return isGE;
        }
        else
            return false;
    }
    else
        return false;
}

std::string DRBUDDI_PARSERBASE::getGradNonlinGradWarpDim()
{
    OptionType::Pointer option = this->GetOption( "grad_nonlin");
    if(option->GetNumberOfFunctions())
    {
        int Nparams= option->GetFunction(0)->GetNumberOfParameters();
        if(Nparams>1)
        {
            std::string warpD= option->GetFunction( 0 )->GetParameter( 1 ) ;
            return warpD;
        }
        else
            return "3D";
    }
    else
        return "3D";
}



std::string DRBUDDI_PARSERBASE::GetInitialFINV()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_initial_fixed_transform");
   if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
   else
       return std::string("");

}
std::string DRBUDDI_PARSERBASE::GetInitialMINV()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_initial_moving_transform");
   if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
   else
       return std::string("");

}

bool DRBUDDI_PARSERBASE::getDisableInitRigid()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_disable_initial_rigid");

    if(option->GetNumberOfFunctions())
         return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 0;
}
bool DRBUDDI_PARSERBASE::getStartWithDiffeo()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_start_with_diffeomorphic_for_rigid_reg");

    if(option->GetNumberOfFunctions())
         return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 0;
}
bool DRBUDDI_PARSERBASE::getEstimateLRPerIteration()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_estimate_LR_per_iteration");

    if(option->GetNumberOfFunctions())
         return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
#ifdef USECUDA
        return 0;
#else
        return 0;
#endif
}

std::string  DRBUDDI_PARSERBASE::getRigidMetricType()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_rigid_metric_type");
    if(option->GetNumberOfFunctions())
         return option->GetFunction(0)->GetName();
    else
        return std::string("CC");
}
float  DRBUDDI_PARSERBASE::getRigidLR()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_rigid_learning_rate");
    if(option->GetNumberOfFunctions())
         return atof(option->GetFunction(0)->GetName().c_str());
    else
        return 0.35;

}
int  DRBUDDI_PARSERBASE::getDWIBvalue()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_DWI_bval_tensor_fitting");
    if(option->GetNumberOfFunctions())
         return atoi(option->GetFunction(0)->GetName().c_str());
    else
        return 0;
}


int DRBUDDI_PARSERBASE::getNumberOfStages()
{
    OptionType::Pointer option = this->GetOption( "DRBUDDI_stage");
    int nstg= option->GetNumberOfFunctions();
    return nstg;
}


std::vector<std::string>  DRBUDDI_PARSERBASE::getStageString(int st)
{
    int nstg= this->getNumberOfStages();

    std::vector<std::string> params;

    OptionType::Pointer option = this->GetOption( "DRBUDDI_stage");
    if(option->GetNumberOfFunctions())
        if(st <option->GetNumberOfFunctions())
        {
            for(int p=0;p<option->GetFunction( nstg-1-st )->GetNumberOfParameters();p++)
            {
                std::string param=( option->GetFunction( nstg-1-st )->GetParameter( p ) );
                params.push_back(param);
            }

            if(params.size()!=5)
            {
                std::cout<<"Incorrect format for stage parameters. There should be 5 groups of parameters. Exiting!!"<<std::endl;
                exit(EXIT_FAILURE);
            }
        }
    return params;
}



int DRBUDDI_PARSERBASE::GetNIter(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("cfs=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(5,pos-5);

            std::string val_string = param_string.substr(0,param_string.find(":"));
            return atoi(val_string.c_str());
        }
    }
    return 100;
}


int DRBUDDI_PARSERBASE::GetF(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("cfs=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(5,pos-5);

            int pos1= param_string.find(":");
            int pos2= param_string.rfind(":");

            std::string val_string = param_string.substr(pos1+1,pos2-pos1-1);
            return atoi(val_string.c_str());
        }
    }
    return 1;
}


float DRBUDDI_PARSERBASE::GetS(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("cfs=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(5,pos-5);

            int pos2= param_string.rfind(":");

            std::string val_string = param_string.substr(pos2+1);
            return atof(val_string.c_str());
        }
    }
    return 0;
}


float DRBUDDI_PARSERBASE::GetLR(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("learning_rate=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(15,pos-15);

            return atof(param_string.c_str());
        }
    }
    return 0.5;
}


float DRBUDDI_PARSERBASE::GetUStd(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("field_smoothing=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(17,pos-17);

            pos= param_string.rfind(":");

            std::string val_string = param_string.substr(0,pos);
            return atof(val_string.c_str());
        }
    }
    return 3;
}

float DRBUDDI_PARSERBASE::GetTStd(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("field_smoothing=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(17,pos-17);

            pos= param_string.rfind(":");

            std::string val_string = param_string.substr(pos+1);
            return atof(val_string.c_str());
        }
    }
    return 0;
}

bool DRBUDDI_PARSERBASE::GetRestrict(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("restrict_constrain=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(20,pos-20);

            pos= param_string.rfind(":");

            std::string val_string = param_string.substr(0,pos);
            return (bool)(atoi(val_string.c_str()));
        }
    }
    return 0;
}

bool DRBUDDI_PARSERBASE::GetConstrain(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("restrict_constrain=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(20,pos-20);

            pos= param_string.rfind(":");

            std::string val_string = param_string.substr(pos+1);
            return (bool)(atoi(val_string.c_str()));
        }
    }
    return 0;
}

int DRBUDDI_PARSERBASE::GetNMetrics(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("metrics=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(9,pos-9);

            int n = std::count(param_string.begin(), param_string.end(), ':');
            return n+1;
        }
    }
    return 0;
}

std::string DRBUDDI_PARSERBASE::GetMetricString(int st,int m)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("metrics=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(9,pos-9);

            for(int m2=0;m2<m;m2++)
            {
                param_string=param_string.substr(param_string.find(":")+1);
            }

            if(param_string.find(":")==std::string::npos)
            {
                std::string val = param_string.substr(0,param_string.rfind("}"));
                return val;
            }
            else
            {
                std::string val = param_string.substr(0,param_string.find(":"));
                return val;
            }

        }
    }
    return "";
}


int DRBUDDI_PARSERBASE::getNumberOfStructurals()
{

    OptionType::Pointer option = this->GetOption( "structural");
    int nstr= option->GetNumberOfFunctions();

    return nstr;
}

std::string DRBUDDI_PARSERBASE::getStructuralNames(int str_id=0)
{
    int nstr= this->getNumberOfStructurals();

   OptionType::Pointer option = this->GetOption( "structural");
   if(option->GetNumberOfFunctions())
       if(str_id <option->GetNumberOfFunctions())
          return option->GetFunction(nstr-1-str_id)->GetName();
       else
           return option->GetFunction(0)->GetName();

   else
       return std::string("");
}



std::string DRBUDDI_PARSERBASE::getRegistrationMethodType()
{
    #ifdef USECUDA
        OptionType::Pointer option = this->GetOption( "transformation_type");
        if(option->GetNumberOfFunctions())
             return option->GetFunction(0)->GetName();
        else
            return "SyN";
    #else
        return "SyN";
    #endif
}
