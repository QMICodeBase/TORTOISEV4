#include "estimate_tensor_wlls_parser.h"
#include <algorithm>
#include <ctype.h>






EstimateTensorWLLS_PARSER::EstimateTensorWLLS_PARSER( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);    
    this->Parse(argc,argv);
    
           
    if( argc == 1 )
    {
        this->PrintMenu( std::cout, 5, false );
        exit(EXIT_FAILURE);
    }
    
    if(checkIfAllRequiredParamsAreEntered()==0)
    {
        std::cout<<"Not all the required Parameters are entered! Exiting!"<<std::endl;
        exit(EXIT_FAILURE);
    }   
} 





EstimateTensorWLLS_PARSER::~EstimateTensorWLLS_PARSER()
{       
}




void EstimateTensorWLLS_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program estimates the diffusion tensor with different types of regression models.. " );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void EstimateTensorWLLS_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;
   
    {
        std::string description = std::string( "Full path to the input NIFTI DWIs" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Full path to the mask NIFTI image" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'm');
        option->SetLongName( "mask");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Maximum b-value volumes to use for tensor fitting. (Default: use all volumes)" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'b');
        option->SetLongName( "bval_cutoff");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Use noise image if present for weigthing and correction of interpolation artifacts. (Default:1)" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'n');
        option->SetLongName( "use_noise");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Use voxelwise Bmatrices for gradient non-linearity correction if present (Default:1)" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'v');
        option->SetLongName( "use_voxelwise_bmat");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "4D Voxelwise inclusion mask image filename" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "inclusion");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Regression mode. Default:WLLS" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "reg_mode");
        option->SetUsageOption(0,"WLLS: Weighted linear least squares");
        option->SetUsageOption(1,"NLLS: Nonlinear least squares");
        option->SetUsageOption(2,"SPD: Positive semi-definite Nonlinear least squares");
        option->SetUsageOption(3,"RESTORE: Robust NLLS");
        option->SetUsageOption(4,"DIAG: Diagonal Only NLLS");
        option->SetUsageOption(5,"N2: Full diffusion tensor + free water NLLS");
        option->SetUsageOption(6,"NT2: One full parenchymal diffusion tensor + one full flow tensor");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Free water diffusivity in (\\mu m)^2/s for N2 fitting. Default: 3000" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "free_water_diffusivity");
        option->SetShortName( 'f');
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "For NT2 fitting, what is the maximum diffusivity for the 1st compartment in  (\\mu m)^2/s . Default: 3000" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "NT2_C1_max_ADC");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "For NT2 fitting, what is the minimum diffusivity for the 2nd compartment in  (\\mu m)^2/s . Default: 9000" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "NT2_C2_min_ADC");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Write the Chi-squred image? Default:0" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "write_CS");
        option->SetDescription( description );
        this->AddOption( option );
    }

}

float EstimateTensorWLLS_PARSER::getNT2C1MaxDiff()
{
    OptionType::Pointer option = this->GetOption( "NT2_C1_max_ADC");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 3000.;
}

float EstimateTensorWLLS_PARSER::getNT2C2MinDiff()
{
    OptionType::Pointer option = this->GetOption( "NT2_C2_min_ADC");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 9000.;
}

float EstimateTensorWLLS_PARSER::getFreeWaterDiffusivity()
{
    OptionType::Pointer option = this->GetOption( "free_water_diffusivity");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 3000.;

}

bool EstimateTensorWLLS_PARSER::getWriteCSImg()
{
    OptionType::Pointer option = this->GetOption( "write_CS");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}

std::string EstimateTensorWLLS_PARSER::getRegressionMode()
{
    OptionType::Pointer option = this->GetOption( "reg_mode");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("WLLS");

}

std::string EstimateTensorWLLS_PARSER::getInclusionImg()
{
    OptionType::Pointer option = this->GetOption( "inclusion");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");

}


std::string EstimateTensorWLLS_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string EstimateTensorWLLS_PARSER::getMaskImageName()
{
    OptionType::Pointer option = this->GetOption( "mask");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

double EstimateTensorWLLS_PARSER::getBValCutoff()
{
    OptionType::Pointer option = this->GetOption( "bval_cutoff");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 1E10;
}


bool EstimateTensorWLLS_PARSER::getUseNoise()
{
    OptionType::Pointer option = this->GetOption( "use_noise");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}

bool EstimateTensorWLLS_PARSER::getUseVoxelwiseBmat()
{
    OptionType::Pointer option = this->GetOption( "use_voxelwise_bmat");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}


bool EstimateTensorWLLS_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




