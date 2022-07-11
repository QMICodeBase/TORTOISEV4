#include "estimate_mapmri_parser.h"
#include <algorithm>
#include <ctype.h>




EstimateMAPMRI_PARSER::EstimateMAPMRI_PARSER( int argc , char * argv[] )
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





EstimateMAPMRI_PARSER::~EstimateMAPMRI_PARSER()
{       
}




void EstimateMAPMRI_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program estiamtes the diffusion tensor with Weighted-Linear-Least-Squares regression.. " );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void EstimateMAPMRI_PARSER::InitializeCommandLineOptions()
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
        std::string description = std::string( "DTI image computed externally. (optional)." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "dti");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "A0 image computed externally. (optional)." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "A0");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "MAPMRI order. Optional. Default:4" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "map_order");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Small delta." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "small_delta");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Big delta." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "big_delta");
        option->SetDescription( description );
        this->AddOption( option );
    }



}


std::string EstimateMAPMRI_PARSER::getDTIImageName()
{
    OptionType::Pointer option = this->GetOption( "dti");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");

}
std::string EstimateMAPMRI_PARSER::getA0ImageName()
{
    OptionType::Pointer option = this->GetOption( "A0");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");

}
int EstimateMAPMRI_PARSER::getMAPMRIOrder()
{
    OptionType::Pointer option = this->GetOption( "map_order");
    if(option->GetNumberOfFunctions())
        return atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 4;

}

float EstimateMAPMRI_PARSER::getSmallDelta()
{
    OptionType::Pointer option = this->GetOption( "small_delta");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0;
}

float EstimateMAPMRI_PARSER::getBigDelta()
{
    OptionType::Pointer option = this->GetOption( "big_delta");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0;
}


std::string EstimateMAPMRI_PARSER::getInclusionImg()
{
    OptionType::Pointer option = this->GetOption( "inclusion");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");

}


std::string EstimateMAPMRI_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string EstimateMAPMRI_PARSER::getMaskImageName()
{
    OptionType::Pointer option = this->GetOption( "mask");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

double EstimateMAPMRI_PARSER::getBValCutoff()
{
    OptionType::Pointer option = this->GetOption( "bval_cutoff");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 1E10;
}


bool EstimateMAPMRI_PARSER::getUseNoise()
{
    OptionType::Pointer option = this->GetOption( "use_noise");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}

bool EstimateMAPMRI_PARSER::getUseVoxelwiseBmat()
{
    OptionType::Pointer option = this->GetOption( "use_voxelwise_bmat");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}


bool EstimateMAPMRI_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




