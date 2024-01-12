#include "CreateDummyJson_parser.h"
#include <algorithm>
#include <ctype.h>




std::string getInputImageName();
std::string getPhaseEncoding();
int getMBFactor();


CreateDummyJson_PARSER::CreateDummyJson_PARSER( int argc , char * argv[] )
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





CreateDummyJson_PARSER::~CreateDummyJson_PARSER()
{       
}




void CreateDummyJson_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program estiamtes the diffusion tensor with Weighted-Linear-Least-Squares regression.. " );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void CreateDummyJson_PARSER::InitializeCommandLineOptions()
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
        std::string description = std::string( "Phase encoding direction. Options: i+ (for RL), i- (for LR), j+ (for AP), j- (for PA)" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'p');
        option->SetLongName( "phase");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Multi band factor" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'm');
        option->SetLongName( "MBf");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Partial Fourier. Options: 1, 0.875 , 0.75" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'f');
        option->SetLongName( "partial_fourier");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Big delta. Diffusion separation time." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "big_delta");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Small delta. Diffusion  time." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "small_delta");
        option->SetDescription( description );
        this->AddOption( option );
    }
}


float CreateDummyJson_PARSER::getBigDelta()
{
    OptionType::Pointer option = this->GetOption( "big_delta");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return -1;
}

float CreateDummyJson_PARSER::getSmallDelta()
{
    OptionType::Pointer option = this->GetOption( "small_delta");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return -1;
}


std::string CreateDummyJson_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

int CreateDummyJson_PARSER::getMBFactor()
{
    OptionType::Pointer option = this->GetOption( "MBf");
    if(option->GetNumberOfFunctions())
        return atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 1;
}

std::string CreateDummyJson_PARSER::getPhaseEncoding()
{
    OptionType::Pointer option = this->GetOption( "phase");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

float CreateDummyJson_PARSER::getPF()
{
    OptionType::Pointer option = this->GetOption( "partial_fourier");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 1;
}


bool CreateDummyJson_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    if(this->getPhaseEncoding()=="")
    {
        std::cout<<"Phase encoding direction not entered...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




