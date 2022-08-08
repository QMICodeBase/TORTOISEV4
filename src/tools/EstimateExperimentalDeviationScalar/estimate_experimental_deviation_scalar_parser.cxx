#include "estimate_experimental_deviation_scalar_parser.h"
#include <algorithm>
#include <ctype.h>



EstimateExperimentalDeviationScalar_PARSER::EstimateExperimentalDeviationScalar_PARSER( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);    
    this->Parse(argc,argv);
    
           
    if( argc == 1 )
    {
        this->PrintMenu( std::cout, 5, false );
        exit(EXIT_SUCCESS);
    }
    
    if(checkIfAllRequiredParamsAreEntered()==0)
    {
        std::cout<<"Not all the required Parameters are entered! Exiting!"<<std::endl;
        exit(EXIT_FAILURE);
    }   
} 



EstimateExperimentalDeviationScalar_PARSER::~EstimateExperimentalDeviationScalar_PARSER()
{       
}


void EstimateExperimentalDeviationScalar_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program estimates the experimental variance of a tensor-derived scalar " );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void EstimateExperimentalDeviationScalar_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;
   
    {
        std::string description = std::string( "Full path to the input nifti file" );
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
        std::string description = std::string( "Scalar modality. FA or TR." );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'd');
        option->SetLongName( "modality");
        option->SetDescription( description );
        this->AddOption( option );
    }

}





std::string EstimateExperimentalDeviationScalar_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string EstimateExperimentalDeviationScalar_PARSER::getMaskImageName()
{
    OptionType::Pointer option = this->GetOption( "mask");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


double EstimateExperimentalDeviationScalar_PARSER::getBValCutoff()
{
    OptionType::Pointer option = this->GetOption( "bval_cutoff");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 1E10;
}


std::string EstimateExperimentalDeviationScalar_PARSER::getModality()
{
    OptionType::Pointer option = this->GetOption( "modality");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("FA");
}



bool EstimateExperimentalDeviationScalar_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




