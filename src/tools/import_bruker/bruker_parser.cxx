#include "bruker_parser.h"
#include <algorithm>
#include <ctype.h>



Bruker_PARSER::Bruker_PARSER( int argc , char * argv[] )
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


Bruker_PARSER::~Bruker_PARSER()
{       
}


void Bruker_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program imports  " )
      + std::string( "diffusion MRI data in Bruker scanner's conventional format into a format compatible with TORTOISE. " )
      + std::string( "Only the diffusion weighted scans are imported and all the diffusion weighted series are combined into a single dataset. " );   
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void Bruker_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;

   
    {
        std::string description = std::string( "Full path to the input data folder." );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "import_folder");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Full path to the output PROC folder. (optional)" );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output_folder");
        option->SetDescription( description );
        this->AddOption( option );
    }


    /*
    {
        std::string description = std::string( "Binary value to indicate whether to compute the Bmatrix from the bvalues and gradients. (optional)" );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'g');
        option->SetLongName( "gradients");
        option->SetDescription( description );
        option->SetUsageOption( 0, "0 (Default. Use the Bmatrix from the Bruker method file)" );
        option->SetUsageOption( 1, "1 (Compute the Bmatrix from bvalues and gradients)" );
        option->AddFunction( std::string( "0"));
        this->AddOption( option );
    }
    */

    {
        std::string description = std::string( "Convert image header from physical to anatomical (0/1/2). (optional)" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'c');
        option->SetLongName( "convert_to_anatomical_header");
        option->SetDescription( description );
        option->SetUsageOption( 0, "0: Default. The image data and the image header orientation are written as is " );
        option->SetUsageOption( 1, "1: Image data is written as is but the image orientation is changed to make it more anatomical. Header orientation and the image data are still consistent." );
        option->SetUsageOption( 2, "2: Image is transformed based on the new anatomical header orientation. A new resampled image with standard axial header orientation is written. " );
        this->AddOption( option );
    }
}


bool Bruker_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputDataFolder()!=std::string(""))
        return 1;
    return 0;
}

std::string Bruker_PARSER::getInputDataFolder()
{
    OptionType::Pointer option = this->GetOption( "import_folder"); 
    if(option->GetNumberOfFunctions())   
        return option->GetFunction(0)->GetName();    
    else
       return std::string("");   
}


std::string Bruker_PARSER::getOutputProcFolder()
{
    OptionType::Pointer option = this->GetOption( "output_folder"); 
    if(option->GetNumberOfFunctions())   
        return option->GetFunction(0)->GetName();    
    else
       return std::string("");  
}


/*
bool Bruker_PARSER::getUseGradientsInsteadOfBMatrix()
{
   OptionType::Pointer option = this->GetOption( "gradients"); 
   
   if(option->GetNumberOfFunctions())   
        return bool(atoi(option->GetFunction(0)->GetName().c_str()));    
   else
       return 0;   
}
*/


int Bruker_PARSER::getConvertToAnatomicalHeader()
{
   OptionType::Pointer option = this->GetOption( "convert_to_anatomical_header");

   if(option->GetNumberOfFunctions())
        return atoi(option->GetFunction(0)->GetName().c_str());
   else
       return 0;
}
