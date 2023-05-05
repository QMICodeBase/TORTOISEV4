#include "bruker_anatomical_parser.h"
#include <algorithm>
#include <ctype.h>



BrukerAnatomical_PARSER::BrukerAnatomical_PARSER( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);    
    this->Parse(argc,argv);
    
       
    
    if( argc == 1 )
    {
        this->PrintMenu( std::cout, 5, false );
        exit(0);
    }
    
    if(checkIfAllRequiredParamsAreEntered()==0)
    {
        std::cout<<"Not all the required Parameters are entered! Exiting!"<<std::endl;
        exit(0);
    }   
} 


BrukerAnatomical_PARSER::~BrukerAnatomical_PARSER()
{       
}


void BrukerAnatomical_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program imports  " )
      + std::string( "anatomical (structural) data from Bruker's native format to NIFTI" );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void BrukerAnatomical_PARSER::InitializeCommandLineOptions()
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
        std::string description = std::string( "Full path to the output filename. (optional)" );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output_filename");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Convert image header from physical to anatomical (0/1). (optional)" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'c');
        option->SetLongName( "convert_to_anatomical_header");
        option->SetDescription( description );
        option->SetUsageOption( 0, "0: Default. The image is written as axial if the acquisition is axial, coronal if it is coronal and sagittal if it is sagittal." );
        option->SetUsageOption( 1, "1: coronal acquisitions are assumed to be the anatomical axial and the headers are changed accordingly." );
        this->AddOption( option );
    }
}


bool BrukerAnatomical_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputDataFolder()!=std::string(""))
        return 1;
    return 0;
}

std::string BrukerAnatomical_PARSER::getInputDataFolder()
{
    OptionType::Pointer option = this->GetOption( "import_folder"); 
    if(option->GetNumberOfFunctions())   
        return option->GetFunction(0)->GetName();    
    else
       return std::string("");   
}


std::string BrukerAnatomical_PARSER::getOutputFilename()
{
    OptionType::Pointer option = this->GetOption( "output_filename");
    if(option->GetNumberOfFunctions())   
        return option->GetFunction(0)->GetName();    
    else
       return std::string("");  
}


bool BrukerAnatomical_PARSER::getConvertToAnatomicalHeader()
{
   OptionType::Pointer option = this->GetOption( "convert_to_anatomical_header");

   if(option->GetNumberOfFunctions())
        return bool(atoi(option->GetFunction(0)->GetName().c_str()));
   else
       return 0;
}
