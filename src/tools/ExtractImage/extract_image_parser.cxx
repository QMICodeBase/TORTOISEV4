#include "extract_image_parser.h"
#include <algorithm>
#include <ctype.h>






Extract_Image_PARSER::Extract_Image_PARSER( int argc , char * argv[] )
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


Extract_Image_PARSER::~Extract_Image_PARSER()
{       
}


void Extract_Image_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program extracts and saves a 3D NIFTI volume out of a 4D NIFTI volume" );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void Extract_Image_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;

   
    {
        std::string description = std::string( "Full path to the input NIFTI image filename" );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input_image");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Full path to the output NIFTI image filename" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output_image");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Volume number to be extracted. Default: 0" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'v');
        option->SetLongName( "vol_id");
        option->SetDescription( description );
        this->AddOption( option );
    }

}




std::string Extract_Image_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::string Extract_Image_PARSER::getOutputImageName()
{
    OptionType::Pointer option = this->GetOption( "output_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


int Extract_Image_PARSER::getVolId()
{
    OptionType::Pointer option = this->GetOption( "vol_id");
    if(option->GetNumberOfFunctions())
        return atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 0;
}



bool Extract_Image_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




