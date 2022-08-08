#include "reorient_image_parser.h"
#include <algorithm>
#include <ctype.h>






Reorient_Image_PARSER::Reorient_Image_PARSER( int argc , char * argv[] )
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


Reorient_Image_PARSER::~Reorient_Image_PARSER()
{       
}


void Reorient_Image_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program reorients a 3D NIFTI image with a given orientation to a standard axial orientation.  " );

    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void Reorient_Image_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;

   
    {
        std::string description = std::string( "Full path to the NIFTI image" );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input_image");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Original orientation of the image. First letter for the anatomical direction that is from left of the image TOWARDS right (The letter if for the ending point of the direction not the beginning). Second letter from the top of the image to bottom. Third letter from the first slice to the last slice. Examples: LPS (axial), PIR (sagittal), LIP (coronal). If NOT provided, image header will be used." );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'r');
        option->SetLongName( "input_orient");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Desired orientation of the image. Default: LPS" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'd');
        option->SetLongName( "output_orient");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Output name" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output");
        option->SetDescription( description );
        this->AddOption( option );
    }





}





std::string Reorient_Image_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string Reorient_Image_PARSER::getOriginalOrientation()
{
    OptionType::Pointer option = this->GetOption( "input_orient");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string Reorient_Image_PARSER::getDesiredOrientation()
{
    OptionType::Pointer option = this->GetOption( "output_orient");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("LPS");
}

std::string Reorient_Image_PARSER::getOutputName()
{
    OptionType::Pointer option = this->GetOption( "output");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}



bool Reorient_Image_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




