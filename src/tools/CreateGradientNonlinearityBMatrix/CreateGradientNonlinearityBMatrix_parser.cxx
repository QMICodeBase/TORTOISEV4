#include "CreateGradientNonlinearityBMatrix_parser.h"
#include <algorithm>
#include <ctype.h>






CreateGradientNonlinearityBMatrix_PARSER::CreateGradientNonlinearityBMatrix_PARSER( int argc , char * argv[] )
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


CreateGradientNonlinearityBMatrix_PARSER::~CreateGradientNonlinearityBMatrix_PARSER()
{       
}


void CreateGradientNonlinearityBMatrix_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program generates the HCP style gradient deviation tensor from gradient nonlinearity information.  " );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void CreateGradientNonlinearityBMatrix_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;
   
    {
        std::string description = std::string( "Final processed b=0 NIFTI image.  This image might be in native space or aligned to another image (such as ana anatomical). The gradient deviation tensor will be generated in this space. " );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'f');
        option->SetLongName( "final_image");
        option->SetDescription( description );        
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Initial b=0 NIFTI image. Optional. If not provided, the code will assume the final image is in native space. This image should pretty much be the raw DWIs." );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "initial_image");
        option->SetDescription( description );        
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Gradient Nonlinearity information. Either a manufacturer coeff file, TORTOISE gcal file or an ITK format gradwarp dispalcement field." );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'g');
        option->SetLongName( "nonlinearity");
        option->SetDescription( description );        
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Is the scanner a GE. Optional. If not provided, the code will assume NONGE." );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "isGE");
        option->SetDescription( description );        
        this->AddOption( option );
    }
}


std::string CreateGradientNonlinearityBMatrix_PARSER::getFinalImageName()
{
    OptionType::Pointer option = this->GetOption( "final_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}



std::string CreateGradientNonlinearityBMatrix_PARSER::getInitialImageName()
{
    OptionType::Pointer option = this->GetOption( "initial_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string CreateGradientNonlinearityBMatrix_PARSER::getNonlinearity()
{
    OptionType::Pointer option = this->GetOption( "nonlinearity");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


bool CreateGradientNonlinearityBMatrix_PARSER::getIsGE()
{
    OptionType::Pointer option = this->GetOption( "isGE");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}


bool CreateGradientNonlinearityBMatrix_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getFinalImageName()==std::string(""))
    {
        std::cout<<"Final image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    if(this->getNonlinearity()==std::string(""))
    {
        std::cout<<"Gradient nonlinearity information not entered...Exiting..."<<std::endl;
        return 0;
    }


    return 1;
}




