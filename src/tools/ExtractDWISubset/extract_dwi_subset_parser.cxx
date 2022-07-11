#include "extract_dwi_subset_parser.h"
#include <algorithm>
#include <ctype.h>






Extract_DWI_Subset_PARSER::Extract_DWI_Subset_PARSER( int argc , char * argv[] )
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


Extract_DWI_Subset_PARSER::~Extract_DWI_Subset_PARSER()
{       
}


void Extract_DWI_Subset_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program extracts a subset of the input DWIs (listfile) and creates another listfile with corresponding NIFTI image and bmtxt file." );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void Extract_DWI_Subset_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;

   
    {
        std::string description = std::string( "Full path to the input nifti" );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input_image");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Full path to the output nifti" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output_image");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "b-values of the volume to be extracted. Comma separated. A range can be given with - .  For example : 0-500,750,3000-5000 will include all the volumes with b-values in between (including) 0 and 500 and bvalue 750 and all volumes in between #K and 5K (included)" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'b');
        option->SetLongName( "bvals");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Volume numbers to be extracted. Comma separated. A range can be given with - .  For example : 0,3-5,7 will include volumes 0,3,4,5,7." );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'v');
        option->SetLongName( "vols");
        option->SetDescription( description );
        this->AddOption( option );
    }
}




std::string Extract_DWI_Subset_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::string Extract_DWI_Subset_PARSER::getOutputImageName()
{
    OptionType::Pointer option = this->GetOption( "output_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}



std::string Extract_DWI_Subset_PARSER::getVolIdsString()
{
    OptionType::Pointer option = this->GetOption( "vols");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName().c_str();
    else
       return std::string("");
}

std::string Extract_DWI_Subset_PARSER::getBvalsString()
{
    OptionType::Pointer option = this->GetOption( "bvals");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName().c_str();
    else
       return std::string("");
}



bool Extract_DWI_Subset_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    if(this->getBvalsString()==std::string("") &&  this->getVolIdsString()==std::string("") )
    {
        std::cout<<"Neither desired b-values or volume ids are entered...Exiting..."<<std::endl;
        return 0;
    }

    if(this->getBvalsString()!=std::string("") &&  this->getVolIdsString()!=std::string("") )
    {
        std::cout<<"Both desired b-values and volume ids are entered. Only one of them can be active...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




