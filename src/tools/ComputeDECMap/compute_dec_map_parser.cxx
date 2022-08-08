#include "compute_dec_map_parser.h"
#include <algorithm>
#include <ctype.h>



ComputeDecMap_PARSER::ComputeDecMap_PARSER( int argc , char * argv[] )
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



ComputeDecMap_PARSER::~ComputeDecMap_PARSER()
{       
}


void ComputeDecMap_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program estimates the diffusion tensor with NonLinear Least-Squares regression.. " );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void ComputeDecMap_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;
   
    {
        std::string description = std::string( "Full path to the input tensor image." );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input_tensor");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Color Parameter 0: increases the brightness of the blue (default:0.35)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "color_par_0");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Color Parameter 1: diminishes green component (default:0.8)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "color_par_1");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Color Parameter 2: maximmum luminance for each color (default:0.7)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "color_par_2");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Color Parameter 3: Saturation linearity (default:0.5)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "color_par_3");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Gamma correction factor (default:1.4)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "gamma_fact");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Percent beta (default:0.4)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "percbeta");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "The cutoff value for the lattice index (maxlattindex) when using truncation instead of scaling (default:0.7) " );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "color_lattmax");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "The cutoff value for the lattice index (minlattindex) when using truncation instead of scaling (default:0.)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "color_lattmin");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Correction for perceived brightness (default:0.6)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "color_scalexp");
        option->SetDescription( description );
        this->AddOption( option );
    }
}





std::string ComputeDecMap_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input_tensor");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

double ComputeDecMap_PARSER::getColorParameter0()
{
    OptionType::Pointer option = this->GetOption( "color_par_0");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.35;
}

double ComputeDecMap_PARSER::getColorParameter1()
{
    OptionType::Pointer option = this->GetOption( "color_par_1");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.8;
}

double ComputeDecMap_PARSER::getColorParameter2()
{
    OptionType::Pointer option = this->GetOption( "color_par_2");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.7;
}

double ComputeDecMap_PARSER::getColorParameter3()
{
    OptionType::Pointer option = this->GetOption( "color_par_3");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.5;
}
double ComputeDecMap_PARSER::getGammaFactor()
{
    OptionType::Pointer option = this->GetOption( "gamma_fact");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 1.4;
}
double ComputeDecMap_PARSER::getPercentBeta()
{
    OptionType::Pointer option = this->GetOption( "percbeta");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.4;
}


double ComputeDecMap_PARSER::getLatticeIndexMax()
{
    OptionType::Pointer option = this->GetOption( "color_lattmax");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.7;
}

double ComputeDecMap_PARSER::getLatticeIndexMin()
{
    OptionType::Pointer option = this->GetOption( "color_lattmin");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.0;
}

double ComputeDecMap_PARSER::getScaleXp()
{
    OptionType::Pointer option = this->GetOption( "color_scalexp");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.6;
}


bool ComputeDecMap_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }

    return 1;
}




