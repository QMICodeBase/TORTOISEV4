#include "defines.h"
#include "fit_nifti_iso_parser.h"
#include <algorithm>
#include <ctype.h>



Fit_NIFTI_iso_parser::Fit_NIFTI_iso_parser( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);
    this->Parse(argc,argv);


    if( argc == 1 )    {
        std::cout<<"Simple Usage:"<<std::endl<<std::endl;
        std::cout<<"Fit_nifti_iso_temp -i full_path_to_your_nifti_file"<<std::endl;
        this->PrintMenu( std::cout, 5, false );
        exit(EXIT_FAILURE);
    }

    if(checkIfAllRequiredParamsAreEntered()==0)
    {
        std::cout<<"Not all the required Parameters are entered! Exiting!"<<std::endl;
        exit(0);
    }
}


Fit_NIFTI_iso_parser::~Fit_NIFTI_iso_parser(){}


void Fit_NIFTI_iso_parser::CreateParserandFillText(int argc, char* argv[])
{
    this->SetCommand( argv[0] );

    std::string commandDescription = std::string( "This program estimates gradient coefficients (fit_nifti_iso of IDL)" );

    this->SetCommandDescription( commandDescription );
    this->InitializeCommandLineOptions();
}



void Fit_NIFTI_iso_parser::InitializeCommandLineOptions()
{

    // Explanation of parameters

    typedef itk::ants::CommandLineParser::OptionType OptionType;

    {
        std::string description = std::string( "Full path to the input image" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input_image");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string("Full path to additional input image files in case several input datasets are to be used during the estimation of nonlinearity. Comma seperated, no spaces. (Optional)");
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'e');
        option->SetLongName( "extra_input_images");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to output file (optional)" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Full path to the mask NIFTI image (optional)" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'm');
        option->SetLongName( "mask");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Threshold for creating the mask. Values lower than thr * max(b0_image) will be excluded. Default: 0.01 (optional)" );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 't');
        option->SetLongName( "threshold");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Number of times erosion is applied in creating the mask (optional), default :3" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "erodeFactor");
        option->SetDescription( description );
        this->AddOption( option );

    }


    {
        std::string description = std::string( "Path to the input manufacturer gradient nonlinearity coefficient filename (such as gw_coils.dat, .grad). When this file is provided, nonlinearities will not be estimated, only the gradient miscalibration will be. (optional)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "gradfn");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Radius of the roi for fitting diffusitivity in number of voxels. Optional. Default:1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "rdfit");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "Known diffusitivity of phantom (optional)" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "phantomD");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Radius for coordinate scaling, default: 250mm" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "r0");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Flag to compute gradient miscalibration. (Optional). Default:1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "compute_grad_miscalibration");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Scanner Type. Default: NonGE" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "scanner_type");
        option->SetUsageOption(0, "GE" );
        option->SetUsageOption(1, "Philips" );
        option->SetUsageOption(2, "Siemens" );
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Gradient Normalization. (Optional)." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "grad_normalization");
        option->SetUsageOption(0, "off: no normalization." );
        option->SetUsageOption(1, "x: x gradient normalization." );
        option->SetUsageOption(2, "y: y gradient normalization." );
        option->SetUsageOption(3, "z: z gradient normalization." );
        option->SetUsageOption(4, "avg: Average scalings will be 1." );


        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Path to the file containing the (l,m) indices of the coefficients to compute. Default: default_keys.txt");
        // Add more explanation, as return to the path of default
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "coeffs_to_compute");
        option->SetDescription( description );
        this->AddOption( option );
    }





}


std::string Fit_NIFTI_iso_parser::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input_image");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::string Fit_NIFTI_iso_parser::getMaskImageName()
{
    OptionType::Pointer option = this->GetOption( "mask");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::string Fit_NIFTI_iso_parser::getOuputFilename()
{
    OptionType::Pointer option = this->GetOption( "output");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");


}


std::string Fit_NIFTI_iso_parser::getGradCalName()
{
    OptionType::Pointer option = this->GetOption( "gradfn");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::string Fit_NIFTI_iso_parser::getCoefFile(){
    OptionType::Pointer option = this->GetOption( "coeffs_to_compute");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");

}


std::string Fit_NIFTI_iso_parser::getExtraInputs(){
    OptionType::Pointer option = this->GetOption( "extra_input_images");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");

}

float Fit_NIFTI_iso_parser::getPhantomDiffusitivity()
{
    OptionType::Pointer option = this->GetOption( "phantomD");
    if(option->GetNumberOfFunctions())
        return atof(option->GetFunction(0)->GetName().c_str());
    else
       return 0.00;
}

int Fit_NIFTI_iso_parser::getErodeFactor(){

    OptionType::Pointer option = this->GetOption("erodeFactor");
    if(option->GetNumberOfFunctions())
    {
         return atoi(option->GetFunction(0)->GetName().c_str());
    }
    else
        return 3;
}


double Fit_NIFTI_iso_parser::getThreshold(){

    OptionType::Pointer option = this->GetOption("threshold");
    if(option->GetNumberOfFunctions())
    {
        return atof(option->GetFunction(0)->GetName().c_str());
    }
    else
        return 0.01;
}

// Scaner type: GE, Philips or Siemens
std::string Fit_NIFTI_iso_parser::getScannerType(){

    OptionType::Pointer option = this->GetOption("scanner_type");
    if(option->GetNumberOfFunctions())
    {
        return option->GetFunction(0)->GetName().c_str();
    }
    else
        return "Siemens";
}


double Fit_NIFTI_iso_parser::getDistance(){

    OptionType::Pointer option = this->GetOption("r0");
    if(option->GetNumberOfFunctions())
    {
        return atof(option->GetFunction(0)->GetName().c_str());
    }
    else
        return 250;
}



int Fit_NIFTI_iso_parser::getRadiusFit(){
    // Radius of region of interest of fitting
    OptionType::Pointer option = this->GetOption("rdfit");
    if(option->GetNumberOfFunctions())
    {
        return atoi(option->GetFunction(0)->GetName().c_str());
    }
    else
        return 0;
}

std::string Fit_NIFTI_iso_parser::getGradient_normalization(){
    OptionType::Pointer option = this->GetOption("grad_normalization");
    if(option->GetNumberOfFunctions())
    {
        return option->GetFunction(0)->GetName().c_str();
    }
    else
       return "avg";
}





bool Fit_NIFTI_iso_parser::IsEstimate_gradient_calibration(){
    OptionType::Pointer option = this->GetOption( "compute_grad_miscalibration");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 1;
}




bool Fit_NIFTI_iso_parser::checkIfAllRequiredParamsAreEntered()
{

    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }
//    else

    return 1;
}

