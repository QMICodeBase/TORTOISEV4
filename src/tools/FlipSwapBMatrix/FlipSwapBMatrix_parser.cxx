#include "FlipSwapBMatrix_parser.h"
#include <algorithm>
#include <ctype.h>






FlipSwapBMatrix_PARSER::FlipSwapBMatrix_PARSER( int argc , char * argv[] )
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


FlipSwapBMatrix_PARSER::~FlipSwapBMatrix_PARSER()
{       
}


void FlipSwapBMatrix_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "This program swaps or flips the Bmatrix based on user input");
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        
}


void FlipSwapBMatrix_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser2::OptionType OptionType;

   
    {
        std::string description = std::string( "Full path to Input Bmatrix." );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input_bmtxt");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "Full path to Output Bmatrix." );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output_bmtxt");
        option->SetDescription( description );
        this->AddOption( option );
    }

    {
        std::string description = std::string( "New X orientation. Examples:  (--X  x), (--X  -z)  (--X  y). Default: x" );

        OptionType::Pointer option = OptionType::New();        
        option->SetLongName( "X");
        option->SetShortName( 'X');
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "New Y orientation. Examples:  (--Y  x), (--Y  -z)  (--Y  y). Default: y" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "Y");
        option->SetShortName( 'Y');
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "New Z orientation. Examples:  (--Z  x), (--Z  -z)  (--Z  y). Default: z" );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "Z");
        option->SetShortName( 'Z');
        option->SetDescription( description );
        this->AddOption( option );
    }
}




std::string FlipSwapBMatrix_PARSER::getInputBMatrix()
{
    OptionType::Pointer option = this->GetOption( "input_bmtxt");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}


std::string FlipSwapBMatrix_PARSER::getOutputBMatrix()
{
    OptionType::Pointer option = this->GetOption( "output_bmtxt");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::string FlipSwapBMatrix_PARSER::getNewX()
{
    OptionType::Pointer option = this->GetOption( "X");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("x");
}


std::string FlipSwapBMatrix_PARSER::getNewY()
{
    OptionType::Pointer option = this->GetOption( "Y");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("y");
}

std::string FlipSwapBMatrix_PARSER::getNewZ()
{
    OptionType::Pointer option = this->GetOption( "Z");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("z");
}



bool FlipSwapBMatrix_PARSER::checkIfAllRequiredParamsAreEntered()
{
    
    if(this->getInputBMatrix()==std::string(""))
    {
        std::cout<<"Input bmtxt name not entered...Exiting..."<<std::endl;
        return 0;
    }

    bool checkeds[3]={0,0,0};

    {
        std::string mx = this->getNewX();
        if(mx.find("x")!=std::string::npos)
            checkeds[0]=1;
        if(mx.find("y")!=std::string::npos)
            checkeds[1]=1;
        if(mx.find("z")!=std::string::npos)
            checkeds[2]=1;
    }
    {
        std::string my = this->getNewY();
        if(my.find("x")!=std::string::npos)
            checkeds[0]=1;
        if(my.find("y")!=std::string::npos)
            checkeds[1]=1;
        if(my.find("z")!=std::string::npos)
            checkeds[2]=1;
    }
    {
        std::string mz = this->getNewZ();
        if(mz.find("x")!=std::string::npos)
            checkeds[0]=1;
        if(mz.find("y")!=std::string::npos)
            checkeds[1]=1;
        if(mz.find("z")!=std::string::npos)
            checkeds[2]=1;
    }

    if(!checkeds[0] | !checkeds[1]  | !checkeds[2])
    {
        std::cout<<"Not all new X, Y, Z directions entered. Exiting..."<<std::endl;
        return 0;
    }


    return 1;
}




