#include "WarpMesh_parser.h"
#include <algorithm>
#include <ctype.h>





WarpMesh_PARSER::WarpMesh_PARSER( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);    

    this->Parse(argc,argv);
    
           
    if( argc == 1)
    {
        this->PrintMenu( std::cout, 5, 0,false );
        exit(EXIT_FAILURE);
    }
    
} 


WarpMesh_PARSER::~WarpMesh_PARSER()
{       
}


void WarpMesh_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "The main WarpMesh program.  Can take multiple fields to display their warping." );
    
    this->SetCommandDescription( commandDescription );    
    this->InitializeCommandLineOptions();        

    if(argc <2)
    {
        itk::Indent indent(5);
        std::cout << std::endl;
        std::cout << "COMMAND: " << std::endl;
        std::cout << indent << argv[0] << std::endl;

        std::stringstream ss1;
        ss1 << indent << indent;

        std::stringstream ss2;
        ss2 << commandDescription ;
        std::string description = this->BreakUpStringIntoNewLines(ss2.str(), ss1.str(), 120 );
        std::cout<< indent << indent << description << std::endl;
        std::cout << std::endl;
    }
}


void WarpMesh_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;
   
    {
        std::string description = std::string( "Full path all the displacement fields. REQUIRED." );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'f');
        option->SetLongName( "fields");
        option->SetDescription( description );        
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to all the anatomicals. OPTIONAL" );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'a');
        option->SetLongName( "anatomicals");
        option->SetDescription( description );
        this->AddOption( option );
    }  
}


int WarpMesh_PARSER::getNFields()
{
    OptionType::Pointer option = this->GetOption( "fields");
    return option->GetNumberOfFunctions();
}



int WarpMesh_PARSER::getNAnatomicals()
{
    OptionType::Pointer option = this->GetOption( "anatomicals");
    return option->GetNumberOfFunctions();
}

std::string WarpMesh_PARSER::getNthField(int n)
{  
   int N= getNFields();
   if(n > N-1)
   {
       std::cout<<"Requested field number larger than total number of fields. Exiting..."<<std::endl;
       exit(EXIT_FAILURE);   
   }
   
    OptionType::Pointer option = this->GetOption( "fields");
    return option->GetFunction(N-1-n)->GetName();
}

std::string WarpMesh_PARSER::getNthAnatomical(int n)
{
   int N= getNAnatomicals();
   if(n > N-1)
   {
       std::cout<<"Requested anatomical number larger than total number of anatomicals. Exiting..."<<std::endl;
       exit(EXIT_FAILURE);   
   }
   
    OptionType::Pointer option = this->GetOption( "anatomicals");
    return option->GetFunction(N-1-n)->GetName();
}


