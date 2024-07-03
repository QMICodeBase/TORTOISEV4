#include "DRTAMAS_parser.h"
#include <algorithm>
#include <ctype.h>





DRTAMAS_PARSER::DRTAMAS_PARSER( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);    

    this->Parse(argc,argv);
    
           
    if( argc == 1)
    {
        this->PrintMenu( std::cout, 5, 0,false );
        exit(EXIT_FAILURE);
    }
    
} 


DRTAMAS_PARSER::~DRTAMAS_PARSER()
{       
}


void DRTAMAS_PARSER::CreateParserandFillText(int argc, char* argv[])
{  
    this->SetCommand( argv[0] );
 
    std::string commandDescription = std::string( "The main DRTAMAS program.  It takes in fixed and moving diffusion tensors (and optionally corresponding anatomical images) and perform diffeomorphic registration." );
    
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


void DRTAMAS_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;
   
    {
        std::string description = std::string( "Full path to the input fixed diffusion tensor. REQUIRED." );
 
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'f');
        option->SetLongName( "fixed_tensor");
        option->SetDescription( description );        
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the input moving diffusion tensor. REQUIRED." );

        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'm');
        option->SetLongName( "moving_tensor");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the fixed anatomical image files. Have to be in the same space as the tensor. Can provide more than one. OPTIONAL." );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "fixed_anatomical");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the moving anatomical image files. Have to be in the same space as the tensor. Can provide more than one. OPTIONAL." );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "moving_anatomical");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Start step. 0: beginning, 1: diffeo, 2:write OPTIONAL." );

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "step");
        option->SetDescription( description );
        this->AddOption( option );
    }


    {
        std::string description = std::string( "DRTAMAS runs many  stages during registration. This tag sets all the parameters for a given stage. Each stage is executed in the order provided on the command line. " ) ;

        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "DRTAMAS_stage" );
        option->SetUsageOption(0, "\[learning_rate=\{learning_rate\},cfs=\{Niterations:downsampling_factor:image_smoothing_stdev\},field_smoothing=\{update_field_smoothing_stdev:total_field_smoothing_stdev\} \]" );
        option->SetUsageOption(1, "\[learning_rate=\{0.5\},cfs=\{100:1:0\},field_smoothing=\{3.:0.1\}\]" );
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }    

    {
        std::string description = std::string("Initial transform field for the fixed data.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "initial_fixed_transform");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Initial transform field for the moving data.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "initial_moving_transform");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Registration transformation type. Options: SyN or TVVF. Default: SyN.  TVVF only works in CUDA version.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "transformation_type");
        option->SetDescription( description );
        option->SetModule(6);
        this->AddOption( option );
    }
    {
        std::string description = std::string("Do only affine and not diffeomorphic?.")  ;
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "only_affine");
        option->SetDescription( description );
        option->SetModule(0);
        this->AddOption( option );
    }

}

bool DRTAMAS_PARSER::getOnlyAffine()
{
    OptionType::Pointer option = this->GetOption( "only_affine");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
       return 0;
}

std::string DRTAMAS_PARSER::GetInitialFINV()
{
    OptionType::Pointer option = this->GetOption( "initial_fixed_transform");
   if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
   else
       return std::string("");

}
std::string DRTAMAS_PARSER::GetInitialMINV()
{
    OptionType::Pointer option = this->GetOption( "initial_moving_transform");
   if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
   else
       return std::string("");

}


int DRTAMAS_PARSER::getStep()
{
    OptionType::Pointer option = this->GetOption( "step");
    if(option->GetNumberOfFunctions())
        return atoi(option->GetFunction(0)->GetName().c_str());
    else
       return 0;
}

std::string DRTAMAS_PARSER::getFixedTensor()
{
    OptionType::Pointer option = this->GetOption( "fixed_tensor");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}

std::string DRTAMAS_PARSER::getMovingTensor()
{
    OptionType::Pointer option = this->GetOption( "moving_tensor");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
       return std::string("");
}





std::vector<std::string> DRTAMAS_PARSER::getFixedStructuralNames()
{
    std::vector<std::string> names;

    OptionType::Pointer option = this->GetOption( "fixed_anatomical");
    int nstr= option->GetNumberOfFunctions();
    if(nstr>0)
    {
        for(int str_id=0;str_id<nstr;str_id++)
        {
            std::string nm= option->GetFunction(nstr-1-str_id)->GetName();
            names.push_back(nm);
        }
    }
    return names;
}


std::vector<std::string> DRTAMAS_PARSER::getMovingStructuralNames()
{
    std::vector<std::string> names;

    OptionType::Pointer option = this->GetOption( "moving_anatomical");
    int nstr= option->GetNumberOfFunctions();
    if(nstr>0)
    {
        for(int str_id=0;str_id<nstr;str_id++)
        {
            std::string nm= option->GetFunction(nstr-1-str_id)->GetName();
            names.push_back(nm);
        }
    }
    return names;
}



int DRTAMAS_PARSER::getNumberOfStages()
{
    OptionType::Pointer option = this->GetOption( "DRTAMAS_stage");
    int nstg= option->GetNumberOfFunctions();
    return nstg;
}


std::vector<std::string>  DRTAMAS_PARSER::getStageString(int st)
{
    int nstg= this->getNumberOfStages();

    std::vector<std::string> params;

    OptionType::Pointer option = this->GetOption( "DRTAMAS_stage");
    if(option->GetNumberOfFunctions())
        if(st <option->GetNumberOfFunctions())
        {
            for(int p=0;p<option->GetFunction( nstg-1-st )->GetNumberOfParameters();p++)
            {
                std::string param=( option->GetFunction( nstg-1-st )->GetParameter( p ) );
                params.push_back(param);
            }

            if(params.size()!=3)
            {
                std::cout<<"Incorrect format for stage parameters. There should be 3 groups of parameters. Exiting!!"<<std::endl;
                exit(EXIT_FAILURE);
            }
        }
    return params;
}



int DRTAMAS_PARSER::GetNIter(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("cfs=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(5,pos-5);

            std::string val_string = param_string.substr(0,param_string.find(":"));
            return atoi(val_string.c_str());
        }
    }
    return 100;
}


int DRTAMAS_PARSER::GetF(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("cfs=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(5,pos-5);

            int pos1= param_string.find(":");
            int pos2= param_string.rfind(":");

            std::string val_string = param_string.substr(pos1+1,pos2-pos1-1);
            return atoi(val_string.c_str());
        }
    }
    return 1;
}


float DRTAMAS_PARSER::GetS(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("cfs=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(5,pos-5);

            int pos2= param_string.rfind(":");

            std::string val_string = param_string.substr(pos2+1);
            return atof(val_string.c_str());
        }
    }
    return 0;
}


float DRTAMAS_PARSER::GetLR(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("learning_rate=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(15,pos-15);

            return atof(param_string.c_str());
        }
    }
    return 0.25;
}


float DRTAMAS_PARSER::GetUStd(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("field_smoothing=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(17,pos-17);

            pos= param_string.rfind(":");

            std::string val_string = param_string.substr(0,pos);
            return atof(val_string.c_str());
        }
    }
    return 3;
}

float DRTAMAS_PARSER::GetTStd(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("field_smoothing=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(17,pos-17);

            pos= param_string.rfind(":");

            std::string val_string = param_string.substr(pos+1);
            return atof(val_string.c_str());
        }
    }
    return 0;
}


int DRTAMAS_PARSER::getNumberOfStructurals()
{

    OptionType::Pointer option = this->GetOption( "fixed_anatomical");
    int nstr= option->GetNumberOfFunctions();

    return nstr;
}

std::string DRTAMAS_PARSER::getFixedStructuralName(int str_id=0)
{
    int nstr= this->getNumberOfStructurals();

   OptionType::Pointer option = this->GetOption( "fixed_anatomical");
   if(option->GetNumberOfFunctions())
       if(str_id <option->GetNumberOfFunctions())
          return option->GetFunction(nstr-1-str_id)->GetName();
       else
           return option->GetFunction(0)->GetName();

   else
       return std::string("");
}


std::string DRTAMAS_PARSER::getMovingStructuralName(int str_id=0)
{
    int nstr= this->getNumberOfStructurals();

   OptionType::Pointer option = this->GetOption( "moving_anatomical");
   if(option->GetNumberOfFunctions())
       if(str_id <option->GetNumberOfFunctions())
          return option->GetFunction(nstr-1-str_id)->GetName();
       else
           return option->GetFunction(0)->GetName();

   else
       return std::string("");
}


int DRTAMAS_PARSER::GetNMetrics(int st)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("metrics=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(9,pos-9);

            int n = std::count(param_string.begin(), param_string.end(), ':');
            return n+1;
        }
    }
    return 0;
}

std::string DRTAMAS_PARSER::GetMetricString(int st,int m)
{
    std::vector<std::string> params= getStageString(st);

    for(int p=0;p<params.size();p++)
    {
        std::string str= params[p];
        str.erase(remove_if(str.begin(), str.end(), isspace), str.end());

        if(str.find("metrics=")!=std::string::npos)
        {
            int pos= str.rfind("}");
            std::string param_string =  str.substr(9,pos-9);

            for(int m2=0;m2<m;m2++)
            {
                param_string=param_string.substr(param_string.find(":")+1);
            }

            if(param_string.find(":")==std::string::npos)
            {
                std::string val = param_string.substr(0,param_string.rfind("}"));
                return val;
            }
            else
            {
                std::string val = param_string.substr(0,param_string.find(":"));
                return val;
            }

        }
    }
    return "";
}
