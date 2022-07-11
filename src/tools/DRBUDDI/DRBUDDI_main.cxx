#include "defines.h"
#include "DRBUDDI.h"
#include "DRBUDDI_parserBase.h"
#include "registration_settings.h"
#include "TORTOISE.h"


int main(int argc,char *argv[])
{
    TORTOISE t;
    DRBUDDI_PARSERBASE *parser= new DRBUDDI_PARSERBASE(argc,argv,1);

    std::string up_name = parser->getUpInputName();
    std::string json_name= parser->getUpJSonName();
    if(json_name=="")
    {
        std::cout<<"The Json name is required for DRBUDDI. Exiting.."<<std::endl;
        return EXIT_FAILURE;
    }

    std::string down_name = parser->getDownInputName();
    std::vector<std::string> structural_names = parser->getStructuralNames();

    json my_json;
    std::ifstream json_file(json_name);
    json_file >> my_json;
    json_file.close();


    RegistrationSettings::get().setValue<int>("DRBUDDI_DWI_bval_tensor_fitting",parser->getDWIBvalue());


    DRBUDDI myDRBUDDI(up_name,down_name,structural_names,my_json);
    myDRBUDDI.SetParser(parser);
    myDRBUDDI.Process();
}
