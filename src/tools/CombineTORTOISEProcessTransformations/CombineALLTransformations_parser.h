#ifndef _COMPUTE_DEC_MAP_PARSER_h
#define _COMPUTE_DEC_MAP_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class CombineALLTransformations_PARSER : public itk::ants::CommandLineParser
{
public:
    CombineALLTransformations_PARSER( int argc , char * argv[] );
    ~CombineALLTransformations_PARSER();
           
    std::string getInputImageName();
    std::string getOutputName();
    std::string getTemplateName();

    std::string getMotionEddyParametersName();
    std::string getPE();
    std::string getRotCenter();
    std::string getBlipDownToUpTransformationName();
    std::string getGradientNonlinearityTransformationName();
    std::string getEPITransformationName();
    std::string getBo2StrTransformationName();




     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
