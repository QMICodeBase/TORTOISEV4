#ifndef _COMPUTE_DEC_MAP_PARSER_h
#define _COMPUTE_DEC_MAP_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class ComputeRotationMatrixFromTORTOISETransformations_PARSER : public itk::ants::CommandLineParser
{
public:
    ComputeRotationMatrixFromTORTOISETransformations_PARSER( int argc , char * argv[] );
    ~ComputeRotationMatrixFromTORTOISETransformations_PARSER();
           
    std::string getInputImageName();
    int getVolIndex();
    std::string getTemplateName();

    std::string getMotionEddyParametersName();        
    std::string getBlipDownToUpTransformationName();        
    std::string getBo2StrTransformationName();




     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
