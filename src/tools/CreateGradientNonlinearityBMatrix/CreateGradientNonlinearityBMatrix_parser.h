#ifndef _CreateGradientNonlinearityBMatrix_PARSER_h
#define _CreateGradientNonlinearityBMatrix_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class CreateGradientNonlinearityBMatrix_PARSER : public itk::ants::CommandLineParser
{
public:
    CreateGradientNonlinearityBMatrix_PARSER( int argc , char * argv[] );
    ~CreateGradientNonlinearityBMatrix_PARSER();
           
    std::string getFinalImageName();
    std::string getInitialImageName();
    
    std::string getNonlinearity();
    bool getIsGE();   
     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
