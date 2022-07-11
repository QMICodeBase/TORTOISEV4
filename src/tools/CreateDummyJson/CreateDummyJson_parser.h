#ifndef _CreateDummyJson_PARSER_h
#define _CreateDummyJson_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class CreateDummyJson_PARSER : public itk::ants::CommandLineParser
{
public:
    CreateDummyJson_PARSER( int argc , char * argv[] );
    ~CreateDummyJson_PARSER();
           
    std::string getInputImageName();
    std::string getPhaseEncoding();
    int getMBFactor();
    float getPF();


     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
