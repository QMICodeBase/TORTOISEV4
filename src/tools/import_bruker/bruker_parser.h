#ifndef _Bruker_PARSER_h
#define _Bruker_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class Bruker_PARSER : public itk::ants::CommandLineParser
{
public:
    Bruker_PARSER( int argc , char * argv[] );
    ~Bruker_PARSER();
           
    std::string getInputDataFolder();
    std::string getOutputProcFolder();
//    bool getUseGradientsInsteadOfBMatrix();

    int getConvertToAnatomicalHeader();
     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
