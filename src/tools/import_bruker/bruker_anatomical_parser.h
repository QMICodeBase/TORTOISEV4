#ifndef _BrukerAnatomical_PARSER_h
#define _BrukerAnatomical_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class BrukerAnatomical_PARSER : public itk::ants::CommandLineParser
{
public:
    BrukerAnatomical_PARSER( int argc , char * argv[] );
    ~BrukerAnatomical_PARSER();
           
    std::string getInputDataFolder();
    std::string getOutputFilename();
    bool getConvertToAnatomicalHeader();
     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
