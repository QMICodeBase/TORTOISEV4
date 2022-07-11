#ifndef _Extract_DWI_Subset_PARSER_h
#define _Extract_DWI_Subset_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class Extract_DWI_Subset_PARSER : public itk::ants::CommandLineParser
{
public:
    Extract_DWI_Subset_PARSER( int argc , char * argv[] );
    ~Extract_DWI_Subset_PARSER();
           
    std::string getInputImageName();
    std::string getOutputImageName();
    std::string getBvalsString();
    std::string getVolIdsString();
     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
