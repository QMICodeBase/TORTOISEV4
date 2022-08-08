#ifndef _Extract_Image_PARSER_h
#define _Extract_Image_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class Extract_Image_PARSER : public itk::ants::CommandLineParser
{
public:
    Extract_Image_PARSER( int argc , char * argv[] );
    ~Extract_Image_PARSER();
           
    std::string getInputImageName();
    std::string getOutputImageName();
    int  getVolId();
     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
