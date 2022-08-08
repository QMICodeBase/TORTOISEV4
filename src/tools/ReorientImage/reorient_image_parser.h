#ifndef _Reorient_Image_PARSER_h
#define _Reorient_Image_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class Reorient_Image_PARSER : public itk::ants::CommandLineParser
{
public:
    Reorient_Image_PARSER( int argc , char * argv[] );
    ~Reorient_Image_PARSER();
           
    std::string getInputImageName();
    std::string getOriginalOrientation();
    std::string getDesiredOrientation();
    std::string getOutputName();
    std::string getPhase();


     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
