#ifndef _EstimateExperimentalDeviationScalar_h
#define _EstimateExperimentalDeviationScalar_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class EstimateExperimentalDeviationScalar_PARSER : public itk::ants::CommandLineParser
{
public:
    EstimateExperimentalDeviationScalar_PARSER( int argc , char * argv[] );
    ~EstimateExperimentalDeviationScalar_PARSER();
           
    std::string getInputImageName();
    std::string getMaskImageName();
    double getBValCutoff();
    std::string getModality();


     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
