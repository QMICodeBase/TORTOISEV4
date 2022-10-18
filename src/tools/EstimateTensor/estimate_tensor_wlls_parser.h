#ifndef _Estimate_Tensor_WLLS_PARSER_h
#define _Estimate_Tensor_WLLS_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class EstimateTensorWLLS_PARSER : public itk::ants::CommandLineParser
{
public:
    EstimateTensorWLLS_PARSER( int argc , char * argv[] );
    ~EstimateTensorWLLS_PARSER();
           
    std::string getInputImageName();
    std::string getMaskImageName();
    double getBValCutoff();
    bool getUseNoise();
    bool getUseVoxelwiseBmat();
    std::string getInclusionImg();
    std::string getRegressionMode();
    float getFreeWaterDiffusivity();

    bool getWriteCSImg();

     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
