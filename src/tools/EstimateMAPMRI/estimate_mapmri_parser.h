#ifndef _Estimate_MAPMRI_PARSER_h
#define _Estimate_MAPMRI_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class EstimateMAPMRI_PARSER : public itk::ants::CommandLineParser
{
public:
    EstimateMAPMRI_PARSER( int argc , char * argv[] );
    ~EstimateMAPMRI_PARSER();
           
    std::string getInputImageName();
    std::string getMaskImageName();
    double getBValCutoff();
    bool getUseNoise();
    bool getUseVoxelwiseBmat();
    std::string getInclusionImg();

    std::string getDTIImageName();
    std::string getA0ImageName();
    int getMAPMRIOrder();   
    float getSmallDelta();
    float getBigDelta();
    
    
    
    
    

     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
