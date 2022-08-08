#ifndef _COMPUTE_NS_DEC_MAP_PARSER_h
#define _COMPUTE_NS_DEC_MAP_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"

class ComputeNSDecMap_PARSER : public itk::ants::CommandLineParser
{
public:
    ComputeNSDecMap_PARSER( int argc , char * argv[] );
    ~ComputeNSDecMap_PARSER();
           
    std::string getInputImageName();
    double getColorParameter0();
    double getColorParameter1();
    double getColorParameter2();
    double getColorParameter3();
    double getGammaFactor();
    double getPercentBeta();
    double getLatticeIndexMax();
    double getLatticeIndexMin();
    double getScaleXp();
     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();
};





#endif

