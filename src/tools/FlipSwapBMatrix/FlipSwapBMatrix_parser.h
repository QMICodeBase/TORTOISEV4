#ifndef _FlipSwapBMatrix_PARSER_h
#define _FlipSwapBMatrix_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser2.h"

class FlipSwapBMatrix_PARSER : public itk::ants::CommandLineParser2
{
public:
    FlipSwapBMatrix_PARSER( int argc , char * argv[] );
    ~FlipSwapBMatrix_PARSER();
           
    std::string getInputBMatrix();
    std::string getOutputBMatrix();
    std::string getNewX();
    std::string getNewY();
    std::string getNewZ();


     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();    
};





#endif
