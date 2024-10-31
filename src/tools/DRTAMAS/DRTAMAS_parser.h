#ifndef _DRTAMAS_PARSER_h
#define _DRTAMAS_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"


class DRTAMAS_PARSER : public itk::ants::CommandLineParser
{
public:
    DRTAMAS_PARSER( int argc , char * argv[]);
    ~DRTAMAS_PARSER();
    
    
    std::string getFixedTensor();
    std::string getMovingTensor();
    
    std::vector<std::string> getFixedStructuralNames();
    std::vector<std::string> getMovingStructuralNames();    
    
    std::string getOutput();
    int getStep();
    
    std::string GetInitialMINV();
    std::string GetInitialFINV();

    std::string getInitialRigidTransform();


    int getNumberOfStages();
    std::vector<std::string>  getStageString(int st);
    int GetNIter(int st);
    int GetF(int st);
    float GetS(int st);
    float GetLR(int st);
    float GetUStd(int st);
    float GetTStd(int st);

    int GetNMetrics(int st);
    std::string GetMetricString(int st,int m);


    int getNumberOfStructurals();
    std::string getFixedStructuralName(int str_id);
    std::string getMovingStructuralName(int str_id);

    std::string getRegistrationMethodType();

    bool getOnlyAffine();



protected:
    void InitializeCommandLineOptions();

     
private:
    virtual void CreateParserandFillText(int argc , char * argv[] );
    
};





#endif
