#ifndef _DRBUDDI_PARSERBASE_h
#define _DRBUDDI_PARSERBASE_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"


class DRBUDDI_PARSERBASE : public itk::ants::CommandLineParser
{
public:
    DRBUDDI_PARSERBASE( int argc , char * argv[],bool print );
    ~DRBUDDI_PARSERBASE();

    std::string getUpInputName();
    std::string getUpJSonName();
    std::string getDownInputName();
    std::vector<std::string> getStructuralNames();
    std::string getDRBUDDIOutput();
    int getDRBUDDIStep();

    std::string getGradNonlinInput();
  //  void setGradNonlinInput(std::string fname);
    bool getGradNonlinIsGE();
    std::string getGradNonlinGradWarpDim();
    bool getNOGradWarp();

    bool getDisableInitRigid();
    bool getStartWithDiffeo();
    std::string  getRigidMetricType();
    float  getRigidLR();
    int getDWIBvalue();

    std::string GetInitialMINV();
    std::string GetInitialFINV();
    int getNumberOfStages();
    std::vector<std::string>  getStageString(int st);
    int GetNIter(int st);
    int GetF(int st);
    float GetS(int st);
    float GetLR(int st);
    float GetUStd(int st);
    float GetTStd(int st);
    bool GetRestrict(int st);
    bool GetConstrain(int st);
    int GetNMetrics(int st);
    std::string GetMetricString(int st,int m);
    bool getEstimateLRPerIteration();

    float getStructuralWeight();
    bool getDisableLastStage();

    bool getEnforceFullAntiSymmetry();

    int getNumberOfStructurals();
    std::string getStructuralNames(int str_id);

    bool getDisableITKThreads();
    std::string getRegistrationMethodType();

    int getNumberOfCores();

protected:
    void InitializeCommandLineOptions();

     
private:
    virtual void CreateParserandFillText(int argc , char * argv[] );
    
};





#endif
