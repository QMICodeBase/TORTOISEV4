#ifndef _TORTOISE_PARSER_h
#define _TORTOISE_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"
#include "../tools/DRBUDDI/DRBUDDI_parserBase.h"


class TORTOISE_PARSER : public DRBUDDI_PARSERBASE
{
public:
    TORTOISE_PARSER( int argc , char * argv[] );
    ~TORTOISE_PARSER();

    using Superclass = DRBUDDI_PARSERBASE;

    //std::string getUpInputName();
    std::string getUpBvecName();
    std::string getUpBvalName();

    //std::string getDownInputName();
    std::string getDownBvecName();
    std::string getDownBvalName();

    std::string getTempProcFolder();

    //std::vector<std::string> getStructuralNames();
    std::string getReorientationName();
    //std::string getGradNonlinInput();
    //void setGradNonlinInput(std::string fname);
    //bool getGradNonlinIsGE();
    //std::string getGradNonlinGradWarpDim();

    int getFlipX();
    int getFlipY();
    int getFlipZ();

    float getBigDelta();
    float getSmallDelta();


    std::string getStartStep();
    bool getDoQC();
    bool getRemoveTempFolder();


    std::string getDenoising();
    int getDenoisingKernelSize();
    bool getGibbs();
    float getGibbsKSpace();
    int getGibbsNsh();
    int getGibbsMinW();
    int getGibbsMaxW();
    int getB0Id();
    bool getIsHuman();
    std::string getRotEddyCenter();
    bool getCenterOfMass();
    std::string getCorrectionMode();
    bool getS2V();
    bool getRepol();
    float getOutlierFrac();
    float getOutlierProbabilityThreshold();
    int getOutlierNumberOfResidualClusters();
    int getOutlierReplacementModeAggessive();

    int getNiter();
    int getDTIBval();
    int getHARDIBval();

    std::string getDrift();

    std::string getEPI();
    void setEPI(std::string nepi);

    /*
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

    int getNumberOfStructurals();
    std::string getStructuralNames(int str_id);
    */

    std::string getOutputName();
    std::string getOutputOrientation();
    std::vector<int> GetOutputNVoxels();
    std::vector<float> GetOutputRes();
    std::string getOutputDataCombination();
    std::string getOutputGradientNonlinearityType();

     
private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();    
};





#endif
