#ifndef _FIT_NIFTI_ISO_PARSER_H
#define _FIT_NIFTI_ISO_PARSER_H


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"


class Fit_NIFTI_iso_parser : public itk::ants::CommandLineParser
{
public:
    Fit_NIFTI_iso_parser( int argc , char * argv[] );
    ~Fit_NIFTI_iso_parser();

    std::string getInputImageName();
    std::string getMaskImageName();
    std::string getOuputFilename();

    std::string getGradCalName();

    std::string getExtraInputs();
    std::string getCoefFile();

    std::string getScannerType();

    std::string getGradient_normalization();

    float getPhantomDiffusitivity();

    int getErodeFactor();
    double getThreshold();
    double getDistance();
    int getRadiusFit();

    bool IsEstimate_gradient_calibration();
    bool isGradient_normalization();

private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();
};

#endif
