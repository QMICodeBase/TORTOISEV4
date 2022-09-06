#ifndef _FIT_NIFTI_ISO_H
#define _FIT_NIFTI_ISO_H
#include "defines.h"


#include "gradcal.h"
#include "fit_nifti_iso_parser.h"
#include "init_iso_gw.h"
#include "iso_gw.h"
#include "erodeMask.h"


typedef ImageType4D DTImageType4D;

class FITNIFTIISO{

    void process();
public:
    FITNIFTIISO(int argc, char *argv[]);
    ~FITNIFTIISO();
   void setInputList(std::string listname );

private:
    ImageType3D::Pointer b0_image;
    MaskImageType::Pointer maskImage;
    ImageType4D::Pointer my_image;

    std::string mask_filename;
    std::string output_filename;
    std::string gradFn;
    std::string scannerType;
    std::string grad_normalization;
    std::string coef_filename;
    std::string nii_name;

    bool compute_normalization = false;
    bool GE_type = false;
    bool computeCalibration = 0;
    bool computeGains = 0;

    double phantomD = 0;
    double threshold = 0.01;
    double R0 = 250;

    int rdfit = 0;
    int npts = 0;
    int erode_factor = 0;
    int init_flag = 0;
    int nVols =0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int na = 0;

    int locx = 0;
    int locy = 0;
    int locz = 0;
    GRADCAL *grads;
    GradCoef gradCoefOutput;
    basisFuntions basis;

    std::string list_inputs;

    vnl_matrix <double> Bmatrix;
    vnl_matrix_fixed <double,4,4> smat;
    vnl_matrix<double> inv_smat;
    vnl_vector<double> aa;
    vnl_vector<int>fita;
    std::vector<int>nodif;
    int nnodif = 0;

    /* Init Functions */
    void set4Dimage(std::string fn);
    void setb0Image(int volid);
    void setMaskImage(double threshold, int erodeFactor);
    void setGradientCalibrationStructure(std::string fn);
    void setNormalizingFlag();
    void setVectorOfFitting();
    void setPermitFlagsforFitting();

    /* funtions to process()*/
    void normalize_image_nodif(std::vector<int> a, int b);
    void normalize_image_b0();
    void PerformNonlinearityMisCalibrationFitting();
    void normalize_gradient();
    void PerformGainFitting();
    void write_gradcal(std::string fn);

    void writeOutputs();


    double regression_at_point(ImageType3D::IndexType ind3);

};

int IsoGW(int m, int n, double *p, double *deviates,   double **derivs, void *vars);
int IsoGW1(int m, int n, double *p, double *deviates,   double **derivs, void *vars);


#endif
