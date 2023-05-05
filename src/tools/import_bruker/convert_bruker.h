#ifndef _CONVERT_BRUKER_h
#define _CONVERT_BRUKER_h

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>


#include "defines.h"

namespace fs = boost::filesystem;





struct METHOD_struct
{
    int N_averages;
    int N_reps;
    float TR;
    std::string ReadOrient;
    std::string SliceOrient;
    int N_A0;
    int NdiffExp;
    int Ndir;
    int N_totalvol;
    float RG;
    float small_delta, BIG_DELTA;
    vnl_vector<double> eff_bval;
    vnl_vector<double> grad_phase, grad_read,grad_slice;

    vnl_matrix<double> Bmatrix;
    int PVM_EPI_BlipDir;
    std::string phase_mode;

    vnl_matrix_fixed<double,3,3> grad_orient;
};


struct VISU_struct
{
    int dim;
    int size[3];
    float extent[3];
    vnl_matrix_fixed<double,3,3> orientation;
    float pos[3];
    float intercept,slope;
    std::string word_type;
    int nslices;
    float slice_thickness;
    std::string subject_position;
};


struct RECO_struct
{
public:

    std::string RECO_WORDTYPE;
    std::string RECO_BYTE_ORDER;
    int RECO_SIZE[3];
    std::string RECO_MAP_MODE;
    int RECO_TRANSPOSITION;
    float RECO_FOV[3];
    vnl_matrix<double> RECO_MAP_SLOPE;
    int RECO_IR_SCALE;
    std::string version;
};







#endif
