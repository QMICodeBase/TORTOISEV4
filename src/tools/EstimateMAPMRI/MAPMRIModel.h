#ifndef _MAPMRIModel_H
#define _MAPMRIModel_H


#include "DiffusionModel.h"


using namespace Eigen;

class MAPMRIModel: public DiffusionModel<MAPImageType>
{

public:
    using EVecType=vnl_matrix_fixed< double, 3, 3 >;
    using EValType=itk::Vector<float,3> ;
    using EVecImageType=itk::Image<EVecType,3>;
    using EValImageType=itk::Image<EValType,3>;

    using Superclass=DiffusionModel<MAPImageType>;
    using OutputImageType= Superclass::OutputImageType;


public:
    void SetSmallDelta(float sd){small_delta=sd;}
    void SetBigDelta(float bd){big_delta=bd;}    
    void SetDTImg(DTImageType::Pointer di){dt_image=di;}
    void SetMAPMRIDegree(int md){MAP_DEGREE=md;}
    void SetDTIIndices(std::vector<int> di){DT_indices=di;}

    EValImageType::Pointer getEvalImage(){return eval_image;}
    void ComputeEigenImages();


    void PerformFitting();
    ImageType3D::Pointer SynthesizeDWI(vnl_vector<double> bmat_vec);

private:

    vnl_matrix<double>  bmat2q(vnl_matrix<double> cBMatrix , std::vector<int> all_indices,bool qspace=true);
    MAPType FitMAPMRI(std::vector<float> &signal,  float A0val, int order, EValType uvec, vnl_matrix<double> &qxyz,double tdiff, double reg_weight,vnl_vector<double> weights_vector);
    MatrixXd mk_ashore_basis(int order, vnl_vector<double> & uvec,vnl_matrix<double> &qxyz, bool qsp);
    vnl_matrix<double>  shore_3d_reconstruction_domain(int order, vnl_vector<double>& uvec);
    MatrixXd shore_car_phi(int nn, double uu, vnl_matrix<double> qarr);
    vnl_matrix<double> hermiteh(int nn, vnl_matrix<double> xx);


private:
    float small_delta{0}, big_delta{0};

    DTImageType::Pointer  dt_image{nullptr};
    std::vector<int>  DT_indices;

    EValImageType::Pointer eval_image{nullptr};
    EVecImageType::Pointer evec_image{nullptr};

    int MAP_DEGREE;


};



#endif
