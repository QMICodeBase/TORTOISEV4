#ifndef _DTIModel_H
#define _DTIModel_H


#include "DiffusionModel.h"


struct vars_struct
{
  vnl_matrix<double> *Bmat;
  vnl_vector<double>  *signal;
  bool useWeights;
  vnl_vector<double> *weights;
};



class DTIModel: public DiffusionModel<DTImageType>
{
public:    
    using Superclass=DiffusionModel<DTImageType>;
    using OutputImageType= Superclass::OutputImageType;

public:
    DTIModel(){};
    ~DTIModel(){};


    void SetFittingMode(std::string s){fitting_mode=s;}


    void PerformFitting();
    ImageType3D::Pointer SynthesizeDWI(vnl_vector<double> bmat_vec);
    ImageType3D::Pointer getCSImg(){return CS_img;}


private:
    void EstimateTensorWLLS();
    void EstimateTensorNLLS();


private:

    std::string fitting_mode;
    ImageType3D::Pointer CS_img{nullptr};
};



#endif
