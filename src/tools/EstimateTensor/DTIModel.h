#ifndef _DTIModel_H
#define _DTIModel_H


#include "DiffusionModel.h"

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


private:
    void EstimateTensorWLLS();


private:

    std::string fitting_mode;
};



#endif
