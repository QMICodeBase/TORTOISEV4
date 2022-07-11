#ifndef _GRADCAL_H
#define _GRADCAL_H

#include "iostream"
#include <vector>
#include <string>
#include "cmath"


struct GradCoef{
    std::vector<double> gradX_coef;
    std::vector<int>Xkeys;
    std::vector<double> gradY_coef;
    std::vector<int>Ykeys;
    std::vector<double> gradZ_coef;
    std::vector<int>Zkeys;
    float R0 = 250.;
    std::string gradType;
 };


class GRADCAL{
public:
    // 2 constructors
    GRADCAL();
    GRADCAL(std::string fileName);


    // destructor
    ~GRADCAL();



    void read_grad_file(std::string fileName);
    void write_ASB_format(GradCoef gc, std::string outFileName);

    std::vector<int> get_X_key(){

        return this->grads_cal.Xkeys;
    }
    std::vector<int> get_Y_key(){
        return this->grads_cal.Ykeys;
    }
    std::vector<int> get_Z_key(){
        return this->grads_cal.Zkeys;
    }
    std::vector<double> get_X_coef(){
        return this->grads_cal.gradX_coef;
    }
    std::vector<double> get_Y_coef(){
        return this->grads_cal.gradY_coef;
    }
    std::vector<double> get_Z_coef(){
        return this->grads_cal.gradZ_coef;
    }
    GradCoef get_struct(){
        return this->grads_cal;
    }

private:
    GradCoef grads_cal;
    std::string gradFilename;
    void read_Siemens_format(std::string fileName);
    void read_GE_format(std::string  fileName);
    void read_ASB_format(std::string fileName);
    void init();

};

#endif
