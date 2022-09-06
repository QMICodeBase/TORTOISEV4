#ifndef _ISO_GW_CXX
#define _ISO_GW_CXX


#include "iso_gw.h"


void multiplyArray(std::vector <double> &a, double multiplier){
    for (int i =0; i < a.size(); i++){
        a[i] *= multiplier;
    }
}
void multiplyArray(vnl_vector <double> &a, double multiplier){
    for (int i =0; i < a.size(); i++){
        a(i) *= multiplier;
    }
}

void expArray(std::vector<double> &a, double exponent = 1){
    for (int i =0; i < a.size(); i ++){
        a[i] = std::exp(a[i]*exponent);
    }
}

void expArray(vnl_vector<double> &a, double exponent = 1){
    for (int i =0; i < a.size(); i ++){
//        double d = a[i];
        a(i) = std::exp(a(i)*exponent);
    }
}

void debugArray(vnl_vector <double> &a, std::string c =""){
    std::cout << c << std::endl;
    for (int i  = 0; i < a.size(); i++){
        std::cout << a[i] << "\t";
    }
    std::cout << std::endl;
}

void debugArray(vnl_vector <int> &a, std::string c =""){
    std::cout << c << std::endl;
    for (int i  = 0; i < a.size(); i++){
        std::cout << a[i] << "\t";
    }
    std::cout << std::endl;
}



void debugArray(std::vector <float> a, std::string c =""){
    std::cout << c << std::endl;
    for (int i  = 0; i < a.size(); i++){
        std::cout << a[i] << "\t";
    }
    std::cout << std::endl;
}

void debug2DArray(vnl_matrix <double> &a, std::string c =""){
    std::cout << c << std::endl;
    for (int r  = 0; r < a.rows(); r++){
        for (int c = 0; c < a.cols(); c ++){
            std::cout << a[r][c] << "\t";

        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
void debug2DArray(vnl_matrix_fixed <double,4,4> &a, std::string c =""){
    std::cout << c << std::endl;
    for (int r  = 0; r < a.rows(); r++){
        for (int c = 0; c < a.cols(); c ++){
            std::cout << a[r][c] << "\t";

        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}



#endif

