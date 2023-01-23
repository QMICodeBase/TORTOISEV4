#ifndef MATHUTILITIES_H
#define MATHUTILITIES_H


#include <vnl/vnl_matrix.h>
#include <fstream>
#include <iostream>
#include <string>
#include "defines.h"

int round50(float n2);

void myfactorial(vnl_vector<double> &facts);

float median(std::vector<float> &v2);

float average(const EigenVecType& x, const EigenVecType& w);

EigenVecType log_gaussian_skewed(const EigenVecType& x,float alpha,float mu , float sigma );
inline double normalCDF_val(double value)
{
   return 0.5 * erfc(-value * M_SQRT1_2);
}

double ComputeResidProb(double val, float mu, float sigma,int agg_level);


bool  SameSide(vnl_vector<double> v1,  vnl_vector<double> v2, vnl_vector<double> v3,vnl_vector<double> v4, vnl_vector<double> p);
bool   PointInTetrahedron(vnl_vector<double> v1,  vnl_vector<double> v2, vnl_vector<double> v3,vnl_vector<double> v4, vnl_vector<double> p);


EigenVecType log_gaussian(const EigenVecType& x, float mu , float sigma);
template <typename T> int sgn(T val);

#endif

