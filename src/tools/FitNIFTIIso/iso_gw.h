#ifndef _ISO_GW_H
#define _ISO_GW_H

#include "init_iso_gw.h"





void multiplyArray(std::vector <double> &a, double multiplier);


void multiplyArray(vnl_vector <double> &a, double multiplier);

void expArray(std::vector<double> &a, double exponent );

void expArray(vnl_vector<double> &a, double exponent );

void debugArray(vnl_vector <int> &a, std::string c );
void debugArray(vnl_vector <double> &a, std::string c );
void debugArray(std::vector <float> a, std::string c );



void debug2DArray(vnl_matrix <double> &a, std::string c );
void debug2DArray(vnl_matrix_fixed <double,4,4> &a, std::string c );



//void iso_gw(LISTFILE &list, basisFuntions &Basis, GradCoef E);
void iso_gw(vnl_matrix<double>Bmatrix, basisFuntions &Basis, GradCoef E, vnl_vector<double>aa);


#endif

