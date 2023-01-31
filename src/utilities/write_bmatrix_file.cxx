#ifndef _WRITE_BMATRIX_FILE_HXX
#define _WRITE_BMATRIX_FILE_HXX

#include <vnl/vnl_matrix.h>
#include <fstream>
#include <iostream>
#include "write_bmatrix_file.h"


void write_bmatrix_file(std::string bmtxt_file,vnl_matrix<double> rotated_bmatrix)
{
    std::ofstream outfile(bmtxt_file.c_str());
    outfile<<rotated_bmatrix;
    outfile.close();
}


#endif
