#ifndef _READBMATRIXFILE_CXX
#define _READBMATRIXFILE_CXX


#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include "read_bmatrix_file.h"




vnl_matrix<double> read_bmatrix_file(std::string filename)
{
    int nrows=0;

    std::ifstream infile(filename.c_str());
    std::string line;
    while (std::getline(infile, line))
    {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace),line.end());
        if(line != std::string(""))
            nrows++;

    }
    infile.close();

    vnl_matrix<double> bmat(nrows,6);

    infile.open(filename.c_str());
    infile>>bmat;
    infile.close();

    return bmat;
}

#endif

