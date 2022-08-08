#ifndef _TORTOISEBmatrixToFSLBVecs_CXX
#define _TORTOISEBmatrixToFSLBVecs_CXX

#include "../utilities/read_bmatrix_file.h"
#include "defines.h"
#include <iostream>
#include <fstream>




int main(int argc, char* argv[])
{
    if(argc==1)
    {
        std::cout<<"Usage: tortoise_bmatrix_to_bt full_path_to_bmatrix_file"<<std::endl;
        exit(EXIT_FAILURE);
    }

    std::string bmat_name=argv[1];
    vnl_matrix<double> Bmatrix = read_bmatrix_file(bmat_name);

    int Nvols = Bmatrix.rows();
    vnl_matrix<double> BT = Bmatrix;
    BT.fill(0);


    for(int v=0;v<Nvols;v++)
    {
        vnl_vector<double> bmat_vec=Bmatrix.get_row(v);

        BT(v,0)= bmat_vec[0]*1E6;
        BT(v,1)= bmat_vec[3]*1E6;
        BT(v,2)= bmat_vec[5]*1E6;

        BT(v,3)= bmat_vec[1]*1E6/sqrt(2);
        BT(v,4)= -bmat_vec[2]*1E6/sqrt(2);
        BT(v,5)= -bmat_vec[4]*1E6/sqrt(2);
    }

    std::string out_name = bmat_name.substr(0,bmat_name.rfind(".bmtxt"))+".bt";

    std::ofstream outbtfile(out_name);
    outbtfile<<BT;
    outbtfile.close();



}








#endif
