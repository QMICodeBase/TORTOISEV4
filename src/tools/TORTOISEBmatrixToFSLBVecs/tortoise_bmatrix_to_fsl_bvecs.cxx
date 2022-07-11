#ifndef _TORTOISEBmatrixToFSLBVecs_CXX
#define _TORTOISEBmatrixToFSLBVecs_CXX

#include "../utilities/read_bmatrix_file.h"
#include "tortoise_bmatrix_to_fsl_bvecs.h"
#include <iostream>
#include <fstream>




int main(int argc, char* argv[])
{
    if(argc==1)
    {
        std::cout<<"Usage: TORTOISEBmatrixToFSLBVecs full_path_to_bmatrix_file"<<std::endl;
        exit(0);
    }

    vnl_matrix<double> Bmatrix = read_bmatrix_file(std::string(argv[1]));
    int Nvols=Bmatrix.rows();
    vnl_matrix<double> bvecs(3,Nvols);

    vnl_matrix<double> bvals =tortoise_bmatrix_to_fsl_bvecs(Bmatrix,bvecs);
    std::string bmtxt_filename(argv[1]);
    std::string bvecs_fname=   bmtxt_filename.substr(0,bmtxt_filename.find(".bmtxt")) + std::string(".bvecs");
    std::string bvals_fname=   bmtxt_filename.substr(0,bmtxt_filename.find(".bmtxt")) + std::string(".bvals");


    std::ofstream bvecs_file(bvecs_fname.c_str());
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<Nvols;j++)
        {
            if(bvals(0,j)==0)
                bvecs_file<<"0 ";
            else
                bvecs_file<< bvecs(i,j)<< " ";
        }
        bvecs_file<<std::endl;
    }
    bvecs_file.close();


    std::ofstream bvals_file(bvals_fname.c_str());


        for(int j=0;j<Nvols;j++)
        {
            bvals_file<< bvals(0,j)<< " ";
        }
    bvals_file.close();
}








#endif
