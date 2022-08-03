#ifndef _TORTOISEBmatrixToFSLBVecs_CXX
#define _TORTOISEBmatrixToFSLBVecs_CXX


#include <iostream>
#include <fstream>
#include "defines.h"



int main(int argc, char* argv[])
{
    if(argc<3)
    {
        std::cout<<"Usage: FSLBVecsToTORTOISEBmatrix full_path_to_bvals_file  full_path_to_bvecs_file"<<std::endl;
        exit(EXIT_FAILURE);
    }


    std::ifstream bvecs_stream(argv[2]);
    std::ifstream bvals_stream(argv[1]);


    vnl_matrix<double> bvecs,bvals;
    if(bvecs_stream.is_open())
    {
        bvecs.read_ascii(bvecs_stream);
        bvecs_stream.close();
    }
    else
    {
        std::cerr<<"Bvecs file does not exist..."<<std::endl;
        exit(EXIT_FAILURE);
    }
    if(bvals_stream.is_open())
    {
        bvals.read_ascii(bvals_stream);
        bvals_stream.close();
    }
    else
    {
        std::cerr<<"Bvals file does not exist..."<<std::endl;
        exit(EXIT_FAILURE);
    }


    if(bvecs.rows()!=3)
        bvecs=bvecs.transpose();
    if(bvals.rows()!=1)
        bvals=bvals.transpose();

    int Nvols = bvecs.columns();

    vnl_matrix<double> Bmatrix(Nvols,6) ;

    for(int i=0;i<Nvols;i++)
    {
        vnl_matrix<double> vec= bvecs.get_n_columns(i,1);
        double nrm= sqrt(vec(0,0)*vec(0,0) + vec(1,0)*vec(1,0) + vec(2,0)*vec(2,0) );
        if(nrm > 1E-3)
        {
            vec(0,0)/=nrm;
            vec(1,0)/=nrm;
            vec(2,0)/=nrm;
        }

        vnl_matrix<double> mat = bvals(0,i) * vec * vec.transpose();
        Bmatrix(i,0)=mat(0,0);
        Bmatrix(i,1)=2*mat(0,1);
        Bmatrix(i,2)=2*mat(0,2);
        Bmatrix(i,3)=mat(1,1);
        Bmatrix(i,4)=2*mat(1,2);
        Bmatrix(i,5)=mat(2,2);
    }

    fs::path p(argv[1]);
    std::string dirname = p.parent_path().string() ;
    if(dirname=="")
        dirname=".";

    std::string bmtxt_filename=  dirname + "/" +   p.stem().string() + ".bmtxt";

    std::ofstream outbmtxtfile(bmtxt_filename);
    outbmtxtfile<<Bmatrix;
    outbmtxtfile.close();

}








#endif
