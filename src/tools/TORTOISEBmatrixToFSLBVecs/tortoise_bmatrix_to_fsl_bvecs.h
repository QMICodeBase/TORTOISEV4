#ifndef _TORTOISEBmatrixToFSLBVecs_H
#define _TORTOISEBmatrixToFSLBVecs_H


#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "vnl/vnl_matrix_fixed.h"
typedef vnl_matrix_fixed< double, 3, 3 > MatrixType;

vnl_matrix<double> tortoise_bmatrix_to_fsl_bvecs(const vnl_matrix<double> &Bmatrix, vnl_matrix<double> &bvecs)
{
    int Nvols=Bmatrix.rows();
    vnl_matrix<double> bvals(1,Nvols);

    for(int i=0;i<Nvols;i++)
    {
        vnl_vector<double> bmat_vec= Bmatrix.get_row(i);

        MatrixType bmat;
        bmat(0,0)= bmat_vec[0];
        bmat(0,1)= bmat_vec[1]/2;
        bmat(1,0)= bmat_vec[1]/2;
        bmat(0,2)= bmat_vec[2]/2;
        bmat(2,0)= bmat_vec[2]/2;
        bmat(1,1)= bmat_vec[3];
        bmat(1,2)= bmat_vec[4]/2;
        bmat(2,1)= bmat_vec[4]/2;
        bmat(2,2)= bmat_vec[5];

        vnl_symmetric_eigensystem<double> eig(bmat);

        bvals(0,i)= eig.D(2,2);
        vnl_vector<double> cbvec= eig.V.get_column(2);
        bvecs(0,i)= cbvec[0];
        bvecs(1,i)= cbvec[1];
        bvecs(2,i)= cbvec[2];
    }
    return bvals;
}








#endif
