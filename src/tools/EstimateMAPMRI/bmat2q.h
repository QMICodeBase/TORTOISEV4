#ifndef _BMAT2Q_H
#define _BMAT2Q_H


#include "defines.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"


#define MPI 3.141592653589793


vnl_matrix<double>  bmat2q(vnl_matrix<double> BMatrix,std::vector<int> all_indices,double small_delta,double big_delta,bool qspace=true)
{
    vnl_matrix<double> qmat;
    qmat.set_size(3,all_indices.size());

    double delta= big_delta-small_delta/3.;

    for(int v=0;v<all_indices.size();v++)
    {
        int vol =all_indices[v];
        vnl_vector<double> bmat_vec= BMatrix.get_row(vol);

        vnl_matrix<double> bmat(3,3);
        bmat(0,0) =  bmat_vec[0];  bmat(0,1) =  bmat_vec[1]/2; bmat(0,2) =  bmat_vec[2]/2;
        bmat(1,0) =  bmat_vec[1]/2;  bmat(1,1) =  bmat_vec[3]; bmat(1,2) =  bmat_vec[4]/2;
        bmat(2,0) =  bmat_vec[2]/2;  bmat(2,1) =  bmat_vec[4]/2; bmat(2,2) =  bmat_vec[5];

        bmat= bmat / 1000.;

        vnl_symmetric_eigensystem<double> eig(bmat);
        if(eig.D(2,2)<0)
            eig.D(2,2)=0;

        double bval=eig.D(2,2);
        vnl_vector<double> bvec= eig.get_eigenvector(2);

        double qq;
        if(qspace)
            qq=sqrt(bval/delta)/(2*MPI)  ;
        else
        {
            qq=bmat_vec[0] + bmat_vec[3] + bmat_vec[5];
        }

        vnl_vector<double> qvec= qq * bvec;
        qmat.set_column(v,qvec);
    }

    return qmat;

}







#endif
