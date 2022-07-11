#ifndef _FITMAPMRI_H
#define _FITMAPMRI_H


#include "defines.h"

#include <complex>
#include <cmath>


#include "vnl/algo/vnl_qr.h"
#include "vnl/algo/vnl_cholesky.h"
#include "vnl/algo/vnl_matrix_inverse.h"

#include <Eigen/Dense>
using namespace Eigen;

#include "constrained_least_squares.h"


void myfactorial(vnl_vector<double> &facts)
{
    facts[0]=1;
    for(int i=1;i<facts.size();i++)
        facts[i]=facts[i-1]*i;
}


vnl_matrix<double> hermiteh(int nn, vnl_matrix<double> xx)
{
    vnl_matrix<double> result(nn+1,xx.columns());
    result.fill(1);
    if(nn==0)
        return result;

    for(int i=0;i<xx.columns();i++)
        result(1,i)= 2* xx(0,i);


    for(int order=1;order<nn;order++)
    {
        for(int i=0;i<xx.columns();i++)
        {
            result(order+1,i) = 2 *  (xx(0,i)* result(order,i)- order * result(order-1,i));
        }
    }

    return result;
}

MatrixXd shore_car_phi(int nn, double uu, vnl_matrix<double> qarr)
{
    vnl_matrix<double> term1(1,qarr.columns());

    for(int c=0;c<qarr.columns();c++)
    {
       term1(0,c)=exp(-(2.* (DPI*uu*qarr(0,c))*(DPI*uu*qarr(0,c))));
    }


    vnl_vector<double> facts(nn+1);
    myfactorial(facts);

    vnl_matrix<double> term2(1,nn+1);


    for(int c=0;c<nn+1;c++)
    {
       term2(0,c)=  1./   sqrt(    std::pow(2.0, c)*facts[c]        );
    }

    vnl_matrix<double> term12=  term2.transpose()*term1;

    vnl_matrix<double> temp = 2* DPI *uu* qarr;
    vnl_matrix<double> term3 = hermiteh(nn, temp );

    MatrixXd final(term3.rows(),term3.columns());

    for(int r=0;r<term3.rows();r++)
        for(int c=0;c<term3.columns();c++)
            final(r,c)= term3(r,c)* term12(r,c);

    return final;
}



MatrixXd mk_ashore_basis(int order, vnl_vector<double> & uvec,vnl_matrix<double> &qxyz, bool qsp)
{
    vnl_vector<double> uu=uvec;


    if(!qsp)
    {
        uu[0]=1./(2*DPI*uvec[0]);
        uu[1]=1./(2*DPI*uvec[1]);
        uu[2]=1./(2*DPI*uvec[2]);
    }


    MatrixXd basx= shore_car_phi(order, uu[0], qxyz.get_n_rows(0,1));
    MatrixXd basy= shore_car_phi(order, uu[1], qxyz.get_n_rows(1,1));
    MatrixXd basz= shore_car_phi(order, uu[2], qxyz.get_n_rows(2,1));

    if(!qsp)
    {
        basx*=sqrt(2*DPI)*uu[0];
        basy*=sqrt(2*DPI)*uu[1];
        basz*=sqrt(2*DPI)*uu[2];
    }


     int nnmax[]={0,6,21,49,94,160};

     int n1a[]={0,2,0,0,1,1,0,4,0,0,3,3,1,1,0,0,2,2,0,2,1,1,6,0,0,5,5,1,1,0,0,4,4,2,2,0,0,4,1,1,3,3,0,3,3,2,2,1,1,2,8,0,0,7,7,1,
         1,0,0,6,6,2,2,0,0,6,1,1,5,5,3,3,0,0,5,5,2,2,1,1,4,4,0,4,4,3,3,1,1,4,2,2,3,3,2,10,0,0,9,9,1,1,0,0,8,8,2,2,0,0,8,1,
         1,7,7,3,3,0,0,7,7,2,2,1,1,6,6,4,4,0,0,6,6,3,3,1,1,6,2,2,5,5,0,5,5,4,4,1,1,5,5,3,3,2,2,4,4,2,4,3,3};

    int n2a[]={0,0,2,0,1,0,1,0,4,0,1,0,3,0,3,1,2,0,2,1,2,1,0,6,0,1,0,5,0,5,1,2,0,4,0,4,2,1,4,1,3,0,3,2,1,3,1,3,2,2,0,8,0,1,0,7,0,
         7,1,2,0,6,0,6,2,1,6,1,3,0,5,0,5,3,2,1,5,1,5,2,4,0,4,3,1,4,1,4,3,2,4,2,3,2,3,0,10,0,1,0,9,0,9,1,2,0,8,0,8,2,1,8,1,3,
         0,7,0,7,3,2,1,7,1,7,2,4,0,6,0,6,4,3,1,6,1,6,3,2,6,2,5,0,5,4,1,5,1,5,4,3,2,5,2,5,3,4,2,4,3,4,3};

    int n3a[]={0,0,0,2,0,1,1,0,0,4,0,1,0,3,1,3,0,2,2,1,1,2,0,0,6,0,1,0,5,1,5,0,2,0,4,2,4,1,1,4,0,3,3,1,2,1,3,2,3,2,0,0,8,0,1,
         0,7,1,7,0,2,0,6,2,6,1,1,6,0,3,0,5,3,5,1,2,1,5,2,5,0,4,4,1,3,1,4,3,4,2,2,4,2,3,3,0,0,10,0,1,0,9,1,9,0,2,0,8,2,8,
         1,1,8,0,3,0,7,3,7,1,2,1,7,2,7,0,4,0,6,4,6,1,3,1,6,3,6,2,2,6,0,5,5,1,4,1,5,4,5,2,3,2,5,3,5,2,4,4,3,3,4};



    int lim= nnmax[order/2]+1;
    MatrixXd basx_sub(lim, basx.cols());
    MatrixXd basy_sub(lim, basx.cols());
    MatrixXd basz_sub(lim, basx.cols());

    for(int i=0;i<lim;i++)
    {
        basx_sub.row(i)= basx.row(n1a[i]);
        basy_sub.row(i)= basy.row(n2a[i]);
        basz_sub.row(i)= basz.row(n3a[i]);
    }

    MatrixXd result(lim,basx.cols());
    for(int r=0;r<result.rows();r++)
        for(int c=0;c<result.cols();c++)
            result(r,c)= basx_sub(r,c)* basy_sub(r,c)* basz_sub(r,c);

    if(qsp)
    {
        std::complex<double> aa(0,1);
        MatrixXcd  ns(lim,1);
        for(int i=0;i<lim;i++)
        {
            ns(i,0)= std::pow(aa,1.*(n1a[i]+n2a[i]+n3a[i]) );
        }


       MatrixXcd sgn(lim,basx.cols());
       for(int i=0;i<basx.cols();i++)
           sgn.col(i) = ns.col(0);

       for(int r=0;r<result.rows();r++)
           for(int c=0;c<result.cols();c++)
               result(r,c)= result(r,c) * (sgn(r,c).real());
    }

   return result;
}


vnl_matrix<double>  shore_3d_reconstruction_domain(int order, vnl_vector<double>& uvec)
{
    int rec_zdim=11;
    double ratio=80;

    vnl_vector<double> max_radius=uvec*sqrt(order*log(ratio));


    int rec_xdim=2*rec_zdim-1;
    int rec_ydim=rec_xdim;

    int npts=long(rec_xdim)*rec_ydim*rec_zdim;


    vnl_matrix<double> rec_xyz_arr(npts,3);
    vnl_vector<double> delta_x=max_radius / (rec_zdim-1) ;

    long cnt=0;
    for( int z=0; z< rec_zdim;z++)
    {
        for( int y=-(rec_ydim-1)/2; y<= (rec_ydim-1)/2;y++)
        {
            for( int x=-(rec_xdim-1)/2; x<= (rec_xdim-1)/2;x++)
            {

                rec_xyz_arr(cnt,0)= delta_x[0]*x;
                rec_xyz_arr(cnt,1)= delta_x[1]*y;
                rec_xyz_arr(cnt,2)= delta_x[2]*z;

                cnt++;
            }
        }
    }

    return rec_xyz_arr.transpose();
}



inline int powm1(int n)
{
    if ((n%2)==0)
        return 1;
    return -1;
}


inline int delta(int n,int m)
{
    return (n==m);
}

double fS(int n, int m,vnl_vector<double> &facts)
{
    //return 2.* pow(-1,n) * pow(DPI,3.5) * ( (m==n)*3*(2*n*n+2*n+1)  + (m==n+2)*(6+4*n)*sqrt(facts[m]/facts[n]) +  (m==n+4)* sqrt(facts[m]/facts[n])  +  (m+2==n)*(6+4*m)*sqrt(facts[n]/facts[m])  +  (m+4==n) * sqrt(facts[n]/facts[m])  ) ;

    double k=2* pow(DPI,3.5) *powm1(n);
    double a0 =3 * (2*n*n + 2*n +1) *delta(n,m);
    double sqmn = sqrt(facts[m]/facts[n]);
    double sqnm= 1./sqmn;
    double an2= 2*(2*n+3)*sqmn*delta(m,n+2);
    double an4= sqmn*delta(m,n+4);
    double am2= 2* (2*m+3) *sqnm *delta(m+2,n);
    double am4= sqnm * delta(m+4,n);

    return k * (a0 + an2 + an4 + am2 + am4);

}


double fT(int n, int m)
{
    //return pow(-1,n+1)* pow(DPI,1.5) *  (  (m==n)*(1+2*n) + (m+2==n)*sqrt(1.*n*(n-1)) + (m==n+2)*sqrt(1.*m*(m-1))  );

    double a= sqrt((m-1)*m) *delta(m-2,n);
    double b= sqrt((n-1)*n)*delta(n-2,m);
    double c= (2*n+1)*delta(m,n);

    return pow(DPI,1.5) * powm1(n+1)*(a+b+c);
}


double fU(int n, int m)
{
    return powm1(n) *delta(n,m) / (2*sqrt(DPI));
}



MAPType FitMAPMRI(std::vector<float> &signal, float A0val, int order, EValType uvec, vnl_matrix<double> &qxyz,double tdiff, double reg_weight=0,vnl_vector<double> weights_vector=vnl_vector<double>())
{

    vnl_vector<double> uu(3);
    uu[0]=sqrt(uvec[0]*2000.*tdiff);
    uu[1]=sqrt(uvec[1]*2000.*tdiff);
    uu[2]=sqrt(uvec[2]*2000.*tdiff);

    typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_type;
    vector_type sig_vec(signal.size());
    for(int i= 0; i<signal.size();i++)
        sig_vec[i]=signal[i];





    MatrixXd qmtrx=mk_ashore_basis(order,uu,qxyz,1);
    qmtrx.transposeInPlace();
    MatrixXd Hmtrx =qmtrx.transpose() * qmtrx;

    vnl_matrix<double> rmtrx2= shore_3d_reconstruction_domain(order, uu);
    MatrixXd rmtrx= mk_ashore_basis(order,uu,rmtrx2,0);
    rmtrx.transposeInPlace();


    Eigen::DiagonalMatrix<double,Eigen::Dynamic> W;
    if(weights_vector.size()!=0)
    {
        W.resize(weights_vector.size());
        for(int vv=0;vv<weights_vector.size();vv++)
            //W.diagonal()[vv]= sqrt(weights_vector[vv]);
            W.diagonal()[vv]= (weights_vector[vv]);

        qmtrx=W*qmtrx;
        sig_vec= W*sig_vec;
    }




    JacobiSVD<MatrixXd> svd(Hmtrx);
    double cond = svd.singularValues()(0)         / svd.singularValues()(svd.singularValues().size()-1);

    ICLS::Problem<double> problem;
    if(cond > 1E10)
    {
        if(weights_vector.size()==0)
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-4,1E-4);
        else
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-4,1E-4,5*qmtrx.cols());
    }
    else
    {
        if(weights_vector.size()==0)
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-10,1E-10);
        else
            problem = ICLS::Problem<double>(qmtrx,rmtrx,1E-10,1E-10,5*qmtrx.cols());

    }




    vector_type x;

    ICLS::Solver<double> solver(problem);
    int niter =solver(x,sig_vec);

    MAPType res;
    res.SetSize(x.rows());
    for(int i=0;i<x.rows();i++)
    {
        res[i]=x[i];
    }

        return res;
}





#endif
