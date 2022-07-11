#ifndef MATHUTILITIES_CXX
#define MATHUTILITIES_CXX


#include "math_utilities.h"

int round50(float n2)
{
    int n= (int)n2;
    int a = (n / 50) * 50;
    int b = a + 50;
    // Return of closest of two
    return (n - a > b - n)? b : a;
}


float median(std::vector<float> &v2)
{
    std::vector<float> v=v2;
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

float average(const EigenVecType& x, const EigenVecType& w)
{
  return x.dot(w) / w.sum();
}


double ComputeResidProb(double val, float mu, float sigma,int agg_level)
{
    double norm_x= (val -mu)/sigma;
    if(agg_level==0)
    {
        norm_x-=1.4;
        norm_x/=0.6;
    }
    if(agg_level==1)
    {
        norm_x-=1.3;
        norm_x/=0.45;
    }
    if(agg_level==2)
    {
        norm_x-=1.275;
        norm_x/=0.4;
    }

    return 1- normalCDF_val(norm_x );
}




EigenVecType gaussianCDF(const EigenVecType& x)
{
    EigenVecType resp=x;
    for(int i=0;i<resp.size();i++)
        resp[i]= normalCDF_val(resp[i]);
    return resp;
}

EigenVecType log_gaussian_skewed(const EigenVecType& x,float alpha=3,float mu = 0., float sigma = 1.)
{
    EigenVecType xp = x.array() - mu;
    xp *= alpha/sigma;

    EigenVecType g_cdf= gaussianCDF(xp);

    EigenVecType res=  log_gaussian(x,mu,sigma).array() +  g_cdf.array().log();
    res= res.array() + M_LN2;// - log(sigma);
    return res;
}

EigenVecType gaussian(const EigenVecType& x, float mu = 0., float sigma = 1.)
{
    EigenVecType resp = log_gaussian(x,mu,sigma);
    resp=resp.array().exp();
    return resp;
}


EigenVecType log_gaussian(const EigenVecType& x, float mu = 0., float sigma = 1.)
{
  EigenVecType resp = x.array() - mu;
  resp /= sigma;
  resp = resp.array().square() + std::log(2*M_PI);
  resp *= -0.5f;
  resp = resp.array() - std::log(sigma);
  return resp;
}

void myfactorial(vnl_vector<double> &facts)
{
    facts[0]=1;
    for(int i=1;i<facts.size();i++)
        facts[i]=facts[i-1]*i;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


template int sgn<double>(double val);
template int sgn<float>(float val);
template int sgn<int>(int val);



#endif

