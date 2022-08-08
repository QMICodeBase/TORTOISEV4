#include "defines.h"

#include "vnl/vnl_gamma.h"

#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "../tools/EstimateMAPMRI/MAPMRIModel.h"

typedef itk::VectorImage<float, 3>  VecImageType;



#define MPI DPI
#define TPI 6.2831855

std::vector<double> cuberoot(double c0,double c1,double c2,double c3)
{
    std::vector<double> result;
    result.resize(3);

    double unreal=-1.0e30;

    double a1=c2/c3;
    double a2=c1/c3;
    double a3=c0/c3;

    double q= (a1*a1 - 3. * a2)/9.;
    double r=(2.*a1*a1*a1 - 9. * a1 *a2 + 27. * a3)/54.;

    if(r*r< q*q*q)
    {
        double theta= acos(r/ pow(q,1.5));
        result[0]=-2. * sqrt(q) * cos(theta/3.)-a1/3.;
        result[1]=-2. * sqrt(q) * cos((theta+TPI)/3.)-a1/3.;
        result[2]=-2. * sqrt(q) * cos((theta-TPI)/3.)-a1/3.;
    }
    else
    {
        double a= -r/fabs(r) * pow((fabs(r)+sqrt(r*r-q*q*q)),1./3.);
        double b=0;
        if(a!=0)
            b=q/a;
        result[0]= (a+b) -a1/3.;
        result[1]=unreal;
        result[2]=unreal;
    }
    return result;
}


double compute_u0(MAPMRIModel::EValType &uvec)
{
    double ux2=uvec[0]*uvec[0];
    double uy2=uvec[1]*uvec[1];
    double uz2=uvec[2]*uvec[2];
    double c0 = 3. * ux2 * uy2 * uz2;
    double c1 = ux2 * uy2 + ux2 * uz2 + uy2 * uz2;
    double c2 = - ( ux2 + uy2 + uz2 );
    double c3 = -3.;


    std::vector<double> roots = cuberoot(c0,c1,c2,c3);

    std::vector<int> root_indices;
    if(roots[0]>0)
        root_indices.push_back(0);
    if(roots[1]>0)
        root_indices.push_back(1);
    if(roots[2]>0)
        root_indices.push_back(2);

    if(root_indices.size()==0)
        return -1;

    double u02=roots[root_indices[0]];
    double u0=sqrt(u02);
    return u0;
}


long long myfactorial2(int val)
{
    long long res=1;
    for(int k=val;k>=2;k--)
        res*=k;
    return res;
}

int myfactorialD(int val)
{

    int res=1;
    for(int k=val;k>=2;k-=2)
        res*=k;
    return res;
}




double gg(int mm, int nn,int rr ,int ss)
{
    double val=  sqrt((double)myfactorial2(2*mm))*sqrt((double)myfactorial2(2*nn)) * (double)myfactorial2(2*(mm+nn-rr-ss)) /
            (pow(2.,rr+ss)*(double)myfactorial2(2*(mm-rr))*(double)myfactorial2(2*(nn-ss))*(double) myfactorial2(mm+nn-rr-ss)*(double)myfactorial2(rr)*(double)myfactorial2(ss)              );
    return val;
}



double t_coefficients(int m, int n, double u, double v)
{
    if( (m+n)%2  !=0 )
        return 0;

    double sm=0;
    for(int r=0;r<=m;r+=2)
    {
        for(int s=0;s<=n;s+=2)
        {
            int mPnMrPs=m+n-r-s;


          //  double ff=  pow(-1,(r+s)/2) * pow(u,m-r) * pow(v,n-s)/pow(u*u+v*v,(mPnMrPs+1)/2.);
            double ff=  pow(-1,(r+s)/2) * pow(v,m-r) /pow(u*u+v*v,(mPnMrPs+1)/2.) * pow(u,n-s);
            sm+= ff*gg(m/2,n/2,r/2,s/2);
        }
    }

    sm/= sqrt(2*MPI);

    return sm;
}


double  shore_change_basis(VecImageType::PixelType coeffs, MAPMRIModel::EValType uvec1,double u0)
{

    int Ncoeffs = coeffs.Size();
    int n_in= Ncoeffs;

    int order;
    for(order=2;order<=10;order+=2)
    {
        int nc= (order/2+1)*(order/2+2)*(2*order+3)/6;
        if(nc==Ncoeffs)
            break;
    }

    int order1= order;
    int order2=10;

    int n_out=order2/2+1;




    std::vector<  std::vector<MAPMRIModel::EValType>  > ncoeffs;
    ncoeffs.resize(order1+1);
    for(int i=0;i<order1+1;i++)
        ncoeffs[i].resize(n_out);

    //vnl_matrix<MVectorType> ncoeffs(order1+1,n_out);

    for(int ll=0;ll<=order1;ll++)
    {
        for(int kk=0;kk<n_out;kk++)
        {
            double bb=1;
            if(kk>0)
            {
                for(int ii=0;ii<kk;ii++)
                    bb*=  1.*(2*ii+1)/(2*ii+2);
                bb= sqrt(bb);
            }
            MAPMRIModel::EValType temp;
            temp[0]= t_coefficients(ll,2*kk,uvec1[0],u0)*bb;
            temp[1]= t_coefficients(ll,2*kk,uvec1[1],u0)*bb;
            temp[2]= t_coefficients(ll,2*kk,uvec1[2],u0)*bb;
            ncoeffs[ll][kk]=temp;
        }
    }

/*
    for(int k=0;k<3;k++)
    {
        for(int j=0;j<6;j++)
        {
            for(int i=0;i<5;i++)
            {
                std::cout<<  t_coefficients(i,2*j,uvec1[k],u0)<<" "    ;
            }
            std::cout<<std::endl;
        }
            std::cout<<std::endl;
    }
*/


    double anorm=1./(8.* pow(MPI,1.5)  *uvec1[0]*uvec1[1]*uvec1[2]);
    vnl_vector<double> kks(n_out);
    vnl_matrix<double> mqpsi(n_in,n_out);
    mqpsi.fill(0);
    vnl_vector<double> knorm(n_out);


    int n1a[]={0,2,0,0,1,1,0,4,0,0,3,3,1,1,0,0,2,2,0,2,1,1,6,0,0,5,5,1,1,0,0,4,4,2,2,0,0,4,1,1,3,3,0,3,3,2,2,1,1,2,8,0,0,7,7,1,
           1,0,0,6,6,2,2,0,0,6,1,1,5,5,3,3,0,0,5,5,2,2,1,1,4,4,0,4,4,3,3,1,1,4,2,2,3,3,2,10,0,0,9,9,1,1,0,0,8,8,2,2,0,0,8,1,
           1,7,7,3,3,0,0,7,7,2,2,1,1,6,6,4,4,0,0,6,6,3,3,1,1,6,2,2,5,5,0,5,5,4,4,1,1,5,5,3,3,2,2,4,4,2,4,3,3};

    int n2a[]={0,0,2,0,1,0,1,0,4,0,1,0,3,0,3,1,2,0,2,1,2,1,0,6,0,1,0,5,0,5,1,2,0,4,0,4,2,1,4,1,3,0,3,2,1,3,1,3,2,2,0,8,0,1,0,7,0,
           7,1,2,0,6,0,6,2,1,6,1,3,0,5,0,5,3,2,1,5,1,5,2,4,0,4,3,1,4,1,4,3,2,4,2,3,2,3,0,10,0,1,0,9,0,9,1,2,0,8,0,8,2,1,8,1,3,
           0,7,0,7,3,2,1,7,1,7,2,4,0,6,0,6,4,3,1,6,1,6,3,2,6,2,5,0,5,4,1,5,1,5,4,3,2,5,2,5,3,4,2,4,3,4,3};

    int n3a[]={0,0,0,2,0,1,1,0,0,4,0,1,0,3,1,3,0,2,2,1,1,2,0,0,6,0,1,0,5,1,5,0,2,0,4,2,4,1,1,4,0,3,3,1,2,1,3,2,3,2,0,0,8,0,1,
           0,7,1,7,0,2,0,6,2,6,1,1,6,0,3,0,5,3,5,1,2,1,5,2,5,0,4,4,1,3,1,4,3,4,2,2,4,2,3,3,0,0,10,0,1,0,9,1,9,0,2,0,8,2,8,
           1,1,8,0,3,0,7,3,7,1,2,1,7,2,7,0,4,0,6,4,6,1,3,1,6,3,6,2,2,6,0,5,5,1,4,1,5,4,5,2,3,2,5,3,5,2,4,4,3,3,4};



    double fac[]={1,1,2,6,24,120,720,5040,40320,362880,3628800};
    double sqrt_fac[11];
    for(int i=0;i<11;i++)
        sqrt_fac[i]=sqrt(fac[i]);
    double dfac[]={1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025,10321920,34459425,185794560,654729075,3715891200};


    vnl_vector<double> s0mtrx_ashore(Ncoeffs,0);
    for(int i=0;i<Ncoeffs;i++)
    {
        if(n1a[i]%2==0 && n2a[i]%2==0 && n3a[i]%2==0)
        {
            s0mtrx_ashore[i] = (sqrt_fac[n1a[i]]/dfac[n1a[i]]) * (sqrt_fac[n2a[i]]/dfac[n2a[i]]) * (sqrt_fac[n3a[i]]/dfac[n3a[i]]);
        }
    }

    double S0value=0;
    for(int i=0;i<Ncoeffs;i++)
    {
        S0value+=coeffs[i]* s0mtrx_ashore[i];
    }

    vnl_vector<double> coeffs_vnl(Ncoeffs);
    for(int i=0;i<Ncoeffs;i++)
    {
        coeffs_vnl[i]= coeffs[i]/S0value;
    }




    for(int ll=0;ll<n_in;ll++)
    {
        for(int kk=0;kk<n_out;kk++)
        {
            knorm[kk]= vnl_gamma(2*kk+1.5)/(myfactorial2(2*kk)*4*MPI*MPI*u0*u0*u0);
            for(int n1=0;n1<=order2/2;n1++)
            {
                for(int n2=0;n2<=order2/2;n2++)
                {
                    for(int n3=0;n3<=order2/2;n3++)
                    {
                        if(n1+n2+n3==kk)
                        {
                            mqpsi(ll,kk)+=ncoeffs[n1a[ll]][n1][0]*ncoeffs[n2a[ll]][n2][1]*ncoeffs[n3a[ll]][n3][2];
                        }
                    }
                }
            }

        }
    }
//    std::cout<<mqpsi<<std::endl;


    vnl_vector<double>  temp= mqpsi.transpose() * coeffs_vnl;
    double sm1=0,sm2=0;
    for(int i=0;i<n_out;i++)
    {
        sm1+= temp[i]*temp[i]/knorm[i];
    }

    for(int i=0;i<Ncoeffs;i++)
    {
        sm2+= coeffs_vnl[i]*coeffs_vnl[i];
    }
    sm2*=anorm;

    double pa=sm1/sm2;

    double sin_th=sin(sqrt(1-pa));

    pa=pow(sin_th,1.2) /  (1- 3* pow(sin_th,0.4)+3*pow(sin_th,0.8));

    return sin_th;

}


double inline trf_func(double sin_th)
{
    return pow(sin_th,1.2) /  (1- 3* pow(sin_th,0.4)+3*pow(sin_th,0.8));
}


bool compute_ooo(VecImageType::PixelType &coeff, VecImageType::PixelType &coeff_iso)
{
    int Ncoeffs = coeff.Size();

    int order;
    for(order=2;order<=10;order+=2)
    {
        int nc= (order/2+1)*(order/2+2)*(2*order+3)/6;
        if(nc==Ncoeffs)
            break;
    }

    int n1a[]={0,2,0,0,1,1,0,4,0,0,3,3,1,1,0,0,2,2,0,2,1,1,6,0,0,5,5,1,1,0,0,4,4,2,2,0,0,4,1,1,3,3,0,3,3,2,2,1,1,2,8,0,0,7,7,1,
           1,0,0,6,6,2,2,0,0,6,1,1,5,5,3,3,0,0,5,5,2,2,1,1,4,4,0,4,4,3,3,1,1,4,2,2,3,3,2,10,0,0,9,9,1,1,0,0,8,8,2,2,0,0,8,1,
           1,7,7,3,3,0,0,7,7,2,2,1,1,6,6,4,4,0,0,6,6,3,3,1,1,6,2,2,5,5,0,5,5,4,4,1,1,5,5,3,3,2,2,4,4,2,4,3,3};
    int n2a[]={0,0,2,0,1,0,1,0,4,0,1,0,3,0,3,1,2,0,2,1,2,1,0,6,0,1,0,5,0,5,1,2,0,4,0,4,2,1,4,1,3,0,3,2,1,3,1,3,2,2,0,8,0,1,0,7,0,
           7,1,2,0,6,0,6,2,1,6,1,3,0,5,0,5,3,2,1,5,1,5,2,4,0,4,3,1,4,1,4,3,2,4,2,3,2,3,0,10,0,1,0,9,0,9,1,2,0,8,0,8,2,1,8,1,3,
           0,7,0,7,3,2,1,7,1,7,2,4,0,6,0,6,4,3,1,6,1,6,3,2,6,2,5,0,5,4,1,5,1,5,4,3,2,5,2,5,3,4,2,4,3,4,3};
    int n3a[]={0,0,0,2,0,1,1,0,0,4,0,1,0,3,1,3,0,2,2,1,1,2,0,0,6,0,1,0,5,1,5,0,2,0,4,2,4,1,1,4,0,3,3,1,2,1,3,2,3,2,0,0,8,0,1,
           0,7,1,7,0,2,0,6,2,6,1,1,6,0,3,0,5,3,5,1,2,1,5,2,5,0,4,4,1,3,1,4,3,4,2,2,4,2,3,3,0,0,10,0,1,0,9,1,9,0,2,0,8,2,8,
           1,1,8,0,3,0,7,3,7,1,2,1,7,2,7,0,4,0,6,4,6,1,3,1,6,3,6,2,2,6,0,5,5,1,4,1,5,4,5,2,3,2,5,3,5,2,4,4,3,3,4};
    double fac[]={1,1,2,6,24,120,720,5040,40320,362880,3628800};
    double sqrt_fac[11];
    for(int i=0;i<11;i++)
        sqrt_fac[i]=sqrt(fac[i]);
    double dfac[]={1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025,10321920,34459425,185794560,654729075,3715891200};

    vnl_vector<double> s0mtrx_ashore(Ncoeffs,0);
    for(int i=0;i<Ncoeffs;i++)
    {
        if(n1a[i]%2==0 && n2a[i]%2==0 && n3a[i]%2==0)
        {
            s0mtrx_ashore[i] = (sqrt_fac[n1a[i]]/dfac[n1a[i]]) * (sqrt_fac[n2a[i]]/dfac[n2a[i]]) * (sqrt_fac[n3a[i]]/dfac[n3a[i]]);
        }
    }
    double S0value=0;
    for(int i=0;i<Ncoeffs;i++)
    {
        S0value+=coeff[i]* s0mtrx_ashore[i];
    }
    for(int i=0;i<Ncoeffs;i++)
    {
        coeff[i]= coeff[i]/S0value;
    }


    vnl_vector<double> s0mtrx_ishore(10,0);
    for(int i=0;i<10;i++)
        s0mtrx_ishore[i] =dfac[2*i+1]/dfac[2*i];


    int nfuncs=(order/2+1)*(order/2+2)*(2*order+3)/6;

    vnl_vector<double> lls(nfuncs);
    vnl_vector<double> kks(nfuncs);
    int ind=0;
    for(int ll=0;ll<=order;ll+=2)
    {
        int kmax= (order-ll)/2;

        for(int kk=0;kk<=kmax;kk++)
        {
            lls[ind]=ll;
            kks[ind]=kk;
            ind++;

            for(int mm=1;mm<=ll;mm++)
            {
                lls[ind]=ll;
                kks[ind]=kk;
                ind++;
                lls[ind]=ll;
                kks[ind]=kk;
                ind++;
            }
        }
    }


    S0value=0;
    for(int i=0;i<Ncoeffs;i++)
    {
        if(lls[i]==0)
        {
            S0value+=coeff_iso[i]*s0mtrx_ishore[kks[i]];
        }
    }
    if(!finite(S0value))
    {
        coeff_iso.Fill(0);
        coeff.Fill(0);
        return 0;
    }
    for(int i=0;i<Ncoeffs;i++)
        coeff_iso[i]/=S0value;


    vnl_matrix<double> rep1(Ncoeffs,1,1);
    vnl_matrix<double> kks_mat(Ncoeffs,1);
    kks_mat.set_column(0,kks);
    vnl_matrix<double> ns(1,Ncoeffs);

    for(int i=0;i<Ncoeffs;i++)
        ns(0,i)=n1a[i]+n2a[i]+n3a[i];

    vnl_matrix<double> kk1= rep1 * ns;
    vnl_matrix<double> kk2= kks_mat * rep1.transpose();
    vnl_matrix<double> mat(Ncoeffs,Ncoeffs);
    for(int i=0;i<Ncoeffs;i++)
        for(int j=0;j<Ncoeffs;j++)
            mat(i,j)=   (kk1(i,j)==2*kk2(i,j));


    vnl_matrix<double> coeff_iso2(1,Ncoeffs,0);
    for(int i=0;i<Ncoeffs;i++)
        if(lls[i]==0)
            coeff_iso2(0,i)=coeff_iso[i];



    vnl_matrix<double> new_coeff_mat = coeff_iso2 * mat ;

    for(int i=0;i<Ncoeffs;i++)
        coeff_iso[i]=new_coeff_mat(0,i)*s0mtrx_ashore[i];


       // std::cout<< coeff_iso<<std::endl<<std::endl;

    return 1;
}


double dissimilarity(VecImageType::PixelType coeff, MAPMRIModel::EValType uvec, VecImageType::PixelType coeff_iso, double u0)
{
    double epsilon=0.4;
    int nbasis= coeff.GetSize();

    int n1a[]={0,2,0,0,1,1,0,4,0,0,3,3,1,1,0,0,2,2,0,2,1,1,6,0,0,5,5,1,1,0,0,4,4,2,2,0,0,4,1,1,3,3,0,3,3,2,2,1,1,2,8,0,0,7,7,1,
           1,0,0,6,6,2,2,0,0,6,1,1,5,5,3,3,0,0,5,5,2,2,1,1,4,4,0,4,4,3,3,1,1,4,2,2,3,3,2,10,0,0,9,9,1,1,0,0,8,8,2,2,0,0,8,1,
           1,7,7,3,3,0,0,7,7,2,2,1,1,6,6,4,4,0,0,6,6,3,3,1,1,6,2,2,5,5,0,5,5,4,4,1,1,5,5,3,3,2,2,4,4,2,4,3,3};
    int n2a[]={0,0,2,0,1,0,1,0,4,0,1,0,3,0,3,1,2,0,2,1,2,1,0,6,0,1,0,5,0,5,1,2,0,4,0,4,2,1,4,1,3,0,3,2,1,3,1,3,2,2,0,8,0,1,0,7,0,
           7,1,2,0,6,0,6,2,1,6,1,3,0,5,0,5,3,2,1,5,1,5,2,4,0,4,3,1,4,1,4,3,2,4,2,3,2,3,0,10,0,1,0,9,0,9,1,2,0,8,0,8,2,1,8,1,3,
           0,7,0,7,3,2,1,7,1,7,2,4,0,6,0,6,4,3,1,6,1,6,3,2,6,2,5,0,5,4,1,5,1,5,4,3,2,5,2,5,3,4,2,4,3,4,3};
    int n3a[]={0,0,0,2,0,1,1,0,0,4,0,1,0,3,1,3,0,2,2,1,1,2,0,0,6,0,1,0,5,1,5,0,2,0,4,2,4,1,1,4,0,3,3,1,2,1,3,2,3,2,0,0,8,0,1,
           0,7,1,7,0,2,0,6,2,6,1,1,6,0,3,0,5,3,5,1,2,1,5,2,5,0,4,4,1,3,1,4,3,4,2,2,4,2,3,3,0,0,10,0,1,0,9,1,9,0,2,0,8,2,8,
           1,1,8,0,3,0,7,3,7,1,2,1,7,2,7,0,4,0,6,4,6,1,3,1,6,3,6,2,2,6,0,5,5,1,4,1,5,4,5,2,3,2,5,3,5,2,4,4,3,3,4};

    double numerator=0;

    for(int na=0;na<nbasis;na++)
    {
        for(int nb=0;nb<nbasis;nb++)
        {
            numerator+= coeff[na]*coeff_iso[nb] * t_coefficients(n1a[na],n1a[nb],uvec[0],u0)*
                                                  t_coefficients(n2a[na],n2a[nb],uvec[1],u0)*
                                                  t_coefficients(n3a[na],n3a[nb],uvec[2],u0);
        }
    }

    numerator*=numerator;
    double suma2=0;
    double sumb2=0;
    for(int i=0;i<nbasis;i++)
    {
        suma2+=coeff[i]*coeff[i];
        sumb2+=coeff_iso[i]*coeff_iso[i];
    }
    double denominator=suma2*sumb2/(64*DPI*DPI*DPI*uvec[0]*uvec[1]*uvec[2]*u0*u0*u0);

    double result= sqrt(1-numerator/denominator);
    return result;
}


int main(int argc, char *argv[])
{

    if(argc<3)
    {
        std::cout<<"EstimateMAPMRI_PA full_path_to_MAPMRI_coeff_image  full_path_to_uvec_image full_path_to_MAPMRI_iso_coeff_image (optional)"<<std::endl;
        return 0;
    }

    std::string filename(argv[1]);


    typedef itk::ImageFileReader<VecImageType> VecReaderType;
    VecReaderType::Pointer reader = VecReaderType::New();
    reader->SetFileName(filename);
    reader->Update();
    VecImageType::Pointer coeff_image= reader->GetOutput();

    VecImageType::Pointer coeff_image_iso=nullptr;
    if(argc>3)
    {
        VecReaderType::Pointer reader2 = VecReaderType::New();
        reader2->SetFileName(argv[3]);
        reader2->Update();
        coeff_image_iso= reader2->GetOutput();
    }


    int Ncoeffs =coeff_image->GetNumberOfComponentsPerPixel();

    int order;
    for(order=2;order<=10;order+=2)
    {
        int nc= (order/2+1)*(order/2+2)*(2*order+3)/6;
        if(nc==Ncoeffs)
            break;
    }


    std::string filename2(argv[2]);
    typedef itk::ImageFileReader<MAPMRIModel::EValImageType> EvalReaderType;
    EvalReaderType::Pointer readere = EvalReaderType::New();
    readere->SetFileName(filename2);
    readere->Update();
    MAPMRIModel::EValImageType::Pointer eval_image= readere->GetOutput();


    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::SizeType sz;
    sz[0]=coeff_image->GetLargestPossibleRegion().GetSize()[0];
    sz[1]=coeff_image->GetLargestPossibleRegion().GetSize()[1];
    sz[2]=coeff_image->GetLargestPossibleRegion().GetSize()[2];
    ImageType3D::RegionType reg(start,sz);

    ImageType3D::PointType orig;
    orig[0]=coeff_image->GetOrigin()[0];
    orig[1]=coeff_image->GetOrigin()[1];
    orig[2]=coeff_image->GetOrigin()[2];

    ImageType3D::SpacingType spc;
    spc[0]=coeff_image->GetSpacing()[0];
    spc[1]=coeff_image->GetSpacing()[1];
    spc[2]=coeff_image->GetSpacing()[2];

    ImageType3D::DirectionType dir;
    dir.SetIdentity();
    dir(0,0)=coeff_image->GetDirection()(0,0);dir(0,1)=coeff_image->GetDirection()(0,1);dir(0,2)=coeff_image->GetDirection()(0,2);
    dir(1,0)=coeff_image->GetDirection()(1,0);dir(1,1)=coeff_image->GetDirection()(1,1);dir(1,2)=coeff_image->GetDirection()(1,2);
    dir(2,0)=coeff_image->GetDirection()(2,0);dir(2,1)=coeff_image->GetDirection()(2,1);dir(2,2)=coeff_image->GetDirection()(2,2);


    ImageType3D::Pointer PA_image = ImageType3D::New();
    PA_image->SetRegions(reg);
    PA_image->Allocate();
    PA_image->SetOrigin(orig);
    PA_image->SetSpacing(spc);
    PA_image->SetDirection(dir);

    ImageType3D::Pointer PAth_image = ImageType3D::New();
    PAth_image->SetRegions(reg);
    PAth_image->Allocate();
    PAth_image->SetOrigin(orig);
    PAth_image->SetSpacing(spc);
    PAth_image->SetDirection(dir);

    ImageType3D::Pointer PA_image_iso =nullptr;
    ImageType3D::Pointer PAth_image_iso =nullptr;
    if(coeff_image_iso)
    {
        PA_image_iso = ImageType3D::New();
        PA_image_iso->SetRegions(reg);
        PA_image_iso->Allocate();
        PA_image_iso->SetOrigin(orig);
        PA_image_iso->SetSpacing(spc);
        PA_image_iso->SetDirection(dir);

        PAth_image_iso = ImageType3D::New();
        PAth_image_iso->SetRegions(reg);
        PAth_image_iso->Allocate();
        PAth_image_iso->SetOrigin(orig);
        PAth_image_iso->SetSpacing(spc);
        PAth_image_iso->SetDirection(dir);
    }

    #pragma omp parallel for
    for(int k=0;k<(int)sz[2];k++)
    {
        ImageType3D::IndexType index;
        index[2]=k;

        for(int j=0;j<(int)sz[1];j++)
        {
            index[1]=j;
            for(int i=0;i<(int)sz[0];i++)
            {
                index[0]=i;
                VecImageType::PixelType coeff= coeff_image->GetPixel(index);
                if(coeff[0]!=0)
                {
                    MAPMRIModel::EValType uvec= eval_image->GetPixel(index);
                    double u0= compute_u0(uvec);
                    double th= shore_change_basis(coeff, uvec,u0);
                    double pa=trf_func(th);
                    PA_image->SetPixel(index,pa);
                    PAth_image->SetPixel(index,th);
                    if(coeff_image_iso)
                    {
                        VecImageType::PixelType coeff_iso= coeff_image_iso->GetPixel(index);

                        compute_ooo(coeff,coeff_iso);
                        th= dissimilarity(coeff, uvec,coeff_iso,u0);
                        pa= trf_func(th);
                        PAth_image_iso->SetPixel(index,th);
                        PA_image_iso->SetPixel(index,pa);
                    }
                }

            }
        }

    }

    {
        std::string outname = filename.substr(0,filename.find(".nii")) + std::string("_PA.nii");
        typedef itk::ImageFileWriter<ImageType3D> WrType;
        WrType::Pointer wr= WrType::New();
        wr->SetFileName(outname);
        wr->SetInput(PA_image);
        wr->Update();
    }
    {
        std::string outname = filename.substr(0,filename.find(".nii")) + std::string("_PAth.nii");
        typedef itk::ImageFileWriter<ImageType3D> WrType;
        WrType::Pointer wr= WrType::New();
        wr->SetFileName(outname);
        wr->SetInput(PAth_image);
        wr->Update();
    }



    if(coeff_image_iso)
    {
        {
        std::string outname2 = filename.substr(0,filename.find(".nii")) + std::string("_PAAI.nii");
        typedef itk::ImageFileWriter<ImageType3D> WrType;
        WrType::Pointer wr2= WrType::New();
        wr2->SetFileName(outname2);
        wr2->SetInput(PA_image_iso);
        wr2->Update();
        }
        {
        std::string outname2 = filename.substr(0,filename.find(".nii")) + std::string("_PAAIth.nii");
        typedef itk::ImageFileWriter<ImageType3D> WrType;
        WrType::Pointer wr2= WrType::New();
        wr2->SetFileName(outname2);
        wr2->SetInput(PAth_image_iso);
        wr2->Update();
        }
    }


    return EXIT_SUCCESS;

}








