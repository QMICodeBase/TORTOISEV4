#ifndef _INIT_ISO_GW_CXX
#define _INIT_ISO_GW_CXX

#include "init_iso_gw.h"
#include "cmath"
#include "defines.h"


double plgndr(int ll, int mm, double x){
    if ((ll < 0 ) | (mm < 0)  | (mm > ll)){
        std::cout << "BAD arguments in plgndr" << std::endl;
        return EXIT_FAILURE;
    }

    //Compute pmm
    double pmm = 1;
    double somx2, fact, pmmp1, pll;
    if (mm > 0){
        somx2 = std::sqrt((1 - x*x));
        fact = 1.;
        for (int i = 1; i <= mm ; i++ ){
            pmm = pmm * (-fact * somx2);
            fact+=2;
        }
    }
    if (ll == mm){
        return pmm;
    }
    else{
        // Use recursion to compute pm m+1
        pmmp1 = x*(2*mm+1) *pmm;
        if (ll == mm+1){
            return pmmp1;
        }

        else{
            for (int lm = mm+2; lm <= ll; lm++ ) {
                double lm1 = x*(2*lm-1) * pmmp1;
                double lm2 = (lm + mm -1) * pmm;
                double lm3 = (lm-mm);
                pll = (lm1- lm2)/(lm3);
                pmm = pmmp1;
                pmmp1 = pll;
            }
            return pll;
        }
   }

}

float factorial(int n) {
   if ((n==0)||(n==1))
   return 1;
   else
   return n*factorial(n-1);
}



double sphericalz(int ll, int mm, double phi,double zz, double theta = 0.){
    double result;
  //  if (zz == 0) {
  //      zz = std::cos(theta);
  //  }
    int mma = std::abs(mm);
    if (mm == 0){
        result = plgndr(ll, mm, zz);
    }
    else {
       float f1 = factorial(ll - mma);
       float f2 = factorial(ll + mma);
       double pl = plgndr(ll, mma, zz);
       double f = std::sqrt(2*f1/f2);
       result = f*pl;
       if (mm <0){
           result = std::pow(-1, mm )* result * std::sin(mma*phi)  ;
        }
       else{
            result = std::pow(-1, mm) * result * std::cos(mma*phi)  ;
        }
    }
    return result;
}

/* code for derivative wrt x */
double dshdx_X(int ll, int mml, double phi, double zz, double theta = 0.){
    int mm;
    float r1, result;
    // case 1:
    if (mml >= 0){
        mm = mml;
        if (mm == 0){
            if (ll ==1){
                return 0;
            }
            r1 = std::sqrt(ll*(ll-1)/2.);
            result = -sphericalz(ll-1, 1, phi, zz, theta);
            result = result * r1;
        }

        else{
            r1 = std::sqrt((ll + mm -1)*(ll+mm));
            result =  r1*sphericalz(ll-1, mm-1, phi, zz, theta)/2;
            if (mm == 1){
                result *= std::sqrt(2.);
            }
            if (mm <= ll-2){
                r1 = std::sqrt((ll-mm-1)*(ll-mm))/2;
                result = result - sphericalz(ll-1,mm+1,phi,zz,theta)*r1;
            }
        }
    //case 2:
    }
    else{
        mm = -mml;
        if (mm == 1){
            result =  0;
        }
        else{
            r1 = std::sqrt((ll+mm-1)*(ll+mm));
            result = r1*sphericalz(ll-1, -(mm-1), phi,zz,theta)/2;

        }
        if (mm <= ll-2){
            r1 = std::sqrt((ll-mm-1)*(ll-mm))/2;
            result -= sphericalz(ll-1,-(mm+1), phi,zz,theta)*r1;
        }

    }
    return result;
}

/* code for derivative wrt y */
double dshdx_Y(int ll, int mml,  double phi, double zz, double theta = 0. ){
    int mm;
    //2 cases
    //case 1:
    double r1, result;
    if (mml >= 0){
        mm = mml;

        if (mm == 0){
            if (ll ==1){
                return 0;
            }
            r1 = std::sqrt(ll*(ll-1)/2.);
            result = -sphericalz(ll-1, -1, phi,zz,theta);
            result = result * r1;
        }

        else{
            if (mm ==1 ){
                result =  0;
            }
            else{
                r1 = -std::sqrt((ll + mm -1)*(ll+mm));
                result = r1 * sphericalz(ll-1, -(mm-1), phi,zz,theta)/2;
            }

            if (mm <= ll-2){
                r1 = std::sqrt((ll-mm-1)*(ll-mm))/2;
                result = result - sphericalz(ll-1,-(mm+1),phi,zz,theta)*r1;
            }
        }
    }
    // case 2:
    else{
        mm = - mml;

        r1 = std::sqrt((ll+mm-1)*(ll + mm));
        result =  r1 * sphericalz(ll-1, (mm-1),phi,zz,theta)/2;
        if (mm==1)
        {
            result *=std::sqrt(2);
        }
        if (mm <= ll-2){
            r1 = (std::sqrt((ll-mm-1)*(ll-mm)))/2;
            result += r1 *sphericalz(ll-1, (mm+1), phi,zz,theta);
        }
    }

    return result;
}

/* code for derivative wrt z*/
double dshdx_Z(int ll, int mml, double phi, double zz, double theta = 0.){
    double result, r1, temp;
    //case 1
    if (mml >= 0){
        int mm = mml;
        if (ll ==mm){
            result =  0;
        }
        else{
            r1 = std::sqrt(float(ll-mm)*float(ll+mm));
            temp = sphericalz(ll-1, mm,phi,zz,theta);
            result = temp * r1;
        }
    }

    // case 2:
    else{
        int mm = -mml;
        if (ll == mm){
            result = 0;
        }
        else{
            r1 = std::sqrt(float(ll-mm)*float(ll+mm));
            temp = sphericalz(ll-1, mml, phi,zz,theta);
            result = temp*r1;
        }
    }
    return result;
}

double dshdx(char grad, int ll, int mml, double rr, double phi,double zz, double theta = 0){
    /* Compute spatial derivative of real sh basis functions
     * r^l P_lm(cos(theta) cos(|m| phi) (m >= 0)
     * r^l P_lm(cos(theta) sin(|m| phi)  (m<0)
     * Input:
     *  grad : direction of derivative (X,Y,Z)
     *  ll   : l-value of associated legendre function
     *  mm   : m-value of associated legendre function
     *  rr   : distance from the origin of point (or array of points)
     *  phi  : azimuthal angle (or array of angles)
     * Output: Value of desired derivative at input points
     * */
    double result;

    if (grad == 'X'){
        result = dshdx_X(ll, mml, phi, zz, theta);
    }
    else if (grad == 'Y'){
        result = dshdx_Y(ll, mml, phi,zz, theta);

    }
    else if (grad == 'Z'){
        result = dshdx_Z(ll, mml, phi,zz,theta);

    }
    result = result * std::pow(rr,ll-1);
    return result;
}

basisFuntions init_iso_gw(vnl_matrix<double> smat, MaskImageType::Pointer maskedIm, GradCoef E){
    itk::ImageRegionIteratorWithIndex<MaskImageType> imageIterator(maskedIm, maskedIm->GetLargestPossibleRegion());
    ImageType3D::IndexType index;
    imageIterator.GoToBegin();
    vnl_vector <double> inds4 (4,1);
    vnl_vector <double> physCoor(4);

    std::vector <int> xkeys = E.Xkeys;
    std::vector <int> ykeys = E.Ykeys;
    std::vector <int> zkeys = E.Zkeys;

    basisFuntions Basis;

    // Counting number of valid index;
    int nIndex = 0;
    while(!imageIterator.IsAtEnd()){
        if (imageIterator.Get() == 0){
            ++imageIterator;
           continue;
        }
        nIndex++;
        ++imageIterator;
    }

    int nxkey, nykey, nzkey, na;
    nxkey = xkeys.size()/2;
    nykey = ykeys.size()/2;
    nzkey = zkeys.size()/2;
    na = nxkey + nykey + nzkey;

    std::vector< vnl_matrix<double> > Xbasis, Ybasis,Zbasis;
    Xbasis.resize(nIndex);
    Ybasis.resize(nIndex);
    Zbasis.resize(nIndex);


    int temp = 0;
    imageIterator.GoToBegin();
    while(!imageIterator.IsAtEnd()){
        if (imageIterator.Get() == 0){
            ++imageIterator;
           continue;
        }

        index = imageIterator.GetIndex();
        inds4[0] = index[0];
        inds4[1] = index[1];
        inds4[2] = index[2];
        inds4[3] = 1;

        physCoor = smat * inds4;
        vnl_vector <double>xyz_r = physCoor/250;
        double x1 = xyz_r[0];
        double y1 = xyz_r[1];
        double z1 = xyz_r[2];

        double rr = std::sqrt(x1*x1 + y1 *y1 + z1*z1);
        double phi = std::atan2(y1,x1);
        double zz = z1/rr;

        vnl_matrix<double> xbasis(3,5),ybasis(3,5),zbasis(3,5);

   //     if(temp==224643)
     //       int ma=0;

        for (int i = 0; i < nxkey; i ++)
        {
            xbasis(0,i)=dshdx('X', xkeys[2*i], xkeys[2*i + 1],rr, phi, zz);
            xbasis(1,i)=dshdx('Y', xkeys[2*i], xkeys[2*i + 1],rr, phi, zz);
            xbasis(2,i)=dshdx('Z', xkeys[2*i], xkeys[2*i + 1],rr, phi, zz);
        }
        Xbasis[temp] = xbasis;

        for (int i = 0; i < nykey; i ++){

            ybasis(0,i) = dshdx('X', ykeys[2*i], ykeys[2*i + 1],rr, phi, zz);
            ybasis(1,i) = dshdx('Y', ykeys[2*i], ykeys[2*i + 1],rr, phi, zz);
            ybasis(2,i) = dshdx('Z', ykeys[2*i], ykeys[2*i + 1],rr, phi, zz);
        }
        Ybasis[temp] = ybasis;

        for (int i = 0; i < nzkey; i ++){

            zbasis(0,i)= dshdx('X', zkeys[2*i], zkeys[2*i + 1],rr, phi, zz);
            zbasis(1,i) = dshdx('Y', zkeys[2*i], zkeys[2*i + 1],rr, phi, zz);
            zbasis(2,i) = dshdx('Z', zkeys[2*i], zkeys[2*i + 1],rr, phi, zz);
        }
        Zbasis[temp] = zbasis;

        ++imageIterator;
        temp++;
    }
//    std::cout << "size of x basis: ";
//    std::cout << Xbasis.size();
    Basis.xbasis = Xbasis;
    Basis.ybasis = Ybasis;
    Basis.zbasis = Zbasis;

    return Basis;
}



#endif


