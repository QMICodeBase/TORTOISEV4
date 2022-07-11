#ifndef _INIT_ISO_GW_H
#define _INIT_ISO_GW_H

#include "gradcal.h"
//#include "LISTFILE.h"


#include "itkNiftiImageIO.h"
#include "itkImportImageFilter.h"
#include "itkImageMomentsCalculator.h"
#include "itkImageDuplicator.h"
#include "itkImageRegionIteratorWithIndex.h"


typedef unsigned char MaskPixelType;
typedef itk::Image<MaskPixelType,3> MaskImageType;



struct basisFuntions{
    std::vector< vnl_matrix<double> > xbasis;
    std::vector< vnl_matrix<double> > ybasis;
    std::vector< vnl_matrix<double> > zbasis;
};


double plgndr(int ll, int mm, double x);
float factorial(int n);

double sphericalz(int ll, int mm, double phi,double zz, double theta );

// code for derivative wrt x,y,z
double dshdx(char grad, int ll, int mml, double rr, double phi,double zz, double theta);
double dshdx_X(int ll, int mml, double phi, double zz, double theta );
double dshdx_Y(int ll, int mml, double phi, double zz, double theta );
double dshdx_Z(int ll, int mml, double phi, double zz, double theta );

basisFuntions init_iso_gw(vnl_matrix<double> smat, MaskImageType::Pointer maskedIm, GradCoef E);


#endif


