#ifndef _ROTATE_BMATRIX_H
#define _ROTATE_BMATRIX_H


#include "defines.h"
#include <itkCompositeTransform.h>
#include "itkOkanQuadraticTransform.h"
#include "itkAffineTransform.h"
#include "itkEuler3DTransform.h"

using CompositeTransformType= itk::CompositeTransform<double,3 > ;
using OkanQuadraticTransformType=itk::OkanQuadraticTransform<double,3,3> ;
using AffineTransformType=itk::AffineTransform<double,3>;
using RigidTransformType= itk::Euler3DTransform<double>;



vnl_vector<double> RotateBMatrixVec(vnl_vector<double> Bmatrixvec, const vnl_matrix_fixed<double,3,3> &rotmat);

vnl_matrix<double> RotateBMatrix(const vnl_matrix<double> &Bmatrix, const vnl_matrix_fixed<double,3,3> &rotmat, const vnl_matrix_fixed<double,3,3> &dirmat );


vnl_matrix<double> RotateBMatrix(vnl_matrix<double> Bmatrix, std::vector<OkanQuadraticTransformType::Pointer> transforms,const vnl_matrix_fixed<double,3,3> &dirmat);


vnl_matrix<double> RotateBMatrix(vnl_matrix<double> Bmatrix, std::vector<CompositeTransformType::Pointer> transforms,const vnl_matrix_fixed<double,3,3> &dirmat);


#endif

