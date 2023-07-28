#ifndef _DRTAMAS_UTILITIES_CP_h
#define _DRTAMAS_UTILITIES_CP_h



#include "defines.h"

#include "DRTAMAS.h"


DTMatrixImageType::Pointer LogTensorImage(DTMatrixImageType::Pointer dt_img);


DTMatrixImageType::Pointer ExpTensorImage(DTMatrixImageType::Pointer dt_img);


InternalMatrixType  InterpolateAt(DTMatrixImageType::Pointer img, DTMatrixImageType::PointType pt);


vnl_matrix_fixed<double,3,3> ComputeRotationFromAffine(vnl_matrix_fixed<double,3,3> matrix);


DTMatrixImageType::Pointer ReadAndOrientTensor(std::string fname);
void OrientAndWriteTensor(DTMatrixImageType::Pointer tens,std::string nm);


DTMatrixImageType::Pointer TransformAndWriteAffineImage(DTMatrixImageType::Pointer moving_tensor,DRTAMAS::AffineTransformType::Pointer my_affine_trans,DTMatrixImageType::Pointer fixed_tensor, std::string output_nii_name);
void TransformAndWriteDiffeoImage(DTMatrixImageType::Pointer moving_tensor,DisplacementFieldType::Pointer disp_field,DTMatrixImageType::Pointer fixed_tensor, std::string output_nii_name);

vnl_matrix_fixed<double,3,3> ComputeJacobian(DisplacementFieldType::Pointer field,DisplacementFieldType::IndexType ind3 );

#endif
