#ifndef _ESTIMATETENSORWLLSSUB_H
#define _ESTIMATETENSORWLLSSUB_H


#include "defines.h"

DTImageType::Pointer   EstimateTensorWLLS_sub_nomm(std::vector<ImageType3D::Pointer> dwis, vnl_matrix<double> Bmatrix,std::vector<int> &DT_indices, ImageType3D::Pointer & A0_image,ImageType3D::Pointer mask_image,std::vector<ImageType3DBool::Pointer> inclusion_img);


#endif
