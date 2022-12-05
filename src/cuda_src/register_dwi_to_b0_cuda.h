#ifndef _RegisterDWIToB0CUDA_H
#define _RegisterDWIToB0CUDA_H

#include "defines.h"
#include "itkOkanQuadraticTransform.h"
#include "mecc_settings.h"


using  TransformType=itk::OkanQuadraticTransform<double,3,3>;


TransformType::Pointer  RegisterDWIToB0_cuda(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string phase, MeccSettings *mecc_settings, bool initialize,std::vector<float> lim_arr,  TransformType::Pointer minit_trans=nullptr );

#endif
