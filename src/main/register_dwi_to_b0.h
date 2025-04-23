#ifndef _RegisterDWIToB0_H
#define _RegisterDWIToB0_H

#include "defines.h"


#include "itkOkanQuadraticTransform.h"
#include "mecc_settings.h"

using  QuadraticTransformType=itk::OkanQuadraticTransform<double,3,3>;


QuadraticTransformType::Pointer  RegisterDWIToB0(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string phase, MeccSettings *mecc_settings, bool initialize,std::vector<float> lim_arr, int vol,  QuadraticTransformType::Pointer minit_trans=nullptr ) ;

#endif
