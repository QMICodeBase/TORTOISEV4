#ifndef _RESAMPLE_DWIS_H
#define _RESAMPLE_DWIS_H



#include "defines.h"


ImageType3D::Pointer resample_3D_image(ImageType3D::Pointer img,std::vector<float> new_res,std::vector<float> up_factors,std::string method);


#endif

