#ifndef _CREATEMASK_H
#define _CREATEMASK_H


#include "defines.h"



#ifdef __APPLE__
ImageType3D::Pointer betApple(ImageType3D::Pointer img);
#endif

ImageType3D::Pointer create_mask(ImageType3D::Pointer img,ImageType3D::Pointer noise_img=nullptr);



#endif

