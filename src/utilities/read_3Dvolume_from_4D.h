#ifndef _READ3DVOLUMEFROM4D_H
#define _READ3DVOLUMEFROM4D_H


#include "defines.h"

void onifti_swap_4bytes( size_t n , void *ar ) ;   /* 4 bytes at a time */


ImageType3D::Pointer read_3D_volume_from_4D(std::string fname, int vol_id);
ImageType3DBool::Pointer read_3D_volume_from_4DBool(std::string fname, int vol_id);


#endif
