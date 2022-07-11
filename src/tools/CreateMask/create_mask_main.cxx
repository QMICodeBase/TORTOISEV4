#ifndef _CREATEMASKMAIN_CXX
#define _CREATEMASKMAIN_CXX

#include "../main/create_mask.h"

int main(int argc, char *argv[])
{
   if(argc==1)
   {
       std::cout<<"Usage: CreateMask input_img output_name"<<std::endl;
       return EXIT_FAILURE;
   }
   ImageType3D::Pointer img = readImageD<ImageType3D>(argv[1]);
   
   ImageType3D::Pointer mask_img = create_mask(img);
   writeImageD<ImageType3D>(mask_img, argv[2]);



}

#endif
