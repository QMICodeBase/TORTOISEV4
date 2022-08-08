#include "dwi_denoise.h"


#include "defines.h"

int main(int argc, char *argv[])
{
    if(argc<3)
    {
        std::cout<< "Usage:   DwiDenoise input_image name_of_output_dwi  name_of_output_noise_img  denoising_kernel_diameter(int. optional)" <<std::endl;
        return 0;
    }

    int desired_extent=0;
    if(argc>=5)
        desired_extent = atoi(argv[4]);


    ImageType4D::Pointer img4D= readImageD<ImageType4D>(argv[1]);


    ImageType3D::Pointer noise_img=nullptr;
    double noise_mean;
    ImageType4D::Pointer output_img = DWIDenoise(img4D,noise_img,noise_mean,true, desired_extent);
    std::cout<< "Average noise std: " <<noise_mean<<std::endl;


    writeImageD<ImageType4D>(output_img,argv[2]);
    writeImageD<ImageType3D>(noise_img,argv[3]);
}


