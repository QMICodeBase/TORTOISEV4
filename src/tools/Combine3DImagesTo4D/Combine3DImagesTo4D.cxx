#include <string>
#include <vector>
#include "defines.h"
#include "../utilities/write_3D_image_to_4D_file.h"


int main(int argc, char *argv[])
{
    if(argc < 4)
    {
        std::cout<< "Usage: Combine3DImagesTo4D output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename "<<std::endl;
        return EXIT_FAILURE;
    }

    int Nimgs= argc-2;

    for(int i=2;i< argc;i++)
    {
        ImageType3D::Pointer img = readImageD<ImageType3D>(argv[i]);

        write_3D_image_to_4D_file<float>(img,argv[1],i-2,Nimgs);

    }


    return EXIT_SUCCESS;

}
