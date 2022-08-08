

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;


#include "defines.h"


int main(int argc, char*argv[])
{
    if(argc<3)    
    {
        std::cout<<"Usage: Set4DImageResolution full_path_to_4d_image full_path_to_output_name  res_x res_y res_z res_t (if origin not provided 0s will be written)"<<std::endl;
        return 0;
    }

    ImageType4D::Pointer image4D= readImageD<ImageType4D>(argv[1]);
    

    ImageType4D::SpacingType orig;
    orig.Fill(0);

    if(argc>3)
    {
        orig[0] = atof(argv[3]);
        orig[1] = atof(argv[4]);
        orig[2] = atof(argv[5]);
        orig[3] = atof(argv[6]);
    }

    image4D->SetSpacing(orig);
    


    writeImageD<ImageType4D>(image4D,argv[2]);

    return EXIT_SUCCESS;
}
