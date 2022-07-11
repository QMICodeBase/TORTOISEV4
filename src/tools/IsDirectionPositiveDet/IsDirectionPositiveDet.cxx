#include "defines.h"


int main(int argc, char *argv[])
{
    if(argc==1)
    {
        std::cout<<"Usage: IsDirectionPositiveDet nifti_image"<<std::endl;
        std::cout<<"Returns 0 or 1 to command line"<<std::endl;
        return EXIT_FAILURE;
    }


    ImageType4D::Pointer img = readImageD<ImageType4D>(argv[1]);

    float det = vnl_determinant( img->GetDirection().GetVnlMatrix());
    std::cout<< "Img det: "<<det<<std::endl;

    if(fabs(det-1)<0.0001)
    {
        return 1;
    }
    else
    {
        return 0;
    }




}
