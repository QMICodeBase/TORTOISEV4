#ifndef SELECTBESTB0MAIN_CXX
#define SELECTBESTB0MAIN_CXX

#include "defines.h"
#include "select_best_b0.h"


int main(int argc, char * argv[])
{

    if(argc==1)
    {
        std::cout<<"Usage: SelectBestB0 NIFTI_file bvals_file output_name"<<std::endl;
        return EXIT_FAILURE;
    }

    TORTOISE t;


    ImageType4D::Pointer img =readImageD<ImageType4D>(argv[1]);
    int Nvols = img->GetLargestPossibleRegion().GetSize()[3];

    vnl_vector<double> bvals(Nvols);
    std::ifstream infileb(argv[2]);
    infileb>>bvals;
    infileb.close();

    ImageType3D::Pointer best_b0_img;

    int id = select_best_b0(img,bvals,best_b0_img);

    writeImageD<ImageType3D>(best_b0_img,argv[3]);
    return EXIT_SUCCESS;

}



#endif
