#include "mk_displacementMaps.h"
#include <algorithm>
#include <ctype.h>
#include "defines.h"



int main(int argc,char *argv[])
{
    if(argc<2)
    {
        std::cout<<"CreateNonlinearityDisplacementMap path_to_coefficient_or_gradcal_file path_to_base_nifti outputname is_GE (0/1. optional.default:0)"<<std::endl;
        exit(EXIT_FAILURE);
    }


    bool is_GE=false;
    if(argc>4)
        is_GE=(bool)atoi(argv[4]);


    ImageType3D::Pointer img =readImageD<ImageType3D>(argv[2]);

    DisplacementFieldType::Pointer output_field= mk_displacement(argv[1],img ,is_GE);


    typedef itk::ImageFileWriter<DisplacementFieldType> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetInput(output_field);
    wr->SetFileName(argv[3]);
    wr->Update();


}
