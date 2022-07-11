#include "mk_displacementMaps.h"
#include "read_3Dvolume_from_4D.h"




int main(int argc,char *argv[])
{
    if(argc<2)
    {
        std::cout<<"CreateNonlinearityDisplacementMap path_to_coefficient_or_gradcal_file path_to_base_nifti outputname is_GE (0/1. optional.default:0)"<<std::endl;
        exit(EXIT_FAILURE);
    }


    bool is_GE=false;
    if(argc>4)
        is_GE=(bool)(argv[4]);

    DisplacementFieldType::Pointer output_field= mk_displacement(argv[2], argv[1],is_GE);


    typedef itk::ImageFileWriter<DisplacementFieldType> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetInput(output_field);
    wr->SetFileName(argv[3]);
    wr->Update();


}
