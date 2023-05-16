#include <string>
#include <vector>
#include "defines.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "../utilities/extract_3Dvolume_from_4D.h"


int main(int argc, char *argv[])
{
    if(argc < 4)
    {
        std::cout<< "Usage: Combine4DImages output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename"<<std::endl;
        return EXIT_FAILURE;
    }

    int Nimgs= (argc-2);

    std::string output_name= argv[1];
    if(output_name.find(".gz")!=std::string::npos)
    {
        output_name= output_name.substr(0,output_name.rfind(".gz")) + ".nii";
    }

    fs::path output_path(output_name);
    if(output_path.parent_path()!="")
    {
        if(!fs::exists(output_path.parent_path()))
                fs::create_directories(output_path.parent_path());
    }


    int tot_Nvols=0;
    for(int ni=0;ni<Nimgs;ni++)
    {
        std::string nii_name = argv[ni+2];
        itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
        myio->SetFileName(nii_name);
        myio->ReadImageInformation();
        int Nvols= myio->GetDimensions(3);
        tot_Nvols+=Nvols;
    }

    std::cout<<"Total volumes: "<< tot_Nvols<<std::endl;


    int vols_so_far=0;
    for(int ni=0;ni<Nimgs;ni++)
    {
        std::string nii_name = argv[ni+2];
        ImageType4D::Pointer img = readImageD<ImageType4D>(nii_name) ;
        int Nvols = img->GetLargestPossibleRegion().GetSize()[3];

        for(int v=0;v<Nvols;v++)
        {
            ImageType3D::Pointer vol = extract_3D_volume_from_4D(img,v);
            write_3D_image_to_4D_file<float>(vol,output_name,vols_so_far+v,tot_Nvols);
        }
        vols_so_far+=Nvols;
    }


    return EXIT_SUCCESS;

}
