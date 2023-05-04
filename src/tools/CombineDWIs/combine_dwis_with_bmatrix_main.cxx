#include <string>
#include <vector>




#include "combine_dwis_with_bmatrix.h"

int main(int argc, char *argv[])
{
    if(argc < 4)
    {
        std::cout<< "Usage: CombineDWIsWithBMatrix output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename"<<std::endl;
        return EXIT_FAILURE;
    }


    int Nimgs= argc-2;

    std::string output_name= argv[1];
    if(output_name.find(".gz")!=std::string::npos)
    {
        output_name= output_name.substr(0,output_name.rfind(".gz")) + ".nii";
    }   

    fs::path output_path(output_name);
    std::string pp=output_path.parent_path().string();
    if(pp=="")
        pp="./";
    if(!fs::exists(pp))
            fs::create_directories(pp);

    std::vector<std::string> nii_names;
    for(int ni=0;ni<Nimgs;ni++)
    {
        nii_names.push_back(argv[ni+2]);
    }

    CombineDWIsWithBMatrix(nii_names,output_name);



    return EXIT_SUCCESS;

}
