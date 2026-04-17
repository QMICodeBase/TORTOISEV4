

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "defines.h"
#include "DRTAMAS_utilities_cp.h"

#include "../tools/ResampleDWIs/resample_dwis.h"
    

int main( int argc , char * argv[] )
{
    if(argc<5)
    {
        std::cout<<"Usage:   ResampleTensor   full_path_to_tensor_to_be_resampled  Res_x Res_y Res_z  "<<std::endl;
        return EXIT_FAILURE;
    }




    auto tensor_img = ReadAndOrientTensor(argv[1]);

    DTMatrixImageType::Pointer log_tensor_img = LogTensorImage(tensor_img);

    ImageType3D::SpacingType new_spc;
    new_spc[0]=atof(argv[2]);
    new_spc[1]=atof(argv[3]);
    new_spc[2]=atof(argv[4]);


    std::vector<float> newres_vec, dummy;
    newres_vec.push_back(new_spc[0]);
    newres_vec.push_back(new_spc[1]);
    newres_vec.push_back(new_spc[2]);

    DTMatrixImageType::Pointer output_img=nullptr;

    for(int r=0;r<3;r++)
    {
        for(int c=r;c<3;c++)
        {
            ImageType3D::Pointer img = ExtractComponentFromTensorImage(log_tensor_img,r,c);
            ImageType3D::Pointer res_img = resample_3D_image(img,newres_vec,dummy,"BSPCubic");

            if(r==0 &&c==0)
            {
                output_img= DTMatrixImageType::New();
                output_img->SetRegions(res_img->GetLargestPossibleRegion());
                output_img->Allocate();
                output_img->SetSpacing(res_img->GetSpacing());
                output_img->SetOrigin(res_img->GetOrigin());
                output_img->SetDirection(res_img->GetDirection());
            }

            itk::ImageRegionIteratorWithIndex<DTMatrixImageType> it(output_img,output_img->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ImageType3D::IndexType ind3= it.GetIndex();
                auto mat =it.Get();
                if(r==c)
                {
                    mat(r,r)= res_img->GetPixel(ind3);
                }
                else
                {
                    mat(r,c)= res_img->GetPixel(ind3);
                    mat(c,r)= res_img->GetPixel(ind3);
                }
                it.Set(mat);
            }
        }
    }

    output_img = ExpTensorImage(output_img);

    std::string currdir;
    std::string nm(argv[1]);
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else
        currdir= nm.substr(0,mypos+1);

    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_resampled.nii");

    OrientAndWriteTensor(output_img,output_name);


    return EXIT_SUCCESS;
}
