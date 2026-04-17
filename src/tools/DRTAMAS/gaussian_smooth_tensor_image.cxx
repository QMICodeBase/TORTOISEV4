

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "defines.h"
#include "DRTAMAS_utilities_cp.h"

#include "itkDiscreteGaussianImageFilter.h"
    

int main( int argc , char * argv[] )
{
    if(argc<3)    
    {
        std::cout<<"Usage:   GaussianSmoothTensorImage   full_path_to_tensor_to_be_smoothed  gaussian_kernel_sigma"<<std::endl;
        return EXIT_FAILURE;
    }

    auto tensor_img = ReadAndOrientTensor(argv[1]);
    double sigma= atof(argv[2]);

    DTMatrixImageType::Pointer log_tensor_img = LogTensorImage(tensor_img);

    for(int r=0;r<3;r++)
    {
        for(int c=r;c<3;c++)
        {
            ImageType3D::Pointer img = ExtractComponentFromTensorImage(log_tensor_img,r,c);

            typedef itk::DiscreteGaussianImageFilter<ImageType3D, ImageType3D> ImageSmoothingFilterType;
            ImageSmoothingFilterType::Pointer SmoothingFilterf = ImageSmoothingFilterType::New();
            SmoothingFilterf->SetUseImageSpacingOn();
            SmoothingFilterf->SetVariance( sigma*sigma );
            SmoothingFilterf->SetMaximumError( 0.01 );
            SmoothingFilterf->SetInput( img );
            SmoothingFilterf->Update();
            ImageType3D::Pointer smooth_img = SmoothingFilterf->GetOutput();

            itk::ImageRegionIteratorWithIndex<DTMatrixImageType> it(log_tensor_img,log_tensor_img->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ImageType3D::IndexType ind3= it.GetIndex();
                auto mat =it.Get();
                if(r==c)
                {
                    mat(r,r)= smooth_img->GetPixel(ind3);
                }
                else
                {
                    mat(r,c)= smooth_img->GetPixel(ind3);
                    mat(c,r)= smooth_img->GetPixel(ind3);
                }
                it.Set(mat);
            }
        }
    }

    DTMatrixImageType::Pointer smooth_tensor_img = ExpTensorImage(log_tensor_img);

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
    std::string output_name=currdir + basename + std::string("_SMTH.nii");

    OrientAndWriteTensor(smooth_tensor_img,output_name);


    return EXIT_SUCCESS;
}
