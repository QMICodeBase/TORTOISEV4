#include "synthesize_dwi_from_map_mri.h"
#include "defines.h"
#include "compute_eigen_images.h"
#include "estimate_tensor_wlls_sub.h"

typedef itk::Vector<float,6> DTType;
typedef itk::Image<DTType,3> DTImageType;

int main(int argc, char*argv[])
{

    std::string mapname(argv[1]);
    std::string listname(argv[2]);
    int vol_id = atoi(argv[3]);
    std::string maskname(argv[4]);

    LISTFILE list(listname);



    typedef itk::ImageFileReader<ImageType3D> RdType2;
    RdType2::Pointer rd2= RdType2::New();
    rd2->SetFileName(maskname);
    rd2->Update();
    ImageType3D::Pointer mask_image= rd2->GetOutput();


    std::vector<int> dummy;

    ImageType3D::Pointer A0_image=NULL;
    DTImageType::Pointer  dt_image=NULL;

    std::cout<<"Estimating the diffusion tensors..."<<std::endl;
    dt_image= EstimateTensorWLLS_sub(list,dummy,A0_image,mask_image);

    std::cout<<"Eigendecomposing the diffusion tensors..."<<std::endl;
    EValImageType::Pointer eval_image=NULL;
    EVecImageType::Pointer evec_image= ComputeEigenImages(dt_image,eval_image);



    typedef itk::ImageFileReader<MAPImageType> RdType;
    RdType::Pointer rd= RdType::New();
    rd->SetFileName(mapname);
    rd->Update();
    MAPImageType::Pointer mapmri_image= rd->GetOutput();


    ImageType3D::Pointer dwi =SynthesizeDWIFromMAPMRI(mapmri_image,A0_image,evec_image,eval_image, list.GetBmatrix().get_n_rows(vol_id,1), list.GetSmallDelta(),list.GetBigDelta() );


    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr= WrType::New();
    std::string nname = mapname.substr(0, mapname.find(".nii")) + std::string("_dwi.nii");
    wr->SetFileName( nname);
    wr->SetInput(dwi);
    wr->Update();





}

