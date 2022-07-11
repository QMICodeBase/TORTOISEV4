#ifndef _CREATEMASK_CXX
#define _CREATEMASK_CXX


#include "create_mask.h"
#include "itkBinaryBallStructuringElement.h"
#include "../../external_libraries/bet/Linux/betokan.h"
#include "itkImageDuplicator.h"


#ifdef __APPLE__
ImageType3D::Pointer betApple(ImageType3D::Pointer img)
{
    std::string DIFFPREP_loc = executable_path("DIFFPREP");
    fs::path DIFFPREP_folder_path= fs::path(DIFFPREP_loc).parent_path();
    fs::path  bet_path= ((DIFFPREP_folder_path / std::string("..")) / std::string("..")) / std::string("external_libraries/bet/Darwin/bet2");
    std::string bet_str= bet_path.string();


    fs::path list_path(list.GetListFileName());
    fs::path list_folder_path = list_path.parent_path();

    std::string home(getenv("HOME"));
    char fname[1000]={0};
    char oname[1000]={0};
    sprintf(fname,"%s/temp_%d.nii",list_folder_path.string().c_str(),vol_id);
    sprintf(oname,"%s/temp_%d_mask.nii",list_folder_path.string().c_str(),vol_id);

    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetInput(img);
    wr->SetFileName(fname);
    wr->Update();


    char cmd[2000]={0};
    sprintf(cmd,"export FSLOUTPUTTYPE=NIFTI && %s %s %s -f 0.1",bet_str.c_str(), fname,oname);
    system(cmd);


    typedef itk::ImageFileReader<ImageType3D> RdType;
    RdType::Pointer rd= RdType::New();
    rd->SetFileName(oname);
    rd->Update();
    b0_mask_img=rd->GetOutput();


    fs::remove( fs::path(std::string(oname)));
    fs::remove( fs::path(std::string(fname)));

}
#endif

ImageType3D::Pointer create_mask(ImageType3D::Pointer img,ImageType3D::Pointer noise_img)
{
    ImageType3D::Pointer b0_mask_img=nullptr;

    if(noise_img==nullptr)
    {
#ifdef __linux__
        b0_mask_img=betokan(img);
#endif

#ifdef __APPLE__
        b0_mask_img=betApple(img);
#endif
        itk::ImageRegionIteratorWithIndex<ImageType3D> it(b0_mask_img,b0_mask_img->GetLargestPossibleRegion());
        while(!it.IsAtEnd())
        {
            float b0_val = it.Get();
            if(b0_val !=0)
                it.Set(1);
            ++it;
        }
    }
    else
    {
        ImageType3D::SizeType sz= img->GetLargestPossibleRegion().GetSize();
        if(sz[2]>4)
        {
            typedef itk::ImageDuplicator<ImageType3D> DupType;
            DupType::Pointer dup =DupType::New();
            dup->SetInputImage(img);
            dup->Update();
            b0_mask_img = dup->GetOutput();

            itk::ImageRegionIteratorWithIndex<ImageType3D> it(b0_mask_img,b0_mask_img->GetLargestPossibleRegion());
            while(!it.IsAtEnd())
            {
                ImageType3D::IndexType ind=it.GetIndex();
                float noise_std= noise_img->GetPixel(ind);
                float b0_val = it.Get();
                if(b0_val < 3.75*noise_std || noise_std <1E-6 )
                    it.Set(0);
                else
                    it.Set(1);
                ++it;
            }
        }
        else
        {
            typedef itk::ImageDuplicator<ImageType3D> DupType;
            DupType::Pointer dup =DupType::New();
            dup->SetInputImage(img);
            dup->Update();
            b0_mask_img = dup->GetOutput();
            b0_mask_img->FillBuffer(1.);
        }
    }


    b0_mask_img->SetDirection(img->GetDirection());
    b0_mask_img->SetOrigin(img->GetOrigin());
    b0_mask_img->SetSpacing(img->GetSpacing());


    return b0_mask_img;


}












#endif

