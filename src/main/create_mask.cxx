#ifndef _CREATEMASK_CXX
#define _CREATEMASK_CXX


#include "create_mask.h"
#include "itkBinaryBallStructuringElement.h"
#ifdef __linux__
#include "../../external_libraries/bet/Linux/betokan.h"
#endif
#include "itkImageDuplicator.h"


#ifdef __APPLE__
// Brain extraction on macOS. There is no arm64 build of the vendored betokan
// library and no source for it in the tree, so the mask comes from the user's own
// FSL install - FSL is separately licensed, so it cannot be vendored either.
//
// Resolution order: $TORTOISE_BET2, else `bet2` on $PATH. Two checks are
// deliberately NOT performed: the binary is not tested for arm64 (an x86-64 bet2
// runs correctly under Rosetta, so the test would reject a working install), and
// wrapper scripts are not resolved to their target (the wrapper is FSL's supported
// entry point and works standalone).
//
// This replaces a branch that had evidently never been compiled: it referenced
// `list`, `vol_id` and `b0_mask_img` from no enclosing scope, had no return
// statement, and shelled out to external_libraries/bet/Darwin/bet2, a path that
// does not exist in the tree.
ImageType3D::Pointer betApple(ImageType3D::Pointer img)
{
    std::string bet = "bet2";
    if(const char *env = getenv("TORTOISE_BET2"))
        bet = env;

    fs::path tmpdir = fs::temp_directory_path();
    char stem[64];
    sprintf(stem, "tortoise_bet_%d", (int)getpid());
    fs::path in_path  = tmpdir / (std::string(stem) + ".nii");
    fs::path out_path = tmpdir / (std::string(stem) + "_mask.nii");

    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr = WrType::New();
    wr->SetInput(img);
    wr->SetFileName(in_path.string());
    wr->Update();

    // Single-quote every interpolated path. $TORTOISE_BET2 and the temp directory are
    // user-controlled, and an ordinary install path with a space would otherwise split
    // into two arguments. Embedded single quotes are escaped the POSIX way ('\'').
    auto shq = [](const std::string &v) {
        std::string o = "'";
        for(char c : v) { if(c == '\'') o += "'\\''"; else o += c; }
        return o + "'";
    };
    // FSLOUTPUTTYPE=NIFTI so bet2 writes .nii and the reader below finds it.
    const std::string cmd = "FSLOUTPUTTYPE=NIFTI " + shq(bet) + " " +
                            shq(in_path.string()) + " " + shq(out_path.string()) + " -f 0.1";
    const int rc = system(cmd.c_str());

    if(rc != 0 || !fs::exists(out_path))
    {
        fs::remove(in_path);
        std::cerr << "TORTOISE: brain extraction failed. This build needs FSL's bet2, which is "
                     "separately licensed and must be installed by you.\n"
                     "  Looked for: " << bet << "\n"
                     "  Set TORTOISE_BET2 to its full path, or put bet2 on your PATH.\n"
                     "  FSL: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index" << std::endl;
        exit(1);
    }

    typedef itk::ImageFileReader<ImageType3D> RdType;
    RdType::Pointer rd = RdType::New();
    rd->SetFileName(out_path.string());
    rd->Update();
    ImageType3D::Pointer mask = rd->GetOutput();

    // bet2 can exit 0 and write a valid, correctly-named NIfTI of ALL ZEROS on a
    // degenerate volume; neither the exit code nor fs::exists catches that, and an empty
    // mask propagates silently to the end of the pipeline. This is NOT the kind of guard
    // CLAUDE.md 0.0 forbids: that rule governs the ported GPU kernels reproducing CUDA,
    // and this branch has no CUDA counterpart - it is the harness boundary, where
    // defence is correct (CLAUDE.md 4.5).
    size_t nonzero = 0;
    itk::ImageRegionIteratorWithIndex<ImageType3D> mit(mask, mask->GetLargestPossibleRegion());
    for(mit.GoToBegin(); !mit.IsAtEnd(); ++mit)
        if(mit.Get() != 0) nonzero++;
    if(nonzero == 0)
    {
        std::cerr << "TORTOISE: brain extraction produced an EMPTY mask (" << bet
                  << " exited 0 but every voxel is zero). Refusing to continue - every "
                     "downstream stage would silently produce garbage." << std::endl;
        fs::remove(in_path);
        fs::remove(out_path);
        exit(1);
    }

    fs::remove(in_path);
    fs::remove(out_path);
    return mask;
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

