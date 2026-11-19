#ifndef _CREATEMASK_CXX
#define _CREATEMASK_CXX


#include "create_mask.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageDuplicator.h"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <vector>

extern char **environ;

namespace {
// A private directory prevents collisions between concurrent processing runs.
struct MaskTemporaryDirectory
{
    fs::path path;
    MaskTemporaryDirectory()
    {
        std::string pattern = (fs::temp_directory_path() / "tortoise-mask-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        char *created = mkdtemp(buffer.data());
        if (!created)
            throw std::runtime_error("Cannot create skull-stripping temporary directory: " +
                                     std::string(std::strerror(errno)));
        path = created;
    }
    ~MaskTemporaryDirectory()
    {
        boost::system::error_code error;
        fs::remove_all(path, error);
    }
};

ImageType3D::Pointer skullStripAFNI(ImageType3D::Pointer img)
{
    MaskTemporaryDirectory temporary;
    std::string input = (temporary.path / "input.nii").string();
    std::string output = (temporary.path / "mask.nii").string();
    auto writer = itk::ImageFileWriter<ImageType3D>::New();
    writer->SetInput(img);
    writer->SetFileName(input);
    writer->Update();

    // Pass arguments directly: paths containing spaces or shell characters are safe.
    std::vector<std::string> arguments = {
        "3dSkullStrip", "-input", input, "-prefix", output, "-mask_vol"
    };
    std::vector<char *> argv;
    for (auto &argument : arguments)
        argv.push_back(&argument[0]);
    argv.push_back(nullptr);
    pid_t child;
    int error = posix_spawnp(&child, argv[0], nullptr, nullptr, argv.data(), environ);
    if (error)
        throw std::runtime_error("Cannot launch 3dSkullStrip; install AFNI and put it on PATH: " +
                                 std::string(std::strerror(error)));
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited == -1 && errno == EINTR);
    if (waited == -1)
        throw std::runtime_error("Cannot wait for 3dSkullStrip: " + std::string(std::strerror(errno)));
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("3dSkullStrip failed; see its output above.");
    if (!fs::exists(output))
        throw std::runtime_error("3dSkullStrip did not produce its mask NIfTI.");

    auto reader = itk::ImageFileReader<ImageType3D>::New();
    reader->SetFileName(output);
    reader->Update();
    ImageType3D::Pointer mask = reader->GetOutput();
    mask->DisconnectPipeline();
    if (mask->GetLargestPossibleRegion().GetSize() != img->GetLargestPossibleRegion().GetSize())
        throw std::runtime_error("3dSkullStrip changed the image dimensions.");
    for (unsigned int i = 0; i < 3; ++i)
    {
        if (std::abs(mask->GetSpacing()[i] - img->GetSpacing()[i]) > 1e-4 ||
            std::abs(mask->GetOrigin()[i] - img->GetOrigin()[i]) > 1e-3)
            throw std::runtime_error("3dSkullStrip changed the image geometry.");
        for (unsigned int j = 0; j < 3; ++j)
            if (std::abs(mask->GetDirection()[i][j] - img->GetDirection()[i][j]) > 1e-4)
                throw std::runtime_error("3dSkullStrip changed the image orientation.");
    }
    // AFNI documents labels 4--6 as inside the surface (step(mask - 3.01)).
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(mask, mask->GetLargestPossibleRegion());
    bool nonempty = false;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        const bool inside = it.Get() > 3.01f;
        it.Set(inside ? 1.0f : 0.0f);
        nonempty = nonempty || inside;
    }
    if (!nonempty)
        throw std::runtime_error("3dSkullStrip produced an empty brain mask.");
    return mask;
}
} // namespace

ImageType3D::Pointer create_mask(ImageType3D::Pointer img,ImageType3D::Pointer noise_img)
{
    ImageType3D::Pointer b0_mask_img=nullptr;

    if(noise_img==nullptr)
    {
        b0_mask_img=skullStripAFNI(img);
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

