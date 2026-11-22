#ifndef _CREATEMASK_CXX
#define _CREATEMASK_CXX


#include "create_mask.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageDuplicator.h"
#if defined(__linux__)
#include "../../external_libraries/bet/Linux/betokan.h"
#elif !defined(__APPLE__)
#error "Automatic brain masking supports Linux and macOS only."
#endif

#ifdef __APPLE__
#include "itkSpatialOrientationAdapter.h"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
    // SkullStrip operates on the voxel lattice and drops obliquity in its
    // output header. Give it an explicitly cardinal copy, preserving axis
    // order and handedness, then restore the original geometry below. This
    // changes only the temporary header; no voxel resampling is performed.
    auto duplicate = itk::ImageDuplicator<ImageType3D>::New();
    duplicate->SetInputImage(img);
    duplicate->Update();
    auto afniInput = duplicate->GetOutput();
    itk::SpatialOrientationAdapter orientation;
    afniInput->SetDirection(orientation.ToDirectionCosines(
        orientation.FromDirectionCosines(img->GetDirection())));
    auto writer = itk::ImageFileWriter<ImageType3D>::New();
    writer->SetInput(afniInput);
    writer->SetFileName(input);
    writer->Update();

    // Pass arguments directly: paths containing spaces or shell characters are safe.
    std::vector<std::string> arguments = {
        "3dSkullStrip", "-input", input, "-prefix", output, "-mask_vol"
    };
    // Quote each argument for an unambiguous, copyable command in the log.
    // Execution below still passes argv directly, without invoking a shell.
    std::cout << "Running:";
    for (const auto &argument : arguments)
    {
        std::cout << " '";
        for (char character : argument)
            if (character == '\'')
                std::cout << "'\\''";
            else
                std::cout << character;
        std::cout << "'";
    }
    std::cout << std::endl;
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
            if (std::abs(mask->GetDirection()[i][j] - afniInput->GetDirection()[i][j]) > 1e-4)
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
#endif // __APPLE__

ImageType3D::Pointer create_mask(ImageType3D::Pointer img,ImageType3D::Pointer noise_img)
{
    ImageType3D::Pointer b0_mask_img=nullptr;

    if(noise_img==nullptr)
    {
#ifdef __APPLE__
        b0_mask_img=skullStripAFNI(img);
#elif defined(__linux__)
        // Preserve Linux's in-process BET implementation (fractional threshold
        // 0.3); it links libbetokan.a and does not search PATH.
        // To use AFNI on Linux instead, enable the AFNI includes/helper above
        // for Linux, call skullStripAFNI(img) here, and remove the betokan
        // include and Linux archive linkage in CMakeLists.txt. posix_spawnp
        // will find 3dSkullStrip on PATH, just as on macOS.
        // To use BET on PATH instead, implement an external-process adapter:
        // write a temporary input, launch `bet input output -m -f 0.3` with
        // posix_spawnp, read output_mask.nii[.gz], and validate the binary mask
        // and geometry. Handle FSL's output type in the child environment;
        // do not reuse AFNI's label threshold or orientation workaround.
        // Neither external alternative is enabled by this branch.
        b0_mask_img=betokan(img);
#endif
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

