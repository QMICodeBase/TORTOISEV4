#include "create_mask.h"
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

namespace {
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
void expectFailure(ImageType3D::Pointer image, const char *expected)
{
    try { create_mask(image); }
    catch (const std::exception &error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "Unexpected error message");
        return;
    }
    throw std::runtime_error("Expected masking to fail");
}
}

int main()
{
    const fs::path directory = fs::temp_directory_path() / fs::unique_path("tortoise test-%%%%-%%%%");
    fs::create_directory(directory);
    const std::string oldPath = std::getenv("PATH") ? std::getenv("PATH") : "";
    const std::string oldTmp = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "";
    int result = EXIT_SUCCESS;
    try {
        const fs::path executable = directory / "3dSkullStrip";
        std::ofstream script(executable.string());
        script << "#!/bin/sh\n"
                  "[ \"$1\" = '-input' ] && [ \"$3\" = '-prefix' ] && [ \"$5\" = '-mask_vol' ] || exit 9\n"
                  "case \"$TORTOISE_TEST_MODE\" in\n"
                  " failure) exit 7 ;;\n"
                  " missing) exit 0 ;;\n"
                  " *) /bin/cp \"$2\" \"$4\" ;;\n"
                  "esac\n";
        script.close();
        chmod(executable.c_str(), 0700);
        setenv("PATH", directory.c_str(), 1);
        setenv("TMPDIR", directory.c_str(), 1);
        setenv("TORTOISE_TEST_MODE", "success", 1);

        auto image = ImageType3D::New();
        ImageType3D::SizeType size; size.Fill(4);
        image->SetRegions(size);
        image->Allocate();
        image->FillBuffer(6);
        ImageType3D::IndexType index; index.Fill(0);
        image->SetPixel(index, 3);
        auto mask = create_mask(image);
        require(mask->GetPixel(index) == 0, "AFNI boundary label must be excluded");
        index[0] = 1;
        require(mask->GetPixel(index) == 1, "AFNI interior label must become one");
        // Only the executable should remain: per-call temporary files were removed.
        require(std::distance(fs::directory_iterator(directory), fs::directory_iterator()) == 1,
                "Temporary masking files were not cleaned up");
        setenv("TORTOISE_TEST_MODE", "failure", 1);
        expectFailure(image, "3dSkullStrip failed");
        setenv("TORTOISE_TEST_MODE", "missing", 1);
        expectFailure(image, "did not produce");
        setenv("TORTOISE_TEST_MODE", "success", 1);
        image->FillBuffer(0);
        expectFailure(image, "empty brain mask");
        fs::remove(executable);
        expectFailure(image, "Cannot launch 3dSkullStrip");
        require(fs::is_empty(directory), "Failure path left temporary files behind");
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        result = EXIT_FAILURE;
    }
    setenv("PATH", oldPath.c_str(), 1);
    if (oldTmp.empty()) unsetenv("TMPDIR"); else setenv("TMPDIR", oldTmp.c_str(), 1);
    unsetenv("TORTOISE_TEST_MODE");
    boost::system::error_code error;
    fs::remove_all(directory, error);
    return result;
}
