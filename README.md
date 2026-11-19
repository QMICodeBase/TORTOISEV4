# TORTOISEV4 
Official TORTOISE Diffusion MRI Processing Pipeline V4 Source Code and Documentation
<img src="https://tortoise.nibib.nih.gov/sites/default/files/inline-images/image1.jpeg" alt="drawing" width="400" />




# What is TORTOISE?

TORTOISE (Tolerably Obsessive registration and Tensor Optimization Indolent Software Ensemble)  is a suite of programs for for pre-processing, post-processing and analyzing diffusion MRI data. It contains C, C++, Cuda, Python programs as well as shell scripts. Begninning with V4 TORTOISE is now open-source and available to all researchers.

DISCLAIMER: TORTOISEV4 IS INCOMPATIBLE WITH PREVIOUS VERSIONS.


## Diffusion MRI Preprocessing Steps:
  * Data Import
  * Denoising
  * Gibbs Ringing Correction
  * Inter-volume motion & Eddy-currents distortion correction.
  * Intra-volume slice-to-volume alignment
  * Outlier detection & replacement
  * Signal drift correction
  * Susceptibility-induced distortion Correction
    *  Diffeomorphic registration to a T2W image
    *  Blip-up blip-down (reverse phase-encoding) correction.
  * Gradient nonlinearity correction
    * Gradwarp
    * Effects on Bmatrices as:
      * HCP style gradient deviation tensors
      * Voxelwise Bmatrices
  * DWI output aligned to an anatomical image, with customizable resolution, field of view, # voxels,  data save orientation with correct overall or voxelwise Bmatrix reorientation
  * Auatomatic quality control and reporting tools.

## Diffusion Modelling:
  * Diffusion Tensor Imaging (DTI).
    * Estimation
      * Weighted Least Squares (WLLS)
      * Non-linear least Squares (NLLS)
      * Diagonal fitting
      * Non-linear 1 full DT +  free-water compartment fitting. (data must have intermediate b=values)
      * Robust Fitting (Restore)
    * A rich set of scalar map computations
  * Mean Apparent Propagator (MAPMRI)
    * Constrained Estimation
    * Derived Scalar map estimation
  *   Conversion tools to other popular software formats

## Diffusion MRI Postprocessing Steps:
  * Diffusion Tensor based registration and atlas creation tools.


Please visit these websites for more information:

TORTOISE homepage: https://tortoise.nibib.nih.gov/

TORTOISE community page: https://discuss.afni.nimh.nih.gov/c/tortoise-message-board/9


# TORTOISEV4 installation

There are 3 ways you can use TORTOISEV4:
1) Download pre-compiled executables for Linux (any distribution) from the Releases section (on the right) on this GitHub page.
2) Download the Docker containers 
3) Compile the source code

Please note that the source code here will always be up-to-date. However, the assembled packages might take a while to be updated.

## TORTOISEV4 installation and use with WSL (recommended way for Windows)

Windows Subsystem for Linux (WSL)  enables users to call Linux executables under Windows. Therefore, it is the recommended way to run TORTOISEV4 under Linux. Here are the instructions:
1) Install WSL by following the instructions at https://learn.microsoft.com/en-us/windows/wsl/install
2) TORTOISEV4 is being developed under Ubuntu24, which is the suggested distribution.
3) Once WSL is installed, open a Windows Powershell and type "bash" . Now you are in Linux.
4) All your windows drives are mounted under /mnt
5) Follow the TORTOISEV4 Source code compilation instructions to compile and install TORTOISE under WSL
   

## TORTOISEV4 Docker installation and use (recommended way for MACOSX)
1) Download and install docker from: https://docs.docker.com/engine/install/
2) Make sure docker works with:  docker run hello-world
3) Download TORTOISEV4 Docker image:
```
docker pull eurotomania/tortoise:latest --platform linux/amd64
```
4) Make sure TORTOISE runs fine:
```
docker run eurotomania/tortoise
```
In Linux, docker might require "sudo" to run.  The above command should run the TORTOISEProcess executable and print out its help. To Run any other executable, just type its name after, as:
```
docker run eurotomania/tortoise EstimateTensor
```

To mount your local data drive to Docker:
```
docker run -v /mydatadrive:/mydatadrive eurotomania/tortoise TORTOISEProcess --up_data /mydatadrive/my.nii
```

To run the CUDA versions of the executables, you need to enable the GPUs:
```
docker run --gpus all --net=host -e DISPLAY -v /mydatadrive:/mydatadrive eurotomania/tortoise TORTOISEProcess_cuda --up_data /mydatadrive/my.nii
```


## TORTOISEV4 Source code compilation

### Native macOS CPU build (Apple Silicon)

The dependency superbuild automatically downloads checksum-pinned versions of
Boost, Eigen, FFTW, zlib, and ITK, then builds TORTOISE and runs its tests.
These instructions were validated on Apple Silicon with GCC 15. CUDA and VTK
are disabled by default.

First install the Xcode command-line tools, if needed, and wait for their
installation to finish:

```bash
xcode-select --install
```

With Homebrew already installed, install the build tools:

```bash
brew install cmake ninja gcc@15
```

After cloning the repository, run the following commands from its root
(the directory containing this README):

```bash
cmake -S SuperBuild -B build/macos-gcc15 -G Ninja \
  -DCMAKE_C_COMPILER="$(command -v gcc-15)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++-15)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTORTOISE_BUILD_JOBS=4

cmake --build build/macos-gcc15 --parallel 2
cmake --install build/macos-gcc15
```

The build command downloads and builds the dependencies, builds TORTOISE,
runs the tests, and installs the executables. The final install command also
installs the build manifest. All generated files stay under
`build/macos-gcc15`; no `sudo` is needed for the superbuild. The first build
requires internet access. If interrupted, rerun the build command to resume.

Add the installed executables to your current shell's path:

```bash
export PATH="$PWD/build/macos-gcc15/install/bin:$PWD/build/macos-gcc15/install/bin/DRTAMAS:$PATH"
```

AFNI must be installed separately for automatic brain masking. Ensure its
`3dSkullStrip` executable is on `PATH`:

```bash
command -v 3dSkullStrip
```

The local installation also requires GCC 15's runtime libraries, so keep that
compiler installation available. This is a CPU build, not a standalone macOS
application bundle. See [SuperBuild/README.md](SuperBuild/README.md) for build
options, directory layout, validation details, and using existing dependencies.
The manual dependency instructions below are an alternative to the superbuild.

### TORTOISEV4 Prerequisite Libraries
TORTOISE requires the following libraries to be installed beforehand:
 * ITK 6.0b,  Boost 1.86, CUDA 11.3 or higher (for CUDA executables), Eigen 3.3, FFTW3,  VTK 8.0.1 (only for a single executable). It also uses the nlohmann/json C++ library (https://github.com/nlohmann/json), MPFIT library from  C. Markwardt (http://cow.physics.wisc.edu/~craigm/idl/idl.html) which are included in the distribution. Automatic brain masking requires AFNI’s `3dSkullStrip` executable on `PATH` at runtime.

The compilation has been tested with GCC-9/G++9 and  GCC-11/G++11.  Nnewer compilers should be okay but we ran into compilation issues compiling with GCC/G++-7.  You also need Cmake for compilation. The version used for testing was Cmake-3.20.

Initial Instructions:
```
mkdir TORTOISE_installation_folder
cd TORTOISE_installation_folder
mkdir libraries
cd libraries
```

#### 1) CUDA installation (OPTIONAL. If you have an NIVIDA GPU and want to make TORTOISE faster)

Follow the instructions at:  https://developer.nvidia.com/cuda-11.3.0-download-archive
Install CUDA to default location at /usr/local/cuda

#### 2) FFTW3  and lz installation

For debian systems:
```
sudo apt-get update -y
sudo apt install zlib1g-dev
sudo apt-get install libfftw3-dev
```

#### 3) Eigen installation 
```
sudo apt-get update -y
sudo apt install libeigen3-dev
```

#### 4) Boost installation 

```
wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz
tar -xvf boost_1_86_0.tar.gz
cd boost_1_86_0
./bootstrap.sh --with-libraries=iostreams,filesystem,system,regex --prefix=/usr/local/boost186
sudo ./b2 install
cd ..
```


#### 5) ITK installation

```
wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v6.0b02/InsightToolkit-6.0b02.tar.gz
tar -xvf InsightToolkit-6.0b02.tar.gz
mkdir InsightToolkit-6.0b02_build
cd InsightToolkit-6.0b02_build

cmake -DITK_USE_SYSTEM_EIGEN:BOOL=ON ../InsightToolkit-6.0b02
make -j 16
cd ..
```

The sed commands are to fix a bug in MattesMutualInformationv4 metric that prevents limiting the number of threads used.

#### 5) VTK installation (Optional. only for the ComputeGlyphMaps executable)

```
wget https://www.vtk.org/files/release/9.4/VTK-9.4.2.tar.gz
tar -xvf  VTK-9.4.2.tar.gz
mkdir VTK-9.4.2_build
cd VTK-9.4.2_build
cmake  ../VTK-9.4.2
make -j 16
cd ..
```


### TORTOISEV4 Compilation

```
cd ..
git clone https://github.com/QMICodeBase/TORTOISEV4.git
cd TORTOISEV4/TORTOISEV4
```
For nonCUDA version:
```
cmake . -D USECUDA=0 -D USE_VTK=0 -D ITK_DIR=../../libraries/InsightToolkit-6.0b02_build 
```

For CUDA version:
```
cmake . -D USECUDA=1 -D USE_VTK=0 -D ITK_DIR=../../libraries/InsightToolkit-6.0b02_build
```

For ComputeGlyphMaps and WarpMesh executable version:
```
cmake . -D USECUDA=0 -D USE_VTK=1 -D ITK_DIR=../../libraries/InsightToolkit-6.0b02_build  -D VTK_DIR=../../libraries/VTK-9.4.2_build
```

Then,
```
make -j 16
export PATH=${PATH}:$(pwd)/bin
export PATH=${PATH}:$(pwd)/bin/DRTAMAS

# Add these PATH entries to your shell configuration if desired.
```

# TORTOISEV4 Usage examples

Assuming you imported your data with dcm2niix and you already have a NIFTI file for DWIs, and the corresponding json, bvecs and bvals files:

#### Simplest Usage:

This will only do (by default)  Gibbs ringing correction,  inter-volume motion and eddy-currents distortion correction.

```
TORTOISEProcess --up_data path_to_your_DWI_NIFTI_file
```

#### Turn on denoising:

```TORTOISEProcess --up_data path_to_your_DWI_NIFTI_file --denoising for_final```


#### Align the DWIs to an anatomical image (and perform b=0 -> T2W susceptibility distortion correction)

```TORTOISEProcess --up_data path_to_your_DWI_NIFTI_file --structural path_to_your_anatomical_NIFTI --denoising for_final ```

#### Bring in Reverse Phase-encoded (blip-down) data for Susceptibility Distortion Correction

```TORTOISEProcess --up_data path_to_your_main_DWI_NIFTI_file  --down_data  path_to_your_down_DWI_NIFTI_file --structural path_to_your_anatomical_NIFTI --denoising for_final ```


#### Intra-Volume Motion correction and Outlier Replacement

```TORTOISEProcess --up_data path_to_your_main_DWI_NIFTI_file  --down_data  path_to_your_down_DWI_NIFTI_file --structural path_to_your_anatomical_NIFTI --denoising for_final --s2v 1 --repol 1```

#### Correct for center frequency signal drift with a linear model

```TORTOISEProcess --up_data path_to_your_main_DWI_NIFTI_file  --down_data  path_to_your_down_DWI_NIFTI_file --structural path_to_your_anatomical_NIFTI --denoising for_final --s2v 1 --repol 1 --drift linear```

#### Give an output name, change the Output resolution, FOV , orientation

```TORTOISEProcess --up_data path_to_your_main_DWI_NIFTI_file  --down_data  path_to_your_down_DWI_NIFTI_file --structural path_to_your_anatomical_NIFTI --denoising for_final --s2v 1 --repol 1 --drift linear --output  path_to_output_NIFTI_file --output_res 1 1 1 --output_voxels 220 220 200 --output_orientation LPS ```

#### Input gradient nonlinearity information and output HCP-style grad_dev tensors

```TORTOISEProcess --up_data path_to_your_main_DWI_NIFTI_file  --down_data  path_to_your_down_DWI_NIFTI_file --structural path_to_your_anatomical_NIFTI --denoising for_final --s2v 1 --repol 1 --drift linear --output  path_to_output_NIFTI_file --output_res 1 1 1 --output_voxels 220 220 200 --output_orientation LPS  --grad_nonlin nonlinearity_coefficients_file_OR_nonlinearity_field --output_gradnonlin_Bmtxt_type grad_dev```

#### Dont'do ANY correction. Just Reorient DWIs to an anatomical image (with Bmatrix rotation)

```TORTOISEProcess --up_data path_to_your_main_DWI_NIFTI_file --structural path_to_your_anatomical_NIFTI --denoising off --gibbs 0 -c off --epi off --s2v 0 --repol 0 ```


### Automatic brain masking

`TORTOISEProcess`, `DRBUDDI`, and `CreateMask` use AFNI `3dSkullStrip` for automatic brain masking on Linux and macOS. Install AFNI separately and ensure `3dSkullStrip` is on `PATH`. The build no longer links the bundled Linux BET archive.

The adapter uses `-mask_vol` and converts AFNI interior labels (values greater than 3.01) to a binary mask. Each invocation uses a private temporary directory and checks process success, output geometry, and nonempty output. Noise-based masking for non-brain data is unchanged. This changes the brain-extraction algorithm; masks and downstream results still need comparison on representative diffusion b=0 and T2-weighted data.
