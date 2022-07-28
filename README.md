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

TORTOISE community page: https://tortoise.nibib.nih.gov/community


# TORTOISEV4 installation

There are 3 ways you can use TORTOISEV4:
1) Download pre-compiled executables for Linux and MACOSX from   https://tortoise.nibib.nih.gov/
2) Download the Docker containers from  https://tortoise.nibib.nih.gov/
3) Compile the source code

Please note that the source code here will always be up-to-date. However, the assembled packages might take a while to be updated.

## TORTOISEV4 Source code compilation

### TORTOISEV4 Prerequisite Libraries
TORTOISE requires the following libraries to be installed beforehand:
 * ITK 5.3.0,  Boost 1.76, CUDA 11.3 (for CUDA executables), Eigen 3.3, FFTW3,  VTK 8.0.1 (only for a single executable). It also uses the nlohmann/json C++ library (https://github.com/nlohmann/json), MPFIT library from  C. Markwardt (http://cow.physics.wisc.edu/~craigm/idl/idl.html) and bet brain masking library from FSL, which are included in the distribution.

The compilation has been tested with GCC-9/G++9 and  GCC-11/G++11.  Nnewer compilers should be okay but we ran into compilation issues compiling with GCC/G++-7.  You also need Cmake for compilation. The version used for testing was Cmake-3.20.

Initial Instructions:
```
mkdir TORTOISE_installation_folder
cd TORTOISE_installation_folder
mkdir libraries
cd libraries
```

#### 0) Install LAPACK and LBLAS
sudo apt-get install libblas-dev liblapack-dev

#### 1) CUDA installation (OPTIONAL. If you have an NIVIDA GPU and want to make TORTOISE faster)

Follow the instructions at:  https://developer.nvidia.com/cuda-11.3.0-download-archive
Install CUDA to default location at /usr/local/cuda

#### 2) FFTW3 installation

For debian systems:
```
sudo apt-get update -y
sudo apt-get install -y fftw3
```

#### 3) Eigen installation 
```
sudo apt-get update -y
sudo apt install libeigen3-dev
```

#### 4) Boost installation 

```
wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
tar -xvf boost_1_76_0.tar.gz
cd boost_1_76_0
./bootstrap.sh --with-libraries=iostreams,filesystem,system,regex --prefix=/usr/local/boost176
sudo ./b2 install
cd ..
```


#### 5) ITK installation

```
wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.3rc02/InsightToolkit-5.3rc03.tar.gz
tar -xvf InsightToolkit-5.3rc03.tar.gz
mkdir InsightToolkit-5.3rc02_build
cd InsightToolkit-5.3rc02_build
cmake ../InsightToolkit-5.3rc02
make -j 16
cd ..
```

#### 5) VTK installation (Optional. only for the ComputeGlyphMaps executable)

```
wget https://gitlab.kitware.com/vtk/vtk/-/archive/v8.0.1/vtk-v8.0.1.zip
unzip  vtk-v8.0.1.zip
mkdir VTK-8.0.1_build
cd VTK-8.0.1_build
cmake ../vtk-v8.0.1
make -j 16
cd ..
```


### TORTOISEV4 Compilation

```
cd ..
git clone https://github.com/eurotomania/TORTOISEV4.git
cd TORTOISEV4/TORTOISEV4
```
For nonCUDA version:
```
cmake . -D USECUDA=0 -D USE_VTK=0 -D ITK_DIR=../../libraries/InsightToolkit-5.3rc02_build 
```

For CUDA version:
```
cmake . -D USECUDA=1 -D USE_VTK=0 -D ITK_DIR=../../libraries/InsightToolkit-5.3rc02_build 
```

For ComputeGlyphMaps executable version:
```
cmake . -D USECUDA=0 -D USE_VTK=1 -D ITK_DIR=../../libraries/InsightToolkit-5.3rc02_build -D VTK_DIR=../../libraries/VTK-8.0.1_build
```

Then,
```
make -j 16
cd ..
export PATH=${PATH}:$(pwd)/bin
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

