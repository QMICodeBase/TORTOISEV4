# TORTOISEV4 
Official TORTOISE Diffusion MRI Processing Pipeline V4 Source Code and Documentation
<img src="https://tortoise.nibib.nih.gov/sites/default/files/inline-images/image1.jpeg" alt="drawing" width="400" />




# What is TORTOISE?

TORTOISE (Tolerably Obsessive registration and Tensor Optimization Indolent Software Ensemble)  is a suite of programs for for pre-processing, post-processing and analyzing diffusion MRI data. It contains C, C++, Cuda, Python programs as well as shell scripts. Begninning with V4 TORTOISE is now open-source and available to all researchers.


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

## TORTOISEV4 Source code compilation

### TORTOISEV4 Prerequisite Libraries
TORTOISE requires the following libraries to be installed beforehand:
 * ITK 5.0.1,  Boost 1.76, CUDA 11.3 (for CUDA executables), Eigen 3.3, FFTW3,  VTK 8.0.1 (only for a single executable)

The compilation has been tested with GCC-9, G++9  however older or newer compilers should be okay. You also need Cmake for compilation.

Initial Instructions:
```
mkdir TORTOISE_installation_folder
cd TORTOISE_installation_folder
mkdir libraries
cd libraries
```


#### 1) CUDA installation (OPTIONAL. If you have an NIVIDA GPU and want to kae TORTOISE faster)

Follow the instructions at:  https://developer.nvidia.com/cuda-11.3.0-download-archive
Install CUDA to default location at /usr/local/cuda

#### 2) FFTW3 installation

For debian systems:
```
sudo apt-get update -y
sudo apt-get install -y fftw3
```

#### 3)Eigen installation 
```
sudo apt-get update -y
sudo apt install libeigen3-dev
```

#### 4) Boost installation 

```
wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
tar -xvf boost_1_76_0.tar.gz
cd boost_1_76_0
/bootstrap.sh --with-libraries=iostreams,filesystem,system,regex --prefix=/usr/local/boost176
sudo ./b2 install
```


#### 5) ITK installation

```
wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.0.1/InsightToolkit-5.0.1.tar.gz
tar -xvf InsightToolkit-5.0.1.tar.gz
mkdir InsightToolkit-5.0.1_build
cd InsightToolkit-5.0.1_build
cmake ../InsightToolkit-5.0.1
make -j 16
```




