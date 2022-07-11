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
