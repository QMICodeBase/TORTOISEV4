# TORTOISEProcess and TORTOISEProcess_cuda

  * These two are main pipeline executables.
  * The version with "_cuda" requires a CUDA installation and uses both the CPU and the GPU for inter-volume motion and eddy-currents distortion correction and the GPU for susceptibility distortion correction.
  * The CUDA version is about 100x faster than the CPU version for EPI distortion correction. For this reason, the optimizer is allowed to do a deeper search and the results might be slightly better than the CPU one.
  * Besides this difference, the two should produce near identical results.

# Settings
  * Except one setting (the input DWI data), all other settings are optional and give users more control over their processing based on their data types and needs.

# Input Settings
    -u, --up_data 
          Full path to the input UP DWI NIFTI file to be corrected. (REQUIRED. The only required parameteter.) 
    --up_json 
          Full path to the JSON file for the up  data. Can be omitted if the file basename is identical to the NIFTI image. Phase encoding direction, k-space coverage, slice timings, diffusion times will be read from this file.

     -d, --down_data 
          Full path to the input DOWN NIFTI file to be corrected. 

     -s, --structural 
          Full path to the structural/anatomical image files. Can provide more than one. These will be used for EPI distortion 
          correction. SO NO T1W images here. 

     --grad_nonlin gradnonlin_file
                   gradnonlin_file[is_GE,warp_dim]
                   example2: coeffs.grad[0,3D]
                   example3: field.nii[1,2D]
          Gradient Nonlinearity information file. Can be in ITK displacement field format, TORTOISE coefficients .gc format, GE 
          coefficients gw_coils format or Siemens coefficients .coeffs format. If it is GE, it should be specified in brackets. If 
          1D or 2D gradwarp is desired, it should be specified. Default:3D 

     --ub 
          Full path to the input UP bval file. (OPTIONAL. If not present, NIFTI file's folder is searched.) 

     --uv 
          Full path to the input UP bvec file. (OPTIONAL. If not present, NIFTI file's folder is searched.) 

     --db 
          Full path to the input DOWN bval file. (OPTIONAL. If not present, NIFTI file's folder is searched.) 

     --dv 
          Full path to the input DOWN bvec file. (OPTIONAL. If not present, NIFTI file's folder is searched.) 

     -t, --temp_folder 
          Temporary processing folder (string). If not provided, a temp folder will be created in the subfolder of the UP data. 

     -r, --reorientation 
          Full path to the structural/anatomical image file for final reorientation. Can have any contrast. If not provided, the 
          first image in the structural image list will be used. 

     --flipX 
          Flip X gradient? Boolean (0/1). Optional. Default:0 

     --flipY 
          Flip Y gradient? Boolean (0/1). Optional. Default:0 

     --flipZ 
          Flip Z gradient? Boolean (0/1). Optional. Default:0 

     --big_delta 
          Big Delta. In case it is not in JSON file and high b-value processing is needed. Default:0 

     --small_delta 
          Small Delta. In case it is not in JSON file and high b-value processing is needed. Default:0 

     --b0_mask_img 
          File name for the b=0 mask image. Optional. For in-vivo human brain, this is not needed. For other organs, or animal 
          data, please provide it. 

