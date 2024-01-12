# TORTOISEProcess and TORTOISEProcess_cuda

  * These two are main pipeline executables.
  * The version with "_cuda" requires a CUDA installation and uses both the CPU and the GPU for inter-volume motion and eddy-currents distortion correction and the GPU for susceptibility distortion correction.
  * The CUDA version is about 100x faster than the CPU version for EPI distortion correction. For this reason, the optimizer is allowed to do a deeper search and the results might be slightly better than the CPU one.
  * Besides this difference, the two should produce near identical results.

# Command Line Parameters
  * Except one setting (the input DWI data), all other settings are optional and give users more control over their processing based on their data types and needs.

## Input Settings
    -u, --up_data 
          Full path to the input UP DWI NIFTI file to be corrected. (REQUIRED. The only required parameteter.) 
    --up_json 
          Full path to the JSON file for the up  data. Can be omitted if the file basename is identical to the NIFTI image. Phase encoding direction, k-space coverage, slice timings, diffusion times will be read from this file.

     -d, --down_data 
          Full path to the input DOWN NIFTI file to be corrected. 

     -s, --structural 
          Full path to the structural/anatomical image files. Can provide more than one. These will be used for EPI distortion correction. SO NO T1W images here. THESE IMAGES SHOULD NOT HAVE ANY SUSCEPTIBILITY DISTORTIONS.

     --grad_nonlin gradnonlin_file
                   gradnonlin_file[is_GE,warp_dim]
                   example2: coeffs.grad[0,3D]
                   example3: field.nii[1,2D]
          Gradient Nonlinearity information file. Can be in ITK displacement field format, TORTOISE coefficients .gc format, GE coefficients gw_coils format or Siemens coefficients .coeffs format. If it is GE, it should be specified in brackets. If 1D or 2D gradwarp is desired, it should be specified. Default:3D 

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
          Full path to the structural/anatomical image file for final reorientation. Can have any contrast. If not provided, the first image in the structural image list will be used. 

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
          File name for the b=0 mask image. Optional. For in-vivo human brain, this is not needed. For other organs, or animal data, please provide it. 

## RUN Settings
     --step import: Starts from big-bang, i.e. very beginning
            denoising: Starts from denoising, skips data check and copy.
            gibbs: Starts from gibbs ringing correction, skips data check/copy and denoising.
            motioneddy: Starts from motion&eddy correction. Assumes all previous steps are done. 
            drift: Starts with signal drift correction. Assumes all previous steps are done.  
            epi: Starts from susceptibility distortion correction. Assumes all previous steps are done.  
            StructuralAlignment: Starts with aligning DWIs to the structural image. Assumes all previous steps are done.  
            finaldata: Writes the final data assuming all the previous steps are already performed.  
          
          The start step for the processing in case a previous run has crashed or in case the user wants to embed external tools. Can be (increasing order): Import Denoising Gibbs MotionEddy Drift EPI StructuralAlignment FinalData. Default:Import 

     --do_QC 
          Perform quality control steps and generate reports? (boolean). Default: 1. 

     --remove_temp 
          Remove the temp folder after processing? (boolean). Default: 0 so the folder is kept. 

## Denoising Settings
     --denoising off: DWI denoising not performed. However, noise variance still be estimated with this method.
                 for_reg: DEFAULT. Denoised DWIs are used for registration but the final outputs are the transformed versions of the original data without any denoising.
                 for_final: Final output is also based on denoised DWIs.
          DWI denoising application (string). (J. Veraart, E. Fieremans, and D.S. Novikov Diffusion MRI noise mapping using random  matrix theory. Magn. Res. Med., 2016). Default:for_reg 

     --denoising_kernel_size 
          Denoising kernel diameter (int). If 0 or not provided, the kernel diameter is automatically estimated from the data based on its size. Default:0 

## Gibbs Ringing Correction Settings
    --gibbs 
          Gibbs ringing correction of DWIs (boolean). Kellner, Dhital, Kiselev and Resiert, MRM 2016, 76:1574-1581 (if full k-space). Lee, Novikov and Fieremans, ISMRM 2021 (if partial k-space). Default:1 

     --gibbs_kspace_coverage 
          Gibbs_kspace_coverage (float). To overwrite what is read from the JSON file, in case something is wrong. Possible values: 0, 1 , 0.875 , 0.75. Default:0, which means do not use this tag 

     --gibbs_nsh 
          Parameter for gibbs correction (int). Default:25 

     --gibbs_minW 
          Parameter for gibbs correction (int). Default:1 

     --gibbs_maxW 
          Parameter for gibbs correction (int). Default:3 

## Inter-Volume Motion and Eddy-Currents Distortion Correction Settings
     --b0_id -1:  will automatically select the best b=0 image.
              0:  will use the first volume in the data.
              vol_id:  Will use the volume with vol_id. It is the user's responsability to make sure this volume is a b=0 image
          Among possibly many b=0 s/mm2 images, the index of the b=0 image (in terms of volume id starting from 0) to be used as template (int). Default:-1 

     --is_human_brain 
          Is it an in-vivo human brain? (boolean). Specialized processing is performed if human brain. Default:1 

     --rot_eddy_center isocenter: the (0,0,0) coordinate from the NIFTI header (ideally the magnet isocenter) will be used as the center
                       center_voxel:  the very center voxel of the image will be used as the isocenter.
                       center_slice:  0,0,center_slice . This is useful when for example an insert coil is used and the NIFTI header is not fully correct.
          Rotation and eddy-currents center (string). Default:isocenter 

     --center_of_mass 
          Place the center of mass of the image to the center pixel (boolean). Affects only processing not the final data. By default it is on. However, for non-centralized images, it should be turned off.

     -c, --correction_mode off: no motion or eddy currents distortion correction
                           motion: Corrects only motion with a rigid transformation
                           eddy_only: Corrects only eddy-currents distortions and no motion. Ideal for phantoms.
                           quadratic: Motion&eddy. Eddy currents are modeled with upto quadratic Laplace bases. Quadratic model is sufficient 99% of the time.
                           cubic: Motion&eddy.  Eddy currents are modeled with upto-including cubic Laplace bases
          Motion & eddy-currents distortions correction mode (string). Specifies which undesirable effects will be corrected. Predefined motion & eddy distortion correction optimization settings. Each setting points to a file in the software's settings/mecc_settings folder. Default:quadratic 

     --s2v 
          Slice to volume or in other-words slice-by-slice correction (boolean). Significantly increases processing time but no other penalties in data quality. Default:0 

     --repol 
          Outlier detection and replacement (boolean). Replaces the automatically detected outlier slices with estimates from the MAPMRI model.Significantly increases processing time but no other penalties in data quality. Default:0 

     --outlier_frac 
          Outlier fraction ratio. If percentace of outlier slices is larger than this threshold, the entire volume is considered troublesome and all the values replaced with the predicted ones. Default:0.5 

     --outlier_prob 
          Outlier probability threshold. If the probability of a slice RMS is lower than this value, that slice is labeled artifactual. Default:0.025 

     --outlier_EM_clusters 
          Number of RMS clusters for EM outlier detection. Ideally there should be 2 clusters: inliers, outliers. However, life is not perfect. Clusters will afterwards be combined till they reach 75% inclusion. Default:4 

     --outlier_replacement_mode conservative: less voxels labeled as outliers
                                middle: somewhere between conservative and aggressive
                                aggressive: more voxels labeled as outliers
          String to determine whether to label more voxels as outliers or less. Default:middle 

     --niter 
          Number of iterations for high_bvalue / s2v / repol correction. Has no effect for dti regime data with s2v and repol disabled. Default:3 . Zero disables all iterative correction such as high-b, s2v or repol. 

     --dti_bval 
          DTI bval (int). In case non in-vivo human brain data, or to overwrite the default value of 1000 s/mm2, what is the bval for DTI regime? Default:1000 

     --hardi_bval 
          HARDI bval (int). In case non in-vivo human brain data, or to overwrite the default value of 2000 s/mm2, what is the bval for higher order regime? Default:2000 

## Signal Drift Settings
     --drift off: No signal drift correction. Default.
             linear: Linear signal drift over time
             quadratic:  Quadratic signal drift over time.
          Signal drift correction method. Data will be checked to automatically determine if this correction can be applied.    

## Susceptibility Distortion Correction settings
     --DRBUDDI_output 
          DRBUDDI transformation output folder. 

     --DRBUDDI_step 
          DRBUDDI start step. 0: beginning. Creates b=0 and FA images. 1: Assumes b=0 and FA images are present with correct name 
          and starts with rigid registration. 2: Assumes all images and rigid transformations are present and starts with 
          diffeomorphic distortion correction. 

     --DRBUDDI_initial_fixed_transform 
          Initial transform field for the up data. 

     --DRBUDDI_initial_moving_transform 
          Initial transform field for the down data. 

     --DRBUDDI_disable_initial_rigid 
          DRBUDDI performs an initial registration between the up and down data. This registration starts with rigid, followed by 
          a quick diffeomorphic and finalized by another rigid. This parameter, when set to 1 disables all these registrations. 
          Default: 0 

     --DRBUDDI_start_with_diffeomorphic_for_rigid_reg 
          DRBUDDI performs an initial registration between the up and down data. This registration starts with rigid, followed by 
          a quick diffeomorphic and finalized by another rigid. This parameter, when set to 1 disables the very initial rigid 
          registration and starts with the quick diffemorphic. This is helpful with VERY DISTORTED data, for which the initial 
          rigid registration is problematic. Default: 0 

     --DRBUDDI_rigid_metric_type 
          Similarity Metric to be used in rigid registration. Options: MI or CC. Default: MI 

     --DRBUDDI_rigid_learning_rate 
          Rigid metric learning rate: Default:0.25 

     --DRBUDDI_DWI_bval_tensor_fitting 
          Up to which b-value should be used for DRBUDDI's tensor fitting. Default: 0 , meaning use all b-values 

     --DRBUDDI_estimate_LR_per_iteration 
          Flag to estimate learning rate at every iteration. Makes DRBUDDI slower but better results. Boolean. Default:0 

     --DRBUDDI_stage [learning_rate={learning_rate},cfs={Niterations:downsampling_factor:image_smoothing_stdev},field_smoothing={update_field_smoothing_stdev:total_field_smoothing_stdev},metrics={metric1:metric2:...metricn},restrict_constrain={restrict_to_phaseencoding:enforce_up_down_deformation_symmetry}]
                     [learning_rate={0.5},cfs={100:1:0},field_smoothing={3.:0.1},metrics={MSJac:CC:CCSK{str_id=0}:CCSK{str_id=1}},restrict_constrain={1:1}]
          DRBUDDI runs many registration stages during correction. This tag sets all the parameters for a given stage. Each stage 
          is executed in the order provided on the command line. Available metrics are: MSJac, CC, CCSK. MSJac uses the b=0 
          images. CC uses FA images. CCSK uses b=0 and the structural images. Which structural image to be used with CCSK is given 
          with an index as: CCSK{str_id=1}. 

     --enforce_full_symmetry 
          Flag to enforce DRBUDDI to enforce blip-up blip-down antisymmetry and phasen-encoding restriction when using the default 
          settings. Boolean. Default:0 

     --DRBUDDI_disable_last_stage 
          Flag to enforce DRBUDDI to enforce blip-up blip-down antisymmetry and phasen-encoding restriction when using the default 
          settings. Boolean. Default:0 

     --DRBUDDI_structural_weight 
          Multiplicative factor for metrics that use the structural image. Might want to reduce it if the structural's contrast is 
          significantly different than the b=0. Float. Default:1 

     --disable_itk_threads 
          The last DRBUDDI stage heavily favors the structural image. If this image is not ideal (not a good contrast), it might 
          be more robust to disable this stage. Boolean. Default:0 

     --transformation_type 
          Registration transformation type. Options: SyN or TVVF. Default: SyN. TVVF only works in CUDA version. 

     --ncores 
          Number of cores to use in the CPU version. The default is 50% of system cores. 

     --epi off: no EPI distortion correction performed.
           T2Wreg: Diffeomorphically register the b=0 image to the provided T2W structural image. 
           DRBUDDI:  Perform blip-up blip-down correction with DRBUDDI.
          EPI Distortion correction method. 
          
## Final Data Generation Settings
           -o, --output 
          Output name of the final NIFTI file 

     --output_orientation 
          Output orientation of the data. 3 characters, eg: LPS, RAI, ILA. First letter for the anatomical direction that is from 
          left of the image TOWARDS right (The letter if for the ending point of the direction not the beginning). Second letter 
          from the top of the image to bottom. Third letter from the first slice to the last slice. Default: LPS. 

     --output_res 
          Resolution of the final output: res_x res_y res_z separated by space. Default: the original resolution 

     --output_voxels 
          Number of voxels in the final image: Nx Ny Nz separated by space. Image might be padded/cropped to match this. Default: 
          computed from structural's FoV and the output resolution. 

     --interp_POW 
          For final interpolation, the power for inverse distance weighting in case s2v is on. Minimum:2. Smaller numbers will 
          yield smoother images. Very large numbers will be closer to nearest neighbor interpolation. Default:8 

     --output_data_combination Merge: If up and down data have the same Bmatrix, the corresponding DWIs are geometrically averaged. Default if this is the case.
                               JacConcat: Up and down DWIs' signals are manipulated by the Jacobian. The two data are concatenated into a single one. Default if upBmtxt != downBmtxt
                               JacSep: Up and down data are Jacobian manipulated and saved separately. The "output" tag has no effect. 
          Output data combination method: 

     --output_signal_redist_method Jac: Signal manipulation done by transformation Jacobian.
                                   LSR: signal manipulation done by estimating the mapping of the geometry-corrected/signal-uncorrected images to geometry-and-signal-corrected images.
          Signal Redistribution method for final output: Jac/LSR . Default: LSR 

     --output_gradnonlin_Bmtxt_type grad_dev: A single gradient deviation tensor image is written in HCP style to be applied to ALL volumes.
                                    vbmat: A voxelwise Bmatrix image is written that also includes the effect of motion. LARGE SIZE. Default.
          Format of the gradient nonlinearity output information. If vbmat is selected, s2v effects will also be considered even 
          if no gradient nonlinearity information is present. 


