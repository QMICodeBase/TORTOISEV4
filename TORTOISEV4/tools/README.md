TORTOISEV4 has many tools for specific purposes.  Each module in the TORTOISEProcess pipeline has also a corresponding executable in case users prefer to use them independently. Here is the list and their descriptions.

# Input Related Tools

## BrukerBmatToTORTOISEBMTXT
Usage:  BrukerBmatToTORTOISEBMTXT input_bmat_file

Converts a bmat file (either output by the scanner or copied from the method file) to a TORTOSIE compatible Bmatrix file.

## Combine3DImagesTo4D
Usage: Combine3DImagesTo4D output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename

Concatenates several 3D images on to a single 4D NIFTI image.

## Combine4DImages
Usage: Combine4DImages output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename

Concatenates multiple 4D images on to a single 4D one.

## CombineDWIs
Usage: CombineDWIs output_nifti_filename nifti1_filename bvals1_filename bvecs1_filename  .......... niftiN_filename bvalsN_filename bvecsN_filename

Concatenates multiple datasets with corresponding bvecs/bvals into a single dataset with the corresponding Bmatrix. Every single input nifti image should have a corresponding bvecs and a bvals file, which should be provided in the command line.

## CombineDWIsWithBmatrix
Usage: CombineDWIsWithBMatrix output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename

Concatenates multiple datasets with corresponding Bmatrices into a single dataset with the total Bmatrix. All Bmatrices should exist with the same basename as the NIFTI images but they  do not need to be provided in the command line.


## ConvertGEGradientsToBMatrix
Usage: ConvertGEGradientsToBMatrix dwi_nifti_file grad_file max_bval

Gradients from a GE scanner used to be hardcoded in the DICOM headers and with custom gradients input to the system, they would be incorrect. If that is the case, this executable takes in a textfile in the format of the GE tensor.dat file, only containing the corresponding number of volumes. The corresponding NIFTI file and the maximum bvalue should also be provided in the command line. Intermediate bvalues will be inferred from gradient scaling.



## CreateDummyJson
Usage: CreateDummyJson

     -i, --input 
          Full path to the input NIFTI DWIs 

     -p, --phase 
          Phase encoding direction. Options: i+ (for RL), i- (for LR), j+ (for AP), j- (for PA) 

     -m, --MBf 
          Multi band factor 

     -l, --interleave 
          Just interleaved 

     -f, --partial_fourier 
          Partial Fourier. Options: 1, 0.875 , 0.75 

     --big_delta 
          Big delta. Diffusion separation time. 

     --small_delta 
          Small delta. Diffusion time. 
          

TORTOISE requires a json file for each NIFTI.  For legacy data without any json files, this executable can be used generate one. Input, phase. multi band factor and partial_fourier coverage are mandatory.


##  DIFFPREPAnonymizer
Usage: DIFFPREPAnonymizer path_to_parent_DICOM_folder path_to_output_parent_DICOM_folder path_to_textfile_containing_tags_to_be_anonymized (optional).

Custom DICOM Anonymizer.

##  ExtractDWISubset
Usage: ExtractDWISubset
     -i, --input_image 
          Full path to the input nifti 

     -o, --output_image 
          Full path to the output nifti 

     -b, --bvals 
          b-values of the volume to be extracted. Comma separated. A range can be given with - . For example : 0-500,750,3000-5000 
          will include all the volumes with b-values in between (including) 0 and 500 and bvalue 750 and all volumes in between #K 
          and 5K (included) 

     -v, --vols 
          Volume numbers to be extracted. Comma separated. A range can be given with - . For example : 0,3-5,7 will include 
          volumes 0,3,4,5,7. 

Extract a subset of a DWI dataset including either using volume indices or the desired b-values.  The corresponding Bmatrix will also be created.

##  FlipSwapBMatrix
Usage: FlipSwapBMatrix
    --input_bmtxt 
          Full path to Input Bmatrix. 

    --output_bmtxt 
          Full path to Output Bmatrix. 

    --X 
          New X orientation. Examples: (--X x), (--X -z) (--X y). Default: x 

    --Y 
          New Y orientation. Examples: (--Y x), (--Y -z) (--Y y). Default: y 

    --Z 
          New Z orientation. Examples: (--Z x), (--Z -z) (--Z y). Default: z 

Flip and swap entries of an existing Bmatrix to create a new one. To be used when the input gradients do not match the NIFTI image.

##  FSLBVecsToTORTOISEBmatrix
Usage:FSLBVecsToTORTOISEBmatrix bvals_file bvecs_file

Converts a bvecs and bvals to a TORTOISE format Bmatrix. Normally not necessary for data processing.

##  TORTOISEBMatrixToBT
Usage: TORTOISEBMatrixToBT bmatrix_file

Converts TORTOISE Bmatrix to a BTensor for multi-dimensional analysis.


##  TORTOISEBmatrixToFSLBVecs
Usage: TORTOISEBmatrixToFSLBVecs bmatrix_file

Converts TORTOISE Bmatrix to bvecs and bvals files by doing an eigendecomposition on the Bmatrix.


# DRTAMAS Tools


## ApplyTransformationToScalar2
Usage:   ApplyTransformationToScalar2   full_path_to_scalar_to_be_transformed  full_path_to_transformation  full_path_to_name_of_output full_path_to_image_with_desired_dimensions InterpolantType (NN, Lin, BSP)

This executable can transform a 3D image using an ITK format (compatible with ANTS) deformation field or an ITK affine transform (text file format. For binary ANTS transforms, convert them to text first with c3d_affine_tool).

## Combine3DImagesTo4D
Usage: Combine3DImagesTo4D output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename 

Combines multiple 3D images into a single 4D image.  The order of volumes will be identical to the order provided in the command line.
