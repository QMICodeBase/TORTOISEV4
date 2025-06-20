TORTOISEV4 has many tools for specific purposes.  Each module in the TORTOISEProcess pipeline has also a corresponding executable in case users prefer to use them independently. Here is the list and their descriptions.

# Input Related Tools



## ApplyTransformationToScalar2
Usage:   ApplyTransformationToScalar2   full_path_to_scalar_to_be_transformed  full_path_to_transformation  full_path_to_name_of_output full_path_to_image_with_desired_dimensions InterpolantType (NN, Lin, BSP)

This executable can transform a 3D image using an ITK format (compatible with ANTS) deformation field or an ITK affine transform (text file format. For binary ANTS transforms, convert them to text first with c3d_affine_tool).

## Combine3DImagesTo4D
Usage: Combine3DImagesTo4D output_nifti_filename nifti1_filename  nifti2_filename  .......... niftiN_filename 

Combines multiple 3D images into a single 4D image.  The order of volumes will be identical to the order provided in the command line.
