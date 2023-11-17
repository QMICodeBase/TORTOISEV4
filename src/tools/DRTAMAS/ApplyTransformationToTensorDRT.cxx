#include "DRTAMAS.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"



#include "itkResampleImageFilter.h"
#include "itkTransformFileReader.h"
#include "itkExtractImageFilter.h"


#include "DRTAMAS_utilities_cp.h"




int main( int argc , char * argv[] )
{
    if(argc<4)    
    {
        std::cout<<"Usage:   ApplyTransformationToTensor   full_path_to_tensor_to_be_transformed  full_path_to_transformation  full_path_to_name_of_output full_path_to_image_with_desired_dimensions "<<std::endl;
        return EXIT_FAILURE;
    }
    
    
    auto tensor = ReadAndOrientTensor(argv[1]);
    auto ref_tensor = ReadAndOrientTensor(argv[4]);


    std::string filename(argv[2]);
    std::string::size_type idx=filename.rfind('.');
    std::string extension = filename.substr(idx+1);
    std::string output_nii_name=argv[3];

    DisplacementFieldType::Pointer disp_field=nullptr;
    using AffineTransformType = DRTAMAS::AffineTransformType;
    AffineTransformType::Pointer affine_trans=nullptr;

    if(idx != std::string::npos)
    {
        if(filename.find(".nii")!=std::string::npos  )
        {
            typedef itk::ImageFileReader<DisplacementFieldType> FieldReaderType;
            typename FieldReaderType::Pointer mreader=FieldReaderType::New();
            mreader->SetFileName(filename);
            mreader->Update();
            disp_field =mreader->GetOutput();

            TransformAndWriteDiffeoImage(tensor,disp_field,ref_tensor, output_nii_name);
        }
        else
        {
            using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
            TransformReaderType::Pointer reader = TransformReaderType::New();
            reader->SetFileName(filename );
            reader->Update();
            const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
            itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
            affine_trans = static_cast<AffineTransformType*>((*it).GetPointer());

            TransformAndWriteAffineImage(tensor,affine_trans,ref_tensor, output_nii_name);

        }
    }

}
