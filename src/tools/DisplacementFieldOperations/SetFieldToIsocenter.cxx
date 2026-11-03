#include <algorithm>
#include <ctype.h>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDisplacementFieldTransform.h"

typedef itk::DisplacementFieldTransform<double,3>::DisplacementFieldType DisplacementFieldType;
typedef itk::Image<float,3> ImageType3D;


int main(int argc,char *argv[])
{
    if(argc<4)
    {
        std::cout<<"SetFieldToIsocenter field_img ref_img output_name"<<std::endl;
        exit(EXIT_FAILURE);
    }


    typedef itk::ImageFileReader<DisplacementFieldType> FieldReaderType;
    FieldReaderType::Pointer rd=FieldReaderType::New();
    rd->SetFileName(argv[1]);
    rd->Update();
    DisplacementFieldType::Pointer field= rd->GetOutput();

    typedef itk::ImageFileReader<ImageType3D> ReaderType;
    ReaderType::Pointer rd2=ReaderType::New();
    rd2->SetFileName(argv[2]);
    rd2->Update();
    ImageType3D::Pointer fvol= rd2->GetOutput();

    itk::ContinuousIndex<double,3> center_ind;
    center_ind[0]= ((int)(fvol->GetLargestPossibleRegion().GetSize()[0])-1)/2.;
    center_ind[1]= ((int)(fvol->GetLargestPossibleRegion().GetSize()[1])-1)/2.;
    center_ind[2]= ((int)(fvol->GetLargestPossibleRegion().GetSize()[2])-1)/2.;

    ImageType3D::PointType center_pt;
    fvol->TransformContinuousIndexToPhysicalPoint(center_ind,center_pt);

    ImageType3D::PointType new_orig=field->GetOrigin();
    new_orig[2]+= center_pt[2];
    field->SetOrigin(new_orig);


    typedef itk::ImageFileWriter<DisplacementFieldType> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetInput(field);
    wr->SetFileName(argv[3]);
    wr->Update();


}
