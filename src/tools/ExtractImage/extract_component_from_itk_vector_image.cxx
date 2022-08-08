
#include "extract_image_parser.h"


#include "itkDisplacementFieldTransform.h"

#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

int main(int argc, char * argv[])
{
    if(argc==1)
    {
        std::cout<<"Usage:  ExtractComponentFromDisplacementField  -i input_vector_image -v component_id "<<std::endl<<std::endl;
        //return EXIT_FAILURE;
    }


    Extract_Image_PARSER *parser = new Extract_Image_PARSER(argc,argv);

    std::string oname;
    if(parser->getOutputImageName()=="")
    {
        std::string nm=parser->getInputImageName();
        char buf[2000];
        sprintf(buf,"_V%.3d.nii",parser->getVolId());
        if(nm.find(".nii.gz")!=std::string::npos)
            oname=nm.substr(0,nm.find(".nii.gz")) + std::string(buf);
        else
            oname=nm.substr(0,nm.find(".nii")) + std::string(buf);
    }
    else
    {
        oname=parser->getOutputImageName();
    }


    typedef itk::DisplacementFieldTransform<double,3> DisplacementFieldTransformType;
    typedef DisplacementFieldTransformType::DisplacementFieldType DisplacementFieldType;


    typedef itk::ImageFileReader<DisplacementFieldType> RdType;
    RdType::Pointer rd= RdType::New();
    rd->SetFileName(parser->getInputImageName());
    rd->Update();
    DisplacementFieldType::Pointer field= rd->GetOutput();


    typedef itk::Image<float,3> ImageType3D;
    ImageType3D::Pointer output_img = ImageType3D::New();
    output_img->SetRegions(field->GetLargestPossibleRegion());
    output_img->Allocate();
    output_img->SetDirection(field->GetDirection());
    output_img->SetOrigin(field->GetOrigin());
    output_img->SetSpacing(field->GetSpacing());

    int vol_id = parser->getVolId();

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(output_img,output_img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3 = it.GetIndex();
        DisplacementFieldType::PixelType vec= field->GetPixel(ind3);
        float val = vec[vol_id];

        it.Set(val);
        ++it;
    }

    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr=WrType::New();
    wr->SetInput(output_img);
    wr->SetFileName(oname);
    wr->Update();



    return EXIT_SUCCESS;
}
