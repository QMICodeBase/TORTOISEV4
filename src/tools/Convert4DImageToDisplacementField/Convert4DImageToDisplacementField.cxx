

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDisplacementFieldTransform.h"



    typedef float RealType;
    typedef itk::Image<RealType,4> ImageType4D;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::DisplacementFieldTransform<double, 3> DisplacementFieldTransformType;
    typedef DisplacementFieldTransformType::DisplacementFieldType DisplacementFieldType;


    
           

int main( int argc , char * argv[] )
{
    if(argc<2)
    {
        std::cout<<"Usage:  Convert4DImageToDisplacementField  path_to_4d_image "<<std::endl;
        return 0;
    }
    
    
    std::string currdir;
    std::string nm(argv[1]);
    
               
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    

    typedef itk::ImageFileReader<ImageType4D> ImageType4DReaderType;
    ImageType4DReaderType::Pointer imager= ImageType4DReaderType::New();
    imager->SetFileName(nm);
    imager->Update();
    ImageType4D::Pointer image4D= imager->GetOutput();


    DisplacementFieldType::SizeType sz;
    sz[0]=image4D->GetLargestPossibleRegion().GetSize()[0];
    sz[1]=image4D->GetLargestPossibleRegion().GetSize()[1];
    sz[2]=image4D->GetLargestPossibleRegion().GetSize()[2];
    DisplacementFieldType::IndexType start;start.Fill(0);
    DisplacementFieldType::RegionType reg(start,sz);

    DisplacementFieldType::PointType orig;
    orig[0]=image4D->GetOrigin()[0];
    orig[1]=image4D->GetOrigin()[1];
    orig[2]=image4D->GetOrigin()[2];

    DisplacementFieldType::SpacingType spc;
    spc[0]=image4D->GetSpacing()[0];
    spc[1]=image4D->GetSpacing()[1];
    spc[2]=image4D->GetSpacing()[2];

    DisplacementFieldType::DirectionType dir;
    dir(0,0)=image4D->GetDirection()(0,0);dir(0,1)=image4D->GetDirection()(0,1);dir(0,2)=image4D->GetDirection()(0,2);
    dir(1,0)=image4D->GetDirection()(1,0);dir(1,1)=image4D->GetDirection()(1,1);dir(1,2)=image4D->GetDirection()(1,2);
    dir(2,0)=image4D->GetDirection()(2,0);dir(2,1)=image4D->GetDirection()(2,1);dir(2,2)=image4D->GetDirection()(2,2);



    DisplacementFieldType::Pointer output_image = DisplacementFieldType::New();
    output_image->SetRegions(reg);
    output_image->Allocate();
    output_image->SetSpacing(spc);
    output_image->SetOrigin(orig);
    output_image->SetDirection(dir);

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(output_image,output_image->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        DisplacementFieldType::IndexType ind3 = it.GetIndex();
        ImageType4D::IndexType ind4;
        ind4[0]=ind3[0];
        ind4[1]=ind3[1];
        ind4[2]=ind3[2];

        DisplacementFieldType::PixelType vec;

        ind4[3]=0;
        float val = image4D->GetPixel(ind4);
        vec[0]=val;

        ind4[3]=1;
        val = image4D->GetPixel(ind4);
        vec[1]=val;

        ind4[3]=2;
        val = image4D->GetPixel(ind4);
        vec[2]=val;

        it.Set(vec);

        ++it;
    }


    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_dispfield.nii");
       
    
    
    typedef itk::ImageFileWriter<DisplacementFieldType> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(output_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
