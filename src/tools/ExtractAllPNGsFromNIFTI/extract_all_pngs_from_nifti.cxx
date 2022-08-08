

#include <iostream>
#include <fstream>
#include <sstream>
#include <strstream>
using namespace std;

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "defines.h"

#include "itkRescaleIntensityImageFilter.h"
#include "itkNiftiImageIO.h"
#include "itkRGBPixel.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
namespace fs = boost::filesystem;


#include "itkCastImageFilter.h"


int main( int argc , char * argv[] )
{
    if(argc<3)
    {
        std::cout<<"Usage: ExtractAllPNGsFromNIFTI  full_path_to_NIFTI_file  full_path_to_output_folder axis (optional)"<<std::endl;
        std::cout<<"Axis:  axial, sagittal or coronal.  Default axial."<<std::endl;        
        return 0;
    }
        

    int axis=2;
    if(argc>3)
    {
        if(std::string(argv[3])=="sagittal")
            axis=0;

        if(std::string(argv[3])=="coronal")
            axis=1;
    }



    itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
    myio->SetFileName(argv[1]);
    myio->ReadImageInformation();
    int Ncomps= myio->GetNumberOfComponents();

    ImageType3D::SizeType sz;
    sz[0]= myio->GetDimensions(0);
    sz[1]= myio->GetDimensions(1);
    sz[2]= myio->GetDimensions(2);

    std::string output_folder(argv[2]);



    if(Ncomps==1)
    {
        typedef itk::ImageFileReader<ImageType3D> ReaderType3;
        ReaderType3::Pointer reader= ReaderType3::New();
        reader->SetFileName(argv[1]);
        reader->Update();
        ImageType3D::Pointer img= reader->GetOutput();


        typedef itk::RescaleIntensityImageFilter< ImageType3D, ImageType3D >        RescaleFilterType;
        RescaleFilterType::Pointer rescale = RescaleFilterType::New();
        rescale->SetInput( img);
        rescale->SetOutputMinimum( 0 );
        rescale->SetOutputMaximum( 255 );
        rescale->UpdateLargestPossibleRegion();
        rescale->Update();
        ImageType3D::Pointer img_scl= rescale->GetOutput();


        typedef itk::Image<float,2> SliceTypeFloat;
        typedef itk::Image<unsigned char,2> SliceTypeChar;


        for(int k=0;k<sz[axis];k++)
        {
            ImageType3D::IndexType index;
            index.Fill(0);
            index[axis]=k;
            ImageType3D::SizeType imsz =sz;
            imsz[axis]=0;
            ImageType3D::RegionType reg(index,imsz);

            typedef itk::ExtractImageFilter<ImageType3D,SliceTypeFloat> ExtracterType;
            ExtracterType::Pointer extractor= ExtracterType::New();
            extractor->SetInput(img_scl);
            extractor->SetExtractionRegion(reg);
            extractor->SetDirectionCollapseToSubmatrix();
            extractor->Update();
            SliceTypeFloat::Pointer  slice_float = extractor->GetOutput();

            typedef itk::CastImageFilter< SliceTypeFloat, SliceTypeChar > CasterType;
            CasterType::Pointer caster= CasterType::New();
            caster->SetInput(slice_float);
            caster->Update();
            SliceTypeChar::Pointer slice= caster->GetOutput();


            fs::path inp_path(argv[1]);
            std::string oname= output_folder + std::string("/") + inp_path.stem().string() + std::string("_");
            char oname2[2000]={0};
            sprintf(oname2,"%s_%.3d.png",oname.c_str(),k);

            typedef itk::ImageFileWriter<SliceTypeChar> WrType;
            WrType::Pointer wr= WrType::New();
            wr->SetFileName(oname2);
            wr->SetInput(slice);
            wr->Update();
        }


    }
    else
    {
        typedef itk::RGBPixel< unsigned char >   PixelType;
        typedef itk::Image< PixelType, 3 >       RGBImageType;
        typedef itk::Image< PixelType, 2 >       RGBSliceType;

        typedef itk::ImageFileReader<RGBImageType> ReaderType3;
        ReaderType3::Pointer reader= ReaderType3::New();
        reader->SetFileName(argv[1]);
        reader->Update();
        RGBImageType::Pointer img= reader->GetOutput();



        for(int k=0;k<sz[axis];k++)
        {
            RGBImageType::IndexType index;
            index.Fill(0);
            index[axis]=k;
            RGBImageType::SizeType imsz =sz;
            imsz[axis]=0;
            RGBImageType::RegionType reg(index,imsz);


            typedef itk::ExtractImageFilter<RGBImageType,RGBSliceType> ExtracterType;
            ExtracterType::Pointer extractor= ExtracterType::New();
            extractor->SetInput(img);
            extractor->SetExtractionRegion(reg);
            extractor->SetDirectionCollapseToSubmatrix();
            extractor->Update();
            RGBSliceType::Pointer slice= extractor->GetOutput();

            fs::path inp_path(argv[1]);
            std::string oname= output_folder + std::string("/") + inp_path.stem().string() + std::string("_");
            char oname2[2000]={0};
            sprintf(oname2,"%s_%.3d.png",oname.c_str(),k);

            typedef itk::ImageFileWriter<RGBSliceType> WrType;
            WrType::Pointer wr= WrType::New();
            wr->SetFileName(oname);
            wr->SetInput(slice);
            wr->Update();
        }
    }


    
    return EXIT_SUCCESS;
}
