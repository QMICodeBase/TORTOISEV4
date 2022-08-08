

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

#include "itkFlipImageFilter.h"

int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<<"Usage: ExtractPNGFromNIFTI  full_path_to_NIFTI_file  axis (optional) slice_number (optional) full_path_to_output (optional)"<<std::endl;
        std::cout<<"Axis:  axial, sagittal or coronal.  Default axial."<<std::endl;
        std::cout<<"slice_number: default center slice"<<std::endl;
        return 0;
    }
    
    

    int axis=2;
    if(argc>2)
    {
        if(std::string(argv[2])=="sagittal")
            axis=0;

        if(std::string(argv[2])=="coronal")
            axis=1;
    }

    int sl_number=-1;




    itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
    myio->SetFileName(argv[1]);
    myio->ReadImageInformation();
    int Ncomps= myio->GetNumberOfComponents();

    ImageType3D::SizeType sz;
    sz[0]= myio->GetDimensions(0);
    sz[1]= myio->GetDimensions(1);
    sz[2]= myio->GetDimensions(2);


    if(argc>3)
    {
        sl_number = atoi(argv[3]);
    }
    else
    {
        sl_number= sz[axis]/2;
    }

    std::string oname;
    if(argc>4)
    {
        oname= std::string(argv[4]);
    }
    else
    {
        std::ostringstream convert;
        convert << sl_number;

        std::string Result = convert.str();

        oname= std::string(argv[1]);
        oname = oname.substr(0,oname.find(".nii")) + std::string("_") + Result + std::string(".png");
    }

    std::cout<<oname<<std::endl;


    if(Ncomps==1)
    {
        typedef itk::ImageFileReader<ImageType3D> ReaderType3;
        ReaderType3::Pointer reader= ReaderType3::New();
        reader->SetFileName(argv[1]);
        reader->Update();
        ImageType3D::Pointer img= reader->GetOutput();

        typedef itk::Image<float,2> SliceTypeFloat;
        typedef itk::Image<unsigned char,2> SliceTypeChar;

        ImageType3D::IndexType index;
        index.Fill(0);
        index[axis]=sl_number;
        ImageType3D::SizeType imsz =sz;
        imsz[axis]=0;
        ImageType3D::RegionType reg(index,imsz);


        typedef itk::ExtractImageFilter<ImageType3D,SliceTypeFloat> ExtracterType;
        ExtracterType::Pointer extractor= ExtracterType::New();
        extractor->SetInput(img);
        extractor->SetExtractionRegion(reg);
        extractor->SetDirectionCollapseToSubmatrix();
        extractor->Update();

        SliceTypeFloat::Pointer slice_float=extractor->GetOutput();
        if(axis==0 || axis==1)
        {
            itk::FixedArray<bool, 2> flipAxes;
            flipAxes[0] = false;
            flipAxes[1] = true;
            typedef itk::FlipImageFilter <SliceTypeFloat>                      FlipImageFilterType;

              FlipImageFilterType::Pointer flipFilter                       = FlipImageFilterType::New ();
              flipFilter->SetInput(slice_float);
              flipFilter->SetFlipAxes(flipAxes);
              flipFilter->Update();
              slice_float=flipFilter->GetOutput();
        }

        typedef itk::RescaleIntensityImageFilter< SliceTypeFloat, SliceTypeChar >        RescaleFilterType;
        RescaleFilterType::Pointer rescale = RescaleFilterType::New();
        rescale->SetInput( slice_float );
        rescale->SetOutputMinimum( 0 );
        rescale->SetOutputMaximum( 255 );
        rescale->UpdateLargestPossibleRegion();
        rescale->Update();
        SliceTypeChar::Pointer slice = rescale->GetOutput();

        typedef itk::ImageFileWriter<SliceTypeChar> WrType;
        WrType::Pointer wr= WrType::New();
        wr->SetFileName(oname);
        wr->SetInput(slice);
        wr->Update();
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


        RGBImageType::IndexType index;
        index.Fill(0);
        index[axis]=sl_number;
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

        typedef itk::ImageFileWriter<RGBSliceType> WrType;
        WrType::Pointer wr= WrType::New();
        wr->SetFileName(oname);
        wr->SetInput(slice);
        wr->Update();
    }



    return EXIT_SUCCESS;
}
