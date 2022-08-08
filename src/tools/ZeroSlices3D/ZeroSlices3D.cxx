

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageDuplicator.h"

#include "itkImageRegionIteratorWithIndex.h"


    typedef float RealType;
    typedef itk::Image<RealType,3> ImageType3D;


int main( int argc , char * argv[] )
{
    if(argc<5)
    {
        std::cout<<"Usage:  ZeroSlices3D input_image output_image  start_slice_index end_index(inclusive)"<<std::endl;
        return 0;
    }
    
    
    

    typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
    ImageType3DReaderType::Pointer imager= ImageType3DReaderType::New();
    imager->SetFileName(argv[1]);
    imager->Update();
    ImageType3D::Pointer image3D= imager->GetOutput();

    ImageType3D::SizeType sz= image3D->GetLargestPossibleRegion().GetSize();

    int start_ind=atoi(argv[3]);
    int end_ind=atoi(argv[4]);

    for(int k=start_ind;k<=end_ind;k++)
    {
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                image3D->SetPixel(ind3,0);
            }
        }
    }


       

    std::string filename(argv[2]);

    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image3D);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
