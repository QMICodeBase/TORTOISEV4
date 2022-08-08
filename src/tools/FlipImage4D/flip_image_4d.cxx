

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
    typedef itk::Image<RealType,4> ImageType4D;


     typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;
    
           

int main( int argc , char * argv[] )
{
    if(argc<5)
    {
        std::cout<<"Usage:   FlipImage4D input_image.nii output_image.nii  flip_axis (x/y/z)   change_direction? (0/1)"<<std::endl;
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
    

    typedef itk::ImageDuplicator<ImageType4D> DupType;
    DupType::Pointer dup = DupType::New();
    dup->SetInputImage(image4D);
    dup->Update();
    ImageType4D::Pointer output_image= dup->GetOutput();



    char flip_axis_char = argv[3][0];
    int flip_axis;
    if(flip_axis_char=='x' || flip_axis_char=='X')
        flip_axis=0;
    if(flip_axis_char=='y' || flip_axis_char=='Y')
        flip_axis=1;
    if(flip_axis_char=='z' || flip_axis_char=='Z')
        flip_axis=2;


    ImageType4D::SizeType sz= output_image->GetLargestPossibleRegion().GetSize();




    itk::ImageRegionIteratorWithIndex<ImageType4D> it(output_image, output_image->GetLargestPossibleRegion());

    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType4D::IndexType oindex= it.GetIndex();
        ImageType4D::IndexType iindex= it.GetIndex();
        iindex[flip_axis] = sz[flip_axis]-1-iindex[flip_axis];

        float val= image4D->GetPixel(iindex);
        it.Set(val);
        ++it;
    }


    if(atoi(argv[4])!=0)
    {
        ImageType4D::DirectionType orig_dir = output_image->GetDirection();

        ImageType4D::DirectionType mod;
        mod.SetIdentity();

        if(flip_axis==0)
            mod(0,0)=-1;
        if(flip_axis==1)
            mod(1,1)=-1;
        if(flip_axis==2)
            mod(2,2)=-1;

        ImageType4D::DirectionType ndir = mod*orig_dir;
        output_image->SetDirection(ndir);

        ImageType4D::PointType new_orig;
        ImageType4D::IndexType new_orig_index;
        new_orig_index.Fill(0);

        new_orig_index[flip_axis]= sz[flip_axis]-1;
        image4D->TransformIndexToPhysicalPoint(new_orig_index,new_orig);
        output_image->SetOrigin(new_orig);
    }



    
       
    
       
        
         std::string filename(argv[2]);

    
    
    typedef itk::ImageFileWriter<ImageType4D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(output_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
