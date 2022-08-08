

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


     typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;
    
           

int main( int argc , char * argv[] )
{
    if(argc<5)
    {
        std::cout<<"Usage:   FlipImage3D input_image.nii output_image.nii  flip_axis (x/y/z)   change_direction? (0/1)"<<std::endl;
        return 0;
    }
    
    
    std::string currdir;
    std::string nm(argv[1]);
    
         

            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    

    typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
    ImageType3DReaderType::Pointer imager= ImageType3DReaderType::New();
    imager->SetFileName(nm);
    imager->Update();
    ImageType3D::Pointer image3D= imager->GetOutput();
    

    typedef itk::ImageDuplicator<ImageType3D> DupType;
    DupType::Pointer dup = DupType::New();
    dup->SetInputImage(image3D);
    dup->Update();
    ImageType3D::Pointer output_image= dup->GetOutput();



    char flip_axis_char = argv[3][0];
    int flip_axis;
    if(flip_axis_char=='x' || flip_axis_char=='X')
        flip_axis=0;
    if(flip_axis_char=='y' || flip_axis_char=='Y')
        flip_axis=1;
    if(flip_axis_char=='z' || flip_axis_char=='Z')
        flip_axis=2;


    ImageType3D::SizeType sz= output_image->GetLargestPossibleRegion().GetSize();

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(output_image, output_image->GetLargestPossibleRegion());

    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType oindex= it.GetIndex();
        ImageType3D::IndexType iindex= it.GetIndex();
        iindex[flip_axis] = sz[flip_axis]-1-iindex[flip_axis];

        float val= image3D->GetPixel(iindex);
        it.Set(val);
        ++it;
    }


    if(atoi(argv[4])!=0)
    {
        ImageType3D::DirectionType orig_dir = output_image->GetDirection();

        ImageType3D::DirectionType mod;
        mod.SetIdentity();

        if(flip_axis==0)
            mod(0,0)=-1;
        if(flip_axis==1)
            mod(1,1)=-1;
        if(flip_axis==2)
            mod(2,2)=-1;

        ImageType3D::DirectionType ndir = mod*orig_dir;
        output_image->SetDirection(ndir);


        ImageType3D::PointType new_orig;
        ImageType3D::IndexType new_orig_index;
        new_orig_index.Fill(0);

        new_orig_index[flip_axis]= sz[flip_axis]-1;
        image3D->TransformIndexToPhysicalPoint(new_orig_index,new_orig);
        output_image->SetOrigin(new_orig);
    }



    
       
    
       
        
    std::string filename(argv[2]);

    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(output_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
