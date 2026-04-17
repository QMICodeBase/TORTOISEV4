

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDisplacementFieldTransform.h"
#include "itkImageRegionIteratorWithIndex.h"




    typedef float RealType;        


    typedef itk::Image<itk::Vector<RealType,3>,  3>          DisplacementFieldType;

    
           

    

int main( int argc , char * argv[] )
{
    if(argc<3)
    {
        std::cout<<"Usage:   MultiplyFieldwithScalar  field.nii scalar"<<std::endl;
        return EXIT_FAILURE;
    }
    
    
    typedef itk::ImageFileReader<DisplacementFieldType> ImageType3DReaderType;
    ImageType3DReaderType::Pointer fixedr= ImageType3DReaderType::New();
    fixedr->SetFileName(argv[1] );
    fixedr->Update();
    DisplacementFieldType::Pointer field= fixedr->GetOutput();
    
    float scalar = atof(argv[2]);

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(field,field->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        DisplacementFieldType::IndexType ind3= it.GetIndex();
        DisplacementFieldType::PixelType vec= it.Get();
        vec= scalar *vec;
        it.Set(vec);
    }
    
    

    
    
    std::string nm= argv[1];
    
    std::string oname;
    oname = nm.substr(0,nm.rfind(".nii"));
    char onm2[1000]={0};

    sprintf(onm2,"%s_f%d.nii",oname.c_str(),(int)(scalar*100));
    
        
    typedef itk::ImageFileWriter<DisplacementFieldType> WriterType;
    WriterType::Pointer writer= WriterType::New();    
    writer->SetFileName(onm2);
    writer->SetInput(field);
    writer->Update();
    
    return EXIT_SUCCESS;
}

