#ifndef _DEFINES_HXX
#define _DEFINES_HXX

#include "itkImage.h"
#include <ostream>

#include "defines.h"



template<typename ImageType>
typename ImageType::Pointer readImageD(std::string filename)
{
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typename ReaderType::Pointer reader= ReaderType::New();
    reader->SetFileName(filename);
    reader->Update();
    typename ImageType::Pointer img= reader->GetOutput();
    return img;
}



template<typename ImageType>
void writeImageD(typename ImageType::Pointer img, std::string filename)
{
    typedef itk::ImageFileWriter<ImageType> WriterType;
    typename WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(img);
    writer->Update();


}


template ImageType3D::Pointer readImageD<ImageType3D>(std::string) ;
template void writeImageD<ImageType3D>(ImageType3D::Pointer , std::string);

template ImageType3DBool::Pointer readImageD<ImageType3DBool>(std::string) ;
template void writeImageD<ImageType3DBool>(ImageType3DBool::Pointer , std::string);


template ImageType4D::Pointer readImageD<ImageType4D>(std::string) ;
template void writeImageD<ImageType4D>(ImageType4D::Pointer , std::string);

template ImageType4DBool::Pointer readImageD<ImageType4DBool>(std::string) ;
template void writeImageD<ImageType4DBool>(ImageType4DBool::Pointer , std::string);


template DisplacementFieldType::Pointer readImageD<DisplacementFieldType>(std::string) ;
template void writeImageD<DisplacementFieldType>(DisplacementFieldType::Pointer , std::string);


template void writeImageD< itk::Image<itk::Vector<float,3>,3> >(itk::Image<itk::Vector<float,3>,3>::Pointer , std::string);

#endif
