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
        std::cout<<"ZeroFieldDims field_img indices_to_zero(comma_separated) output_name"<<std::endl;
        exit(EXIT_FAILURE);
    }


    typedef itk::ImageFileReader<DisplacementFieldType> FieldReaderType;
    FieldReaderType::Pointer rd=FieldReaderType::New();
    rd->SetFileName(argv[1]);
    rd->Update();
    DisplacementFieldType::Pointer field= rd->GetOutput();

    std::string ind_string(argv[2]);

    std::vector<int> inds;

    while(ind_string.find(",")!=std::string::npos)
    {
        int pos=ind_string.find(",");
        std::string sub= ind_string.substr(0,pos);
        inds.push_back(atoi(sub.c_str()));
        ind_string= ind_string.substr(pos+1);
    }
    inds.push_back(atoi(ind_string.c_str()));

    itk::ImageRegionIterator<DisplacementFieldType> it(field,field->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        DisplacementFieldType::PixelType vec= it.Get();
        for(int d=0;d<inds.size();d++)
        {
            vec[inds[d]]=0;
        }
        it.Set(vec);
        ++it;
    }


    typedef itk::ImageFileWriter<DisplacementFieldType> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetInput(field);
    wr->SetFileName(argv[3]);
    wr->Update();


}
