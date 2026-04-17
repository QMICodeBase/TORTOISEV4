

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sstream>
#include <iomanip> // for std::ws
using namespace std;


#include "itkImage.h"
#include "itkDisplacementFieldTransform.h"
#include "defines.h"


bool isFloat(const std::string& s) {
    try {
        std::size_t pos;
        // Use stod to convert to double (generally safer/more precise than float)
        std::stod(s, &pos);

        // Skip potential trailing whitespace
        std::size_t first_non_whitespace = s.find_first_not_of(" \t\n\r\f\v", pos);

        // Check if the entire string was consumed (or only whitespace remains)
        return first_non_whitespace == std::string::npos;
    } catch (const std::out_of_range& oor) {
        // The number is a float, but too large/small for the type
        return true;
    } catch (const std::invalid_argument& ia) {
        // The string does not contain a number
        return false;
    }
}

    typedef float RealType;
    typedef itk::Image<RealType,4> ImageType4D;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::DisplacementFieldTransform<double, 3> DisplacementFieldTransformType;
    typedef DisplacementFieldTransformType::DisplacementFieldType DisplacementFieldType;


    
           

int main( int argc , char * argv[] )
{
    if(argc<5)
    {
        std::cout<<"Usage:  DisplacementFieldOperations output_name operation (m,-,+,/) field1 float/field2 "<<std::endl;
        return EXIT_FAILURE;
    }
    

    DisplacementFieldType::Pointer field1= readImageD<DisplacementFieldType>(argv[3]);
    DisplacementFieldType::Pointer field2=nullptr;
    float val2=1;


    if(isFloat(argv[4]))
        val2= std::stof(argv[4]);
    else
        field2= readImageD<DisplacementFieldType>(argv[4]);


    std::string op = argv[2];

    DisplacementFieldType::Pointer output_field = DisplacementFieldType::New();
    output_field->SetRegions(field1->GetLargestPossibleRegion());
    output_field->Allocate();
    output_field->SetDirection(field1->GetDirection());
    output_field->SetOrigin(field1->GetOrigin());
    output_field->SetSpacing(field1->GetSpacing());

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(output_field,output_field->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        DisplacementFieldType::IndexType ind3= it.GetIndex();
        DisplacementFieldType::PixelType vec1= field1->GetPixel(ind3);
        DisplacementFieldType::PixelType vec2; vec2.Fill(0);
        if(field2)
        {
            vec2=field2->GetPixel(ind3);
        }

        if( strcmp(op.c_str(), "+") == 0 )
        {
            if(field2)
            {
                vec1= vec1+vec2;
            }
            else
            {
                vec1=vec1+val2;
            }
        }
        if( strcmp(op.c_str(), "-") == 0 )
        {
            if(field2)
            {
                vec1= vec1-vec2;
            }
            else
            {
                vec1=vec1-val2;
            }
        }
        if( strcmp(op.c_str(), "m") == 0 )
        {
            if(field2)
            {
                vec1[0]= vec1[0]*vec2[0];
                vec1[1]= vec1[1]*vec2[1];
                vec1[2]= vec1[2]*vec2[2];
            }
            else
            {
                vec1=vec1*val2;
            }
        }
        if( strcmp(op.c_str(), "/") == 0 )
        {
            if(field2)
            {
                vec1[0]= vec1[0]/vec2[0];
                vec1[1]= vec1[1]/vec2[1];
                vec1[2]= vec1[2]/vec2[2];
            }
            else
            {
                vec1=vec1*val2;
            }
        }

        it.Set(vec1);
    }

    writeImageD<DisplacementFieldType>(output_field,argv[1]);

    
    return EXIT_SUCCESS;
}
