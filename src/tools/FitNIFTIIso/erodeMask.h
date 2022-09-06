#ifndef _ERODEMASK_H
#define _ERODEMASK_H


#include "itkImageRegionIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkGrayscaleErodeImageFilter.h"
#include "itkBinaryErodeImageFilter.h"

#include "itkFlatStructuringElement.h"
#include "itkImageDuplicator.h"
#include "defines.h"


typedef unsigned char MaskPixelType;
typedef itk::Image<MaskPixelType,3> MaskImageType;



MaskImageType::Pointer computeMask(ImageType3D::Pointer b0_image,
                                   float threshold);

MaskImageType::Pointer erodeMask(MaskImageType::Pointer mask_image);

MaskImageType::Pointer compute_erode_Mask(ImageType3D::Pointer b0_image,
                                          int erodeFactor , float threshold );


/*

int read_coef_file(std::string gradtxtfile){
    std::ifstream inFile(gradtxtfile);
    if (!inFile)
    {
       std::cerr << "File " << gradtxtfile << " not found." << std::endl;
       return 0;
    }
    std::vector< std::vector<double>>  x_key (1,std::vector<double> (2,0));
    std::vector< std::vector<double>>  y_key (1,std::vector<double> (2,0));
    std::vector< std::vector<double>>  z_key (1,std::vector<double> (2,0));

    std::string line;
    char curr_comp='z';
    while (std::getline(inFile, line))
    {
        if (line.empty())
            continue;

        if(line =="z")
            continue;

        if(line=="x")
        {
            curr_comp='x';
            continue;
        }

        if(line=="y")
        {
            curr_comp='y';
            continue;
        }


        std::istringstream iss(line);
        int l, m;
        double coeff;

        iss >> l >> m >> coeff;
        std::cout<<curr_comp << ": ";
        std::cout <<coeff;

        if(curr_comp=='x')
        {
            x_key[l][m]=coeff;
        }
        if(curr_comp=='y')
        {
            y_key[l][m]=coeff;
        }
        if(curr_comp=='z')
        {
            z_key[l][m]=coeff;
        }

     }
    return 1;

}
*/


#endif
