#ifndef _ERODEMASK_CXX
#define _ERODEMASK_CXX

#include "erodeMask.h"



MaskImageType::Pointer computeMask(ImageType3D::Pointer b0_image,
                                   float threshold = 0.1){
    // Compute global-maximum in entired image
    itk::ImageRegionIteratorWithIndex<ImageType3D> imageIterator(b0_image,
                                                                    b0_image->GetLargestPossibleRegion());
    imageIterator.GoToBegin();
    float mx = -1E10;
    while(!imageIterator.IsAtEnd()){
        float currentValue = imageIterator.Get();
        if(currentValue > mx) {
            mx = currentValue;
        }
        ++imageIterator;
    }

    // Create mask, based on Threshold * mx
    MaskImageType::Pointer mask_image= MaskImageType::New();
    mask_image->SetRegions(b0_image->GetLargestPossibleRegion());
    mask_image->Allocate();
    mask_image->SetOrigin(b0_image->GetOrigin());
    mask_image->SetSpacing(b0_image->GetSpacing());
    mask_image->SetDirection(b0_image->GetDirection());



    // Create the iterator for thresholding image
    itk::ImageRegionIteratorWithIndex<MaskImageType> maskIterator(mask_image,
                                                                    mask_image->GetLargestPossibleRegion());
    maskIterator.GoToBegin();
    ImageType3D::IndexType index;
    // Iterating over index and threshold.
    while(!maskIterator.IsAtEnd()){
        index = maskIterator.GetIndex();
        float b0_value = b0_image->GetPixel(index);
        if (b0_value >= threshold*mx){
            maskIterator.Set(1);
        }
        else{maskIterator.Set(0);}
        ++maskIterator;
    }
    return mask_image;
}


MaskImageType::Pointer erodeMask(MaskImageType::Pointer mask_image)
{

    typedef itk::FlatStructuringElement<3> StructuringElementType;
    StructuringElementType::RadiusType radius;
    radius.Fill(1);

    StructuringElementType structureElement = StructuringElementType::Box(radius);
    typedef itk::BinaryErodeImageFilter<MaskImageType, MaskImageType, StructuringElementType> ErodeImageFilterType;

    ErodeImageFilterType::Pointer erodeFilter = ErodeImageFilterType::New();
    erodeFilter->SetInput(mask_image);
    erodeFilter->SetKernel(structureElement);
    erodeFilter->SetForegroundValue(1);
    erodeFilter->SetBackgroundValue(0);
    erodeFilter->Update();

    MaskImageType::Pointer eroded_imaged = MaskImageType::New();
    eroded_imaged = erodeFilter->GetOutput();
    return eroded_imaged;
}

MaskImageType::Pointer compute_erode_Mask(ImageType3D::Pointer b0_image,
                                          int erodeFactor = 0,
                                   float threshold = 0.1){
    // Compute global-maximum in entired image
    itk::ImageRegionIteratorWithIndex<ImageType3D> imageIterator(b0_image,
                                                                    b0_image->GetLargestPossibleRegion());
    imageIterator.GoToBegin();
    float mx = -1E10;
    while(!imageIterator.IsAtEnd()){
        float currentValue = imageIterator.Get();
        if(currentValue > mx) {
            mx = currentValue;
        }
        ++imageIterator;
    }

    // Create mask, based on Threshold * mx
    MaskImageType::Pointer mask_image= MaskImageType::New();
    mask_image->SetRegions(b0_image->GetLargestPossibleRegion());
    mask_image->Allocate();
    mask_image->SetOrigin(b0_image->GetOrigin());
    mask_image->SetSpacing(b0_image->GetSpacing());
    mask_image->SetDirection(b0_image->GetDirection());



    // Create the iterator for thresholding image
    itk::ImageRegionIteratorWithIndex<MaskImageType> maskIterator(mask_image,
                                                                    mask_image->GetLargestPossibleRegion());
    maskIterator.GoToBegin();
    ImageType3D::IndexType index;
    // Iterating over index and threshold.
    while(!maskIterator.IsAtEnd()){
        index = maskIterator.GetIndex();
        float b0_value = b0_image->GetPixel(index);
        if (b0_value >= threshold*mx){
            maskIterator.Set(1);
        }
        else{maskIterator.Set(0);}
        ++maskIterator;
    }

    if (erodeFactor != 0){
        for (int i = 0; i < erodeFactor ; i++) {
            mask_image = erodeMask(mask_image);

        }
    }
    mask_image->SetRegions(b0_image->GetLargestPossibleRegion());
    mask_image->Allocate();
    mask_image->SetOrigin(b0_image->GetOrigin());
    mask_image->SetSpacing(b0_image->GetSpacing());
    mask_image->SetDirection(b0_image->GetDirection());



    return mask_image;
}




//int read_coef_file(std::string gradtxtfile){
//    std::ifstream inFile(gradtxtfile);
//    if (!inFile)
//    {
//       std::cerr << "File " << gradtxtfile << " not found." << std::endl;
//       return 0;
//    }
//    std::vector< std::vector<double>>  x_key (1,std::vector<double> (2,0));
//    std::vector< std::vector<double>>  y_key (1,std::vector<double> (2,0));
//    std::vector< std::vector<double>>  z_key (1,std::vector<double> (2,0));

//    std::string line;
//    char curr_comp='z';
//    while (std::getline(inFile, line))
//    {
//        if (line.empty())
//            continue;

//        if(line =="z")
//            continue;

//        if(line=="x")
//        {
//            curr_comp='x';
//            continue;
//        }

//        if(line=="y")
//        {
//            curr_comp='y';
//            continue;
//        }


//        std::istringstream iss(line);
//        int l, m;
//        double coeff;

//        iss >> l >> m >> coeff;
//        std::cout<<curr_comp << ": ";
//        std::cout <<coeff;

//        if(curr_comp=='x')
//        {
//            x_key[l][m]=coeff;
//        }
//        if(curr_comp=='y')
//        {
//            y_key[l][m]=coeff;
//        }
//        if(curr_comp=='z')
//        {
//            z_key[l][m]=coeff;
//        }

//     }
//    return 1;

//}



#endif
