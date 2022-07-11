#ifndef _SynthesizeDWIFromTensor_H
#define _SynthesizeDWIFromTensor_H


#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"

#include <math.h>





ImageType3D::Pointer  SynthesizeDWIFromTensor(DTImageType::Pointer dt_image, ImageType3D::Pointer A0_image, vnl_vector<double> bmatrix_vec )
{
    ImageType3D::Pointer synth_image = ImageType3D::New();
    synth_image->SetRegions(A0_image->GetLargestPossibleRegion());
    synth_image->Allocate();
    synth_image->SetOrigin(A0_image->GetOrigin());
    synth_image->SetDirection(A0_image->GetDirection());

    synth_image->SetSpacing(A0_image->GetSpacing());
    synth_image->FillBuffer(0.);



    itk::ImageRegionIteratorWithIndex<ImageType3D> it(synth_image,synth_image->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind=it.GetIndex();
        DTType tensor= dt_image->GetPixel(ind);
        double exp_term=0;
        for(int i=0;i<6;i++)
            exp_term += tensor[i] * bmatrix_vec[i];


        if(exp_term < 0)
            exp_term=0;

         float A0val= A0_image->GetPixel(ind);

        float signal = A0val * exp(-exp_term);
        if(!isnan(signal) && isfinite(signal))
            it.Set(signal);
        ++it;
    }

    return synth_image;
}







#endif
