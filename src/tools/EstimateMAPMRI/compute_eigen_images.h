#ifndef _ComputeEigenImages_H
#define _ComputeEigenImages_H

#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"


EVecImageType::Pointer  ComputeEigenImages(DTImageType::Pointer dt_image,  EValImageType::Pointer & eval_image)
{
    EVecImageType::Pointer evec_image = EVecImageType::New();
    evec_image->SetRegions(dt_image->GetLargestPossibleRegion());
    evec_image->Allocate();
    evec_image->SetOrigin(dt_image->GetOrigin());
    evec_image->SetDirection(dt_image->GetDirection());
    evec_image->SetSpacing(dt_image->GetSpacing());
    EVecType zerovec;
    zerovec.set_identity();
    evec_image->FillBuffer(zerovec);



    eval_image = EValImageType::New();
    eval_image->SetRegions(dt_image->GetLargestPossibleRegion());
    eval_image->Allocate();
    eval_image->SetOrigin(dt_image->GetOrigin());
    eval_image->SetDirection(dt_image->GetDirection());
    eval_image->SetSpacing(dt_image->GetSpacing());

    EValType zeroval; zeroval.Fill(0);

    itk::ImageRegionIteratorWithIndex<EVecImageType> it(evec_image,evec_image->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
        EVecImageType::IndexType ind=it.GetIndex();
        DTType vec=  dt_image->GetPixel(ind);

        if(vec[0]+ vec[3]+ vec[5] !=0)
        {


            EVecType tensor;
            tensor(0,0)= vec[0]; tensor(0,1)= vec[1]; tensor(0,2)= vec[2];
            tensor(1,0)= vec[1]; tensor(1,1)= vec[3]; tensor(1,2)= vec[4];
            tensor(2,0)= vec[2]; tensor(2,1)= vec[4]; tensor(2,2)= vec[5];


            vnl_symmetric_eigensystem<double> eig(tensor);

            EVecType evec;
            evec.set_column(0, eig.V.get_column(2));
            evec.set_column(1, eig.V.get_column(1));
            evec.set_column(2, eig.V.get_column(0));

            double mdet= vnl_determinant<double>( evec);
            if(mdet<0)
            {
                evec.set_column(2, -1.* evec.get_column(2));
            }

            evec=evec.transpose();

            EValType eval;

            if(eig.D(0,0)< 0)
                eig.D(0,0)=0.000000000001;
            if(eig.D(1,1)< 0)
                eig.D(1,1)=0.000000000001;
            if(eig.D(2,2)< 0)
                eig.D(2,2)=0.000000000001;

            eval[2]= eig.D(0,0);
            eval[1]= eig.D(1,1);
            eval[0]= eig.D(2,2);

            it.Set(evec);
            eval_image->SetPixel(ind,eval);
        }
        else
        {
            eval_image->SetPixel(ind,zeroval);
            evec_image->SetPixel(ind,zerovec);
        }
        ++it;
    }

    return evec_image;
}







#endif

