#ifndef _COMPUTEMETRICSMSJAC_H
#define _COMPUTEMETRICSMSJAC_H

#include "defines.h"





float ComputeMetric_MSJac(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, 
                          const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV   ,
                          DisplacementFieldType::Pointer &updateFieldF, DisplacementFieldType::Pointer &updateFieldM,
                          vnl_vector<double> phase_vector,float update_std)
{
    updateFieldF->SetRegions(def_FINV->GetLargestPossibleRegion());
    updateFieldF->SetDirection(this->def_FINV->GetDirection());
    updateFieldF->SetOrigin(this->def_FINV->GetOrigin());
    updateFieldF->SetSpacing(this->def_FINV->GetSpacing());
    updateFieldF->Allocate();
    CurrentFieldType::PixelType zero; zero.Fill(0);
    updateFieldF->FillBuffer(zero);

    updateFieldM->SetRegions(def_FINV->GetLargestPossibleRegion());
    updateFieldM->SetDirection(this->def_FINV->GetDirection());
    updateFieldM->SetOrigin(this->def_FINV->GetOrigin());
    updateFieldM->SetSpacing(this->def_FINV->GetSpacing());
    updateFieldM->Allocate();
    updateFieldM->FillBuffer(zero);

    int phase=0;
    if( (fabs(phase_vector[1]) > fabs(phase_vector[0])) && (fabs(phase_vector[1]) > fabs(phase_vector[2])))
        phase=1;
    if( (fabs(phase_vector[2]) > fabs(phase_vector[0])) && (fabs(phase_vector[2]) > fabs(phase_vector[1])))
        phase=2;
        
    itk::GaussianOperator<float,3> oper;
    oper.SetDirection(phase);
    oper.SetVariance(update_std);
    oper.SetMaximumError(0.001);
    oper.SetMaximumKernelWidth(31);
    oper.CreateDirectional();

    auto aa= oper.GetBufferReference();
     

    
    float metric_value;
    
    ComputeMetric_MSJac_cuda(up_img->getFloatdata(), down_img->getFloatdata(),
		   up_img->sz, up_img->spc, 
		   up_img->dir(0,0),up_img->dir(0,1),up_img->dir(0,2),up_img->dir(1,0),up_img->dir(1,1),up_img->dir(1,2),up_img->dir(2,0),up_img->dir(2,1),up_img->dir(2,2),
		   def_FINV->getFloatdata(), def_MINV->getFloatdata(),
   		   updateFieldF->getFloatdata(), updateFieldM->getFloatdata(),
        phase_vector, kernel_sz,h_kernel, metric_value
		   );

    delete[] h_kernel;

    return metric_value;
}



}


float ComputeMetric_CCSK(const ImageType3D::Pointer up_img, const ImageType3D::Pointer down_img, 
                         const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV)
{


}
                         
                         

float ComputeMetric_CC(const CUDAIMAGE::Pointer up_img, const CUDAIMAGE::Pointer down_img,
                       const DisplacementFieldType::Pointer def_FINV, const DisplacementFieldType::Pointer def_MINV  )
{


}




#endif
