#include "DRTAMASRigid.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"


#include "itkImageRegionIteratorWithIndex.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "DRTAMAS_utilities_cp.h"
#include "DRTAMASRigid_Bulk.h"




void DRTAMASRigid::Process2()
{
    Step0_ReadImages();        
    Step1_RigidRegistration();
    Step2_TransformAndWriteAffineImage();

    std::cout<<"DRTAMASRigid completed sucessfully..."<<std::endl;

}

void DRTAMASRigid::Step1_RigidRegistration()
{



    DRTAMASRigid_Bulk *myDRTAMASRigid_processor = new DRTAMASRigid_Bulk;
    myDRTAMASRigid_processor->SetFixedTensor(this->fixed_tensor);
    myDRTAMASRigid_processor->SetMovingTensor(this->moving_tensor);
    myDRTAMASRigid_processor->SetParser(parser);
    myDRTAMASRigid_processor->Process();


    this->my_rigid_trans = myDRTAMASRigid_processor->GetRigidTrans();

    delete myDRTAMASRigid_processor;

}



void DRTAMASRigid::Step2_TransformAndWriteAffineImage()
{

    std::string moving_dt_name = parser->getMovingTensor();
    std::string output_nii_name = moving_dt_name.substr(0,moving_dt_name.rfind(".nii")) + "_aff.nii";

    using AffineTransformType=  TORTOISE::AffineTransformType;
    AffineTransformType::Pointer naff= AffineTransformType::New();

    naff->SetIdentity();
    naff->SetMatrix(this->my_rigid_trans->GetMatrix());
    naff->SetFixedParameters(this->my_rigid_trans->GetFixedParameters());
    naff->SetCenter(this->my_rigid_trans->GetCenter());
    naff->SetOffset(this->my_rigid_trans->GetOffset());
    naff->SetTranslation(this->my_rigid_trans->GetTranslation());


    TransformAndWriteAffineImage(this->moving_tensor,naff, this->fixed_tensor, output_nii_name);
}



void DRTAMASRigid::Step0_ReadImages()
{

    std::string fixed_tensor_fname = parser->getFixedTensor();
    this->fixed_tensor = ReadAndOrientTensor(fixed_tensor_fname);

    std::string moving_tensor_fname = parser->getMovingTensor();
    this->moving_tensor = ReadAndOrientTensor(moving_tensor_fname);
}
