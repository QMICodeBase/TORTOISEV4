#ifndef RIGIDREGISTERIMAGES_H
#define RIGIDREGISTERIMAGES_H


#include "TORTOISE.h"



using QuadraticTransformType=TORTOISE::OkanQuadraticTransformType;
using CompositeTransformType=TORTOISE::CompositeTransformType;
using RigidTransformType= TORTOISE::RigidTransformType;


QuadraticTransformType::Pointer CompositeLinearToQuadratic(const CompositeTransformType * compositeTransform, std::string phase);
QuadraticTransformType::Pointer RigidRegisterImages(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string metric_type="CC");
RigidTransformType::Pointer RigidRegisterImagesEuler(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string metric_type="CC",float lr=0.4,RigidTransformType::Pointer in_trans=nullptr);


RigidTransformType::Pointer MultiStartRigidSearch(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img);



#endif
