

#include <iostream>
#include <fstream>
using namespace std;

#include "CombineALLTransformations_parser.h"
#include "defines.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"

#include "itkTransformFileReader.h"
#include "itkOkanQuadraticTransform.h"
#include "itkCompositeTransform.h"
#include "itkEuler3DTransform.h"
#include "itkImageDuplicator.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"


using CoordType=double;
using OkanQuadraticTransformType=itk::OkanQuadraticTransform<CoordType,3,3>;
using DisplacementFieldTransformType= itk::DisplacementFieldTransform<CoordType,3> ;
using DisplacementFieldType= DisplacementFieldTransformType::DisplacementFieldType;
using CompositeTransformType= itk::CompositeTransform<CoordType,3>;
using RigidTransformType= itk::Euler3DTransform<CoordType>;



ImageType3D::Pointer ChangeImageHeaderToDP(ImageType3D::Pointer img,std::string rot_center)
{


    /*********************************************************************************
     We are doing this for several reasons.
     1:  We have to operate on Read/Phase/Slice coordinate system, not x/y/z like ITK.
         So we set the image direction to identity.
         Yes this causes an inconsistency between images and transformations so
         we have to be really careful everytime we register/transform an image.
     2:  Eddy currents do not affect the scanner isocenter (besides a translation which is accounted for in motion correction).
         If the image header is correct and the image coordinate (0,0,0) is indeed the scanner isocenter
         we should use that one.
         But if the header is wrong, we can use the closest thing which is the center voxel of the image.
    **************************************************************************************/

    // do not want to touch the original image so we duplicate it
    using DupType= itk::ImageDuplicator<ImageType3D>;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(img);
    dup->Update();
    ImageType3D::Pointer nimg= dup->GetOutput();

    ImageType3D::DirectionType id_dir;     id_dir.SetIdentity();
    nimg->SetDirection(id_dir);
    ImageType3D::PointType new_orig;

    if(rot_center=="isocenter")
    {
        // The center is the isocenter.
        // If we werent changing the Direction matrix, we would not have to do ANYTHING here.
        // But we are, so keep  the same location as the (0,0,0) coordinate with the new Id direction matrix.

        vnl_matrix<double> Sinv(3,3,0);
        Sinv(0,0)= 1./img->GetSpacing()[0];
        Sinv(1,1)= 1./img->GetSpacing()[1];
        Sinv(2,2)= 1./img->GetSpacing()[2];
        vnl_matrix<double> S(3,3,0);
        S(0,0)= img->GetSpacing()[0];
        S(1,1)= img->GetSpacing()[1];
        S(2,2)= img->GetSpacing()[2];

        vnl_vector<double> indo= Sinv*img->GetDirection().GetTranspose() * (-1.*img->GetOrigin().GetVnlVector());   //this is the continuous index (i,j,k) of the isocenter
        vnl_vector<double> new_orig_v= -S*indo;
        new_orig[0]=new_orig_v[0];
        new_orig[1]=new_orig_v[1];
        new_orig[2]=new_orig_v[2];
    }
    else
    {
        if(rot_center=="center_voxel")
        {
            //Make the rotation and eddy center the image center voxel.
            new_orig[0]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[0])-1)/2. * img->GetSpacing()[0];
            new_orig[1]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[1])-1)/2. * img->GetSpacing()[1];
            new_orig[2]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[2])-1)/2. * img->GetSpacing()[2];
        }
        else
        {
            //center_slice
            vnl_matrix<double> Sinv(3,3,0);
            Sinv(0,0)= 1./img->GetSpacing()[0];
            Sinv(1,1)= 1./img->GetSpacing()[1];
            Sinv(2,2)= 1./img->GetSpacing()[2];
            vnl_matrix<double> S(3,3,0);
            S(0,0)= img->GetSpacing()[0];
            S(1,1)= img->GetSpacing()[1];
            S(2,2)= img->GetSpacing()[2];


            vnl_vector<double> center_voxel_index(3,0);
            center_voxel_index[0]= ((int)(img->GetLargestPossibleRegion().GetSize()[0])-1)/2.;
            center_voxel_index[1]= ((int)(img->GetLargestPossibleRegion().GetSize()[1])-1)/2.;
            center_voxel_index[2]= ((int)(img->GetLargestPossibleRegion().GetSize()[2])-1)/2.;

            vnl_vector<double> center_voxel_point = img->GetDirection().GetVnlMatrix()*S*center_voxel_index + img->GetOrigin().GetVnlVector();

            vnl_vector<double> center_point(3,0);
            center_point[2]= center_voxel_point[2];

            vnl_vector<double> indo= Sinv*img->GetDirection().GetTranspose() * (center_voxel_point- img->GetOrigin().GetVnlVector());   //this is the continuous index (i,j,k) of the isocenter
            vnl_vector<double> new_orig_v= -S*indo;
            new_orig[0]=new_orig_v[0];
            new_orig[1]=new_orig_v[1];
            new_orig[2]=new_orig_v[2];
        }

    }
    nimg->SetOrigin(new_orig);

    return nimg;

}


CompositeTransformType::Pointer  GenerateCompositeTransformForVolume(ImageType3D::Pointer ref_img, int vol, ImageType3D::Pointer template_structural,
                                                                    std::string rot_center,
                                                                    RigidTransformType::Pointer b0_t0_str_trans,
                                                                     RigidTransformType::Pointer b0down_t0_b0up_trans,
                                                                     DisplacementFieldType::Pointer  epi_field,
                                                                     DisplacementFieldType::Pointer  gradwarp_field,
                                                                     std::vector<OkanQuadraticTransformType::Pointer> dwi_transforms
                                                                                )
{
    CompositeTransformType::Pointer all_trans= CompositeTransformType::New();
    ImageType3D::Pointer ref_img_DP= ChangeImageHeaderToDP(ref_img,rot_center);

    DisplacementFieldTransformType::Pointer gradwarp_trans=nullptr;
    if(gradwarp_field)
    {
        gradwarp_trans= DisplacementFieldTransformType::New();
        gradwarp_trans->SetDisplacementField(gradwarp_field);
    }

    DisplacementFieldTransformType::Pointer epi_trans=nullptr;
    if(epi_field)
    {
        epi_trans= DisplacementFieldTransformType::New();
        epi_trans->SetDisplacementField(epi_field);
    }

    DisplacementFieldType::Pointer total_field= DisplacementFieldType::New();
    total_field->SetRegions(template_structural->GetLargestPossibleRegion());
    total_field->Allocate();
    DisplacementFieldType::PixelType zero; zero.Fill(0);
    total_field->FillBuffer(zero);
    total_field->SetSpacing(template_structural->GetSpacing());
    total_field->SetOrigin(template_structural->GetOrigin());
    total_field->SetDirection(template_structural->GetDirection());

    using LinDispInterpolatorType = itk::LinearInterpolateImageFunction<DisplacementFieldType,double>;
    LinDispInterpolatorType::Pointer field_interpolator=LinDispInterpolatorType::New();
    field_interpolator->SetInputImage(epi_field);


    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(total_field,total_field->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        ImageType3D::PointType pt,pt_trans;
        total_field->TransformIndexToPhysicalPoint(ind3,pt);
        pt_trans=pt;

        if(b0_t0_str_trans)
        {
            pt_trans=b0_t0_str_trans->TransformPoint(pt_trans);

        }
        if(epi_trans)
        {
            pt_trans=epi_trans->TransformPoint(pt_trans);
        }
        if(b0down_t0_b0up_trans)
        {
            pt_trans=b0down_t0_b0up_trans->TransformPoint(pt_trans);
        }
        if(gradwarp_trans)
        {
            pt_trans=gradwarp_trans->TransformPoint(pt_trans);
        }

        if(dwi_transforms.size())
        {
            itk::ContinuousIndex<double,3> cind3;
            ImageType3D::PointType pt_trans2;
            ref_img->TransformPhysicalPointToContinuousIndex(pt_trans,cind3);
            ref_img_DP->TransformContinuousIndexToPhysicalPoint(cind3,pt_trans2);

            OkanQuadraticTransformType::Pointer curr_mot_eddy_trans= dwi_transforms[vol];
            pt_trans2= curr_mot_eddy_trans->TransformPoint(pt_trans2);
            ref_img_DP->TransformPhysicalPointToContinuousIndex(pt_trans2,cind3);
            ref_img->TransformContinuousIndexToPhysicalPoint(cind3,pt_trans);
        }

        DisplacementFieldType::PixelType vec;
        vec[0]=pt_trans[0]-pt[0];
        vec[1]=pt_trans[1]-pt[1];
        vec[2]=pt_trans[2]-pt[2];
        it.Set(vec);
    } //for voxels


    DisplacementFieldTransformType::Pointer total_trans=DisplacementFieldTransformType::New();
    total_trans->SetDisplacementField(total_field);
    all_trans->AddTransform(total_trans);


    return all_trans;
}

std::vector<ImageType3D::Pointer> ReadAndCombineTransformations(CombineALLTransformations_PARSER * parser)
{
    std::string b02str_name = parser->getBo2StrTransformationName();
    RigidTransformType::Pointer b0_t0_str_trans=nullptr;
    if(b02str_name!="" && fs::exists(b02str_name))
    {
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(b02str_name );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();
        b0_t0_str_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
    }

    std::string b0down_t0_b0up_trans_name = parser->getBlipDownToUpTransformationName();
    RigidTransformType::Pointer b0down_t0_b0up_trans=nullptr;
    if(b0down_t0_b0up_trans_name!="" && fs::exists(b0down_t0_b0up_trans_name))
    {
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(b0down_t0_b0up_trans_name);
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();
        b0down_t0_b0up_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
    }

    std::string epi_name= parser->getEPITransformationName();
    DisplacementFieldType::Pointer epi_field=nullptr;
    if(epi_name!="" && fs::exists(epi_name) )
    {
        epi_field=readImageD<DisplacementFieldType>(epi_name);
    }

    std::string gnonlin_name= parser->getGradientNonlinearityTransformationName();
    DisplacementFieldType::Pointer gnonlin_field=nullptr;
    if(gnonlin_name!="" && fs::exists(gnonlin_name))
    {
        gnonlin_field=readImageD<DisplacementFieldType>(gnonlin_name);
    }


    ImageType4D::Pointer input_img = readImageD<ImageType4D>(parser->getInputImageName());
    int Nvols = input_img->GetLargestPossibleRegion().GetSize()[3];

    std::vector<OkanQuadraticTransformType::Pointer>  dwi_transforms;
    {


        std::string moteddy_name=parser->getMotionEddyParametersName();
        if(moteddy_name!="" && fs::exists(moteddy_name))
        {
            std::string PE = parser->getPE();
            if(PE=="")
            {
                std::cout<< "Phase encoding must be provided if mot eddy file is input. Exiting..."<<std::endl;
                exit(EXIT_FAILURE);
            }

            std::ifstream moteddy_text_file(moteddy_name);
            for( int vol=0; vol<Nvols;vol++)
            {
                std::string line;
                std::getline(moteddy_text_file,line);

                OkanQuadraticTransformType::Pointer quad_trans= OkanQuadraticTransformType::New();
                quad_trans->SetPhase(PE);
                quad_trans->SetIdentity();

                OkanQuadraticTransformType::ParametersType params=quad_trans->GetParameters();
                line=line.substr(1);
                for(int p=0;p<OkanQuadraticTransformType::NQUADPARAMS;p++)
                {
                    int npos = line.find(", ");
                    std::string curr_p_string = line.substr(0,npos);

                    double val = atof(curr_p_string.c_str());
                    params[p]=val;
                    line=line.substr(npos+2);
                }
                quad_trans->SetParameters(params);
                OkanQuadraticTransformType::ParametersType flags;
                flags.SetSize(OkanQuadraticTransformType::NQUADPARAMS);
                flags.Fill(0);
                flags[0]=flags[1]=flags[2]=flags[3]=flags[4]=flags[5]=1;
                quad_trans->SetParametersForOptimizationFlags(flags);
                dwi_transforms[vol]    = quad_trans;
            }
            moteddy_text_file.close();
        }
    }

    ImageType3D::Pointer template_structural= readImageD<ImageType3D>(parser->getTemplateName());


    std::string rot_center= parser->getRotCenter();

    std::vector<ImageType3D::Pointer> final_data;

    ImageType4D::Pointer temp = readImageD<ImageType4D>(parser->getInputImageName());

    for(int vol=0;vol<Nvols;vol++)
    {
        ImageType4D::IndexType ind4;
        ind4[3]=vol;

        ImageType3D::Pointer curr_img=nullptr;
        if(temp->GetLargestPossibleRegion().GetSize()[3]!=1)
            curr_img= read_3D_volume_from_4D(parser->getInputImageName(),vol);
        else
            curr_img=readImageD<ImageType3D>(parser->getInputImageName());


        CompositeTransformType::Pointer all_trans= GenerateCompositeTransformForVolume(curr_img, vol, template_structural,
        rot_center,
        b0_t0_str_trans,
        b0down_t0_b0up_trans,
        epi_field,
        gnonlin_field,
        dwi_transforms
        );

        using BSInterpolatorType = itk::BSplineInterpolateImageFunction<ImageType3D, double>;
        BSInterpolatorType::Pointer BSinterpolator = BSInterpolatorType::New();
        BSinterpolator->SetSplineOrder(3);
        BSinterpolator->SetInputImage(curr_img);

        using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D,ImageType3D,double>;
        ResampleImageFilterType::Pointer resampler = ResampleImageFilterType::New();
        resampler->SetOutputParametersFromImage(template_structural);
        resampler->SetInput(curr_img);
        resampler->SetTransform( all_trans);
        resampler->SetDefaultPixelValue(0);
        resampler->Update();
        ImageType3D::Pointer final_img= resampler->GetOutput();

        final_data.push_back(final_img);
    }

    return final_data;


}


int main( int argc , char * argv[] )
{

    CombineALLTransformations_PARSER *parser = new CombineALLTransformations_PARSER(argc,argv);
    
    std::vector<ImageType3D::Pointer> trans_imgs = ReadAndCombineTransformations(parser);

    std::string output_name = parser->getOutputName();
    int Nvols = trans_imgs.size();

    for(int vol=0;vol<Nvols;vol++)
    {
        if(Nvols==1)
            writeImageD<ImageType3D>(trans_imgs[vol],output_name);
        else
            write_3D_image_to_4D_file<float> (trans_imgs[vol],output_name,vol,Nvols);
    }

    
    return EXIT_SUCCESS;
}
