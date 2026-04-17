

#include <iostream>
#include <fstream>
using namespace std;

#include "ComputeRotationMatrixFromTORTOISETransformations_parser.h"
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
#include "../tools/RotateBMatrix/rotate_bmatrix.h"


using CoordType=double;
using OkanQuadraticTransformType=itk::OkanQuadraticTransform<CoordType,3,3>;
using DisplacementFieldTransformType= itk::DisplacementFieldTransform<CoordType,3> ;
using DisplacementFieldType= DisplacementFieldTransformType::DisplacementFieldType;
using CompositeTransformType= itk::CompositeTransform<CoordType,3>;
using RigidTransformType= itk::Euler3DTransform<CoordType>;


int main( int argc , char * argv[] )
{

    ComputeRotationMatrixFromTORTOISETransformations_PARSER *parser = new ComputeRotationMatrixFromTORTOISETransformations_PARSER(argc,argv);


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



   // ImageType4D::Pointer input_img = readImageD<ImageType4D>(parser->getInputImageName());
   // int Nvols = input_img->GetLargestPossibleRegion().GetSize()[3];


    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(parser->getInputImageName().c_str(), itk::ImageIOFactory::ReadMode);
    imageIO->SetFileName(parser->getInputImageName());
    imageIO->ReadImageInformation();
    int Nvols=1;
    if (imageIO->GetNumberOfDimensions() > 3)
    {
        Nvols = imageIO->GetDimensions(3); // 4th dimension (time/volumes)
    }

    std::vector<OkanQuadraticTransformType::Pointer>  dwi_transforms;
    dwi_transforms.resize(Nvols);
    {
        std::string moteddy_name=parser->getMotionEddyParametersName();
        if(moteddy_name!="" && fs::exists(moteddy_name))
        {
            std::ifstream moteddy_text_file(moteddy_name);
            for( int vol=0; vol<Nvols;vol++)
            {
                std::string line;
                std::getline(moteddy_text_file,line);

                OkanQuadraticTransformType::Pointer quad_trans= OkanQuadraticTransformType::New();
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
    ImageType3D::Pointer vol0 = read_3D_volume_from_4D(parser->getInputImageName(),0);
    ImageType3D::DirectionType dir = vol0->GetDirection();
    vnl_matrix_fixed<double,3,3> dirm=dir.GetVnlMatrix();

    vnl_matrix_fixed<double,3,3> id_trans; id_trans.set_identity();
    vnl_matrix_fixed<double,3,3> rotmat;
    rotmat.set_identity();

    {
        if(dwi_transforms.size())
        {
            int vol_id = parser->getVolIndex();
            OkanQuadraticTransformType::Pointer quad_trans= dwi_transforms[vol_id];
            rotmat=quad_trans->GetMatrix().GetVnlMatrix();
        }
    }

    {
        if(b0down_t0_b0up_trans)
        {
            vnl_matrix_fixed<double,3,3> rotmat2= b0down_t0_b0up_trans->GetMatrix().GetVnlMatrix();
            vnl_matrix_fixed<double,3,3> rotmat3= dirm.transpose() * rotmat2 * dirm;

            rotmat= rotmat3 *rotmat;
        }
    }
    {
        if(b0_t0_str_trans)
        {
            vnl_matrix_fixed<double,3,3> fixed_dirmat= template_structural->GetDirection().GetVnlMatrix();
            vnl_matrix_fixed<double,3,3> moving_dirmat = dirm;

            vnl_matrix_fixed<double,3,3> rotmat2= b0_t0_str_trans->GetMatrix().GetVnlMatrix();
            vnl_matrix_fixed<double,3,3> rotmat3= moving_dirmat.transpose() * rotmat2 * fixed_dirmat;

            rotmat= rotmat3 *rotmat;
        }
    }

    std::cout<<"Rotation matrix for volume " <<parser->getVolIndex() << ": "<<std::endl;
    std::cout<<rotmat(0,0)<< " "<<rotmat(0,1)<< " "<<rotmat(0,2)<< std::endl;
    std::cout<<rotmat(1,0)<< " "<<rotmat(1,1)<< " "<<rotmat(1,2)<< std::endl;
    std::cout<<rotmat(2,0)<< " "<<rotmat(2,1)<< " "<<rotmat(2,2)<< std::endl;


    return EXIT_SUCCESS;
}
