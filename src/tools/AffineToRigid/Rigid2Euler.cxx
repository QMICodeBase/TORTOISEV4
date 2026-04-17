

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;


#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkImage.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "vnl/vnl_cross.h"

#include "itkEuler3DTransform.h"

using CoordType=double;
using RigidTransformType= itk::Euler3DTransform<CoordType>;
using AffineTransformType= itk::AffineTransform<CoordType,3>;

int main( int argc , char * argv[] )
{
    if(argc<2)
    {
        std::cout<<"Usage:   Rigid2Euler   input_rigid_transform "<<std::endl;
        return EXIT_FAILURE;
    }
    
    
    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string extension = filename.substr(idx+1);

    RigidTransformType::Pointer rigid_trans=nullptr;

    if(extension=="hdf5")
    {
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(argv[1] );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();
        rigid_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
    }
    else
    {
        AffineTransformType::Pointer affine_trans=nullptr;
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(argv[1] );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
        affine_trans = static_cast<AffineTransformType*>((*it).GetPointer());

        vnl_matrix<double>  A= affine_trans->GetMatrix().GetVnlMatrix();

        auto AAT = A * A.transpose();
        vnl_symmetric_eigensystem<double> eig(AAT);
        eig.D(0,0)= pow(eig.D(0,0), -0.5);
        eig.D(1,1)= pow(eig.D(1,1), -0.5);
        eig.D(2,2)= pow(eig.D(2,2), -0.5);

        auto AAT_sq_inv = eig.recompose();
        auto rigid_mat= AAT_sq_inv * A;

        AffineTransformType::Pointer rigid_trans= AffineTransformType::New();
        AffineTransformType::MatrixType trans_rigid_mat;
        trans_rigid_mat(0,0)= rigid_mat(0,0);trans_rigid_mat(0,1)= rigid_mat(0,1);trans_rigid_mat(0,2)= rigid_mat(0,2);
        trans_rigid_mat(1,0)= rigid_mat(1,0);trans_rigid_mat(1,1)= rigid_mat(1,1);trans_rigid_mat(1,2)= rigid_mat(1,2);
        trans_rigid_mat(2,0)= rigid_mat(2,0);trans_rigid_mat(2,1)= rigid_mat(2,1);trans_rigid_mat(2,2)= rigid_mat(2,2);

        rigid_trans->SetMatrix(trans_rigid_mat);
        rigid_trans->SetOffset(affine_trans->GetOffset());
        rigid_trans->SetFixedParameters(affine_trans->GetFixedParameters());
    }


    std::string oname = filename + ".euler";

    std::ofstream outFile(oname);
    if (outFile.is_open())
    {
        outFile<<"Angles in degrees (X, Y, Z):"<<std::endl;
        outFile<< rigid_trans->GetAngleX()/M_PI*180 << " "<< rigid_trans->GetAngleY()/M_PI*180 << " "<< rigid_trans->GetAngleZ()/M_PI*180 << " "<<std::endl;
        outFile.close();
        std::cout<<"Angles in degrees (X, Y, Z):"<<std::endl;
        std::cout<< rigid_trans->GetAngleX()/M_PI*180 << " "<< rigid_trans->GetAngleY()/M_PI*180 << " "<< rigid_trans->GetAngleZ()/M_PI*180 << " "<<std::endl;
    }
    else
    {
        std::cout<<"Couldn't open file for writing..."<<std::endl;
    }




    
    return EXIT_SUCCESS;
}

