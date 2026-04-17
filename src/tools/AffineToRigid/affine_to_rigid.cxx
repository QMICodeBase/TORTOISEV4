

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



    typedef double RealType;        
    typedef itk::Transform<RealType,3,3> TransformType;
    typedef itk::AffineTransform<RealType,3> AffineTransformType;

           

    

int main( int argc , char * argv[] )
{
    if(argc<3)
    {
        std::cout<<"Usage:   Affine2Rigid   input_affine_transform.txt  output_rigid_transform.txt "<<std::endl;
        return EXIT_FAILURE;
    }
    
    
    std::string filename(argv[2]);
    std::string::size_type idx=filename.rfind('.');
    std::string extension = filename.substr(idx+1);


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

    itk::TransformFileWriter::Pointer trwriter = itk::TransformFileWriter::New();
    trwriter->SetInput(rigid_trans);
    trwriter->SetFileName(argv[2]);
    trwriter->Update();



    
    return EXIT_SUCCESS;
}

