#ifndef _ROTATE_BMATRIXMAIN_CXX
#define _ROTATE_BMATRIXMAIN_CXX


#include "rotate_bmatrix.h"
#include "../utilities/read_bmatrix_file.h"
#include "itkTransformFileReader.h"
#include "../utilities/read_3Dvolume_from_4D.h"

int main(int argc, char* argv[])
{
    if(argc==1)
    {
        std::cout<<"Usage: RotateBmatrix bmat_file trans_file corresponding_image_that_will_be_rotated"<<std::endl;
        return EXIT_FAILURE;
    }

    vnl_matrix<double> Bmatrix =read_bmatrix_file(argv[1]);

    std::ifstream intrans;
    intrans.open(argv[2]);
    std::string line;
    std::getline(intrans,line);
    intrans.close();


    ImageType3D::Pointer ref_img=read_3D_volume_from_4D(argv[3],0);

    vnl_matrix_fixed<double,3,3> dirmat=ref_img->GetDirection().GetVnlMatrix();


    vnl_matrix_fixed<double,3,3> rot_mat;

    if(line.find("#Insight Transform File")!=std::string::npos)
    {
        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(argv[2] );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();

        RigidTransformType::Pointer rigid_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
        AffineTransformType::Pointer affine_trans = static_cast< AffineTransformType * >( (*it).GetPointer() );

        if(rigid_trans)
            rot_mat=rigid_trans->GetMatrix().GetVnlMatrix();
        else
            rot_mat=affine_trans->GetMatrix().GetVnlMatrix();
    }
    else
    {

    }

    vnl_matrix<double> newBmat= RotateBMatrix(Bmatrix,rot_mat,dirmat);

    std::string bmat_name= argv[1];
    std::string new_bmat_name=  bmat_name.substr(0,bmat_name.find(".bmtxt")) + "_rot.bmtxt";
    std::ofstream outfile(new_bmat_name);
    outfile<<newBmat;
    outfile.close();
}





#endif

