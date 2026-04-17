

#include <iostream>
using namespace std;


#include "defines.h"

#include "itkTransformFileReader.h"
#include "itkEuler3DTransform.h"
#include "itkTransformFileWriter.h"

int main( int argc , char * argv[] )
{
    if(argc<2)
    {
        std::cout<<"Usage: ConvertRigidHDF2Text hdf5_file.hdf "<<std::endl;
        return EXIT_FAILURE;
    }
    
    

        using TransformReaderType =itk::TransformFileReaderTemplate< double > ;
        using RigidTransformType= itk::Euler3DTransform<double>;
        
        TransformReaderType::Pointer reader = TransformReaderType::New();
        reader->SetFileName(argv[1] );
        reader->Update();
        const TransformReaderType::TransformListType * transforms =       reader->GetTransformList();
        TransformReaderType::TransformListType::const_iterator it      = transforms->begin();
        RigidTransformType::Pointer rigid_trans = static_cast< RigidTransformType * >( (*it).GetPointer() );
        
        std::string iname(argv[1]);
        std::string oname = iname.substr(0,iname.rfind(".hdf5")) + ".txt";
        
        itk::TransformFileWriter::Pointer trwriter = itk::TransformFileWriter::New();
        trwriter->SetInput(rigid_trans);
        trwriter->SetFileName(oname);
        trwriter->Update();
        
        
    

    
    return EXIT_SUCCESS;
}
