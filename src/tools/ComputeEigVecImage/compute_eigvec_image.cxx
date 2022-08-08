

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "itkDiffusionTensor3D.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"


    typedef double RealType;     
    typedef itk::Image<RealType,4> ImageType4D;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::DiffusionTensor3D<RealType> TensorPixelType;
    typedef itk::Vector<RealType,6> VectorPixelType;
    
    typedef itk::Image<itk::DiffusionTensor3D<RealType>,3>         TensorImageType;
    typedef itk::Image<VectorPixelType,3>         VectorImageType;
    
    typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;
    
    
           

int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<<"Usage:   ComputeEigVecImage full_path_to_tensor_image"<<std::endl;
        return 0;
    }
    
    
    std::string currdir;
    std::string nm(argv[1]);
    
   
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    

    typedef itk::ImageFileReader<ImageType4D> ImageType4DReaderType;
    ImageType4DReaderType::Pointer imager= ImageType4DReaderType::New();
    imager->SetFileName(nm);
    imager->Update();
    ImageType4D::Pointer image4D= imager->GetOutput();
    
    
    ImageType4D::SizeType imsize= image4D->GetLargestPossibleRegion().GetSize();
    ImageType4D::IndexType index;
    index.Fill(0);
    
    
    ImageType4D::SizeType nsize;
    nsize[0]=imsize[0];
    nsize[1]=imsize[1];
    nsize[2]=imsize[2];
    nsize[3]=9;


    ImageType4D::RegionType reg(index,nsize);

    ImageType4D::Pointer vec_image = ImageType4D::New();
    vec_image->SetRegions(reg);
    vec_image->Allocate();
    vec_image->SetOrigin(image4D->GetOrigin());
    vec_image->SetSpacing(image4D->GetSpacing());
    vec_image->SetDirection(image4D->GetDirection());
    vec_image->FillBuffer(0.);
    
    
    
    ImageType3D::IndexType ind;
    
    for(int k=0;k<imsize[2];k++)
    {
        index[2]=k;        
        ind[2]=k;
        for(int j=0;j<imsize[1];j++)
        {
            index[1]=j;            
            ind[1]=j;
            for(int i=0;i<imsize[0];i++)
            {
                index[0]=i;    
                ind[0]=i;
                
                InternalMatrixType curr_tens;
                
                index[3]=0;
                curr_tens(0,0)=image4D->GetPixel(index);
                index[3]=1;
                curr_tens(1,1)=image4D->GetPixel(index);
                index[3]=2;
                curr_tens(2,2)=image4D->GetPixel(index);
                index[3]=3;
                curr_tens(0,1)=image4D->GetPixel(index);
                curr_tens(1,0)=image4D->GetPixel(index);
                index[3]=4;
                curr_tens(0,2)=image4D->GetPixel(index);
                curr_tens(2,0)=image4D->GetPixel(index);
                index[3]=5;
                curr_tens(1,2)=image4D->GetPixel(index);
                curr_tens(2,1)=image4D->GetPixel(index);                                   
                
                if( (curr_tens(0,0)!=curr_tens(0,0)) ||
                    (curr_tens(0,1)!=curr_tens(0,1)) ||
                    (curr_tens(0,2)!=curr_tens(0,2)) ||
                    (curr_tens(1,1)!=curr_tens(1,1)) ||
                    (curr_tens(1,2)!=curr_tens(1,2)) ||
                    (curr_tens(2,2)!=curr_tens(2,2)))
                    continue;

                if(curr_tens(0,0)+curr_tens(1,1)+curr_tens(2,2)  <1)
                    continue;

                vnl_symmetric_eigensystem<double>  eig(curr_tens);

                vnl_vector<double> e1= eig.get_eigenvector(2);
                vnl_vector<double> e2= eig.get_eigenvector(1);
                vnl_vector<double> e3= eig.get_eigenvector(0);

                double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
                double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
                double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);
                

                index[3]=0;
                vec_image->SetPixel(index, e1[0]);
                index[3]=1;
                vec_image->SetPixel(index, e1[1]);
                index[3]=2;
                vec_image->SetPixel(index, e1[2]);

                index[3]=3;
                vec_image->SetPixel(index, e2[0]);
                index[3]=4;
                vec_image->SetPixel(index, e2[1]);
                index[3]=5;
                vec_image->SetPixel(index, e2[2]);

                index[3]=6;
                vec_image->SetPixel(index, e3[0]);
                index[3]=7;
                vec_image->SetPixel(index, e3[1]);
                index[3]=8;
                vec_image->SetPixel(index, e3[2]);


                                                 
            }   
        }   
    }
    
       
    
       
        
         std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_EV.nii");
       
    
    
    typedef itk::ImageFileWriter<ImageType4D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(vec_image);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
