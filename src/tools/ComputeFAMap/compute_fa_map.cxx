

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
#include "itkImageDuplicator.h"


    typedef double RealType;     
    typedef itk::Image<RealType,4> ImageType4D;
    typedef itk::Image<RealType,3> ImageType3D;
    typedef itk::DiffusionTensor3D<RealType> TensorPixelType;
    typedef itk::Vector<RealType,6> VectorPixelType;
    
    typedef itk::Image<itk::DiffusionTensor3D<RealType>,3>         TensorImageType;
    typedef itk::Image<VectorPixelType,3>         VectorImageType;
    
    typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;
    

float median(std::vector<float> &v)
{
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

           

int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<<"Usage:   ComputeFAMap full_path_to_tensor_image filter_outliers (0/1) (optional)"<<std::endl;
        return 0;
    }
    
    bool filter=false;
    if(argc>2)
        filter =(bool)(atoi(argv[2]));
    
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
    
    
    ImageType3D::SizeType nsize;
    nsize[0]=imsize[0];
    nsize[1]=imsize[1];
    nsize[2]=imsize[2];
    
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,nsize);
    ImageType3D::PointType norig;
    norig[0]=image4D->GetOrigin()[0];
    norig[1]=image4D->GetOrigin()[1];
    norig[2]=image4D->GetOrigin()[2];
    
    ImageType3D::SpacingType nspc;
    nspc[0]=image4D->GetSpacing()[0];
    nspc[1]=image4D->GetSpacing()[1];
    nspc[2]=image4D->GetSpacing()[2];
    
    ImageType3D::DirectionType ndir;
    ImageType4D::DirectionType dir = image4D->GetDirection();
    ndir(0,0)=dir(0,0);ndir(0,1)=dir(0,1);ndir(0,2)=dir(0,2);
    ndir(1,0)=dir(1,0);ndir(1,1)=dir(1,1);ndir(1,2)=dir(1,2);
    ndir(2,0)=dir(2,0);ndir(2,1)=dir(2,1);ndir(2,2)=dir(2,2);          
    
    

    ImageType3D::Pointer FAimage = ImageType3D::New();
    FAimage->SetRegions(reg);
    FAimage->Allocate();
    FAimage->SetOrigin(norig);
    FAimage->SetSpacing(nspc);
    FAimage->SetDirection(ndir);
    FAimage->FillBuffer(0.);

    ImageType3D::Pointer outlier_image;
    outlier_image = ImageType3D::New();
    outlier_image->SetRegions(reg);
    outlier_image->Allocate();
    outlier_image->SetOrigin(norig);
    outlier_image->SetSpacing(nspc);
    outlier_image->SetDirection(ndir);
    outlier_image->FillBuffer(0.);

    
    
    
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

                if(filter)
                {
                    if( (eig.D(0,0)<0) && (eig.D(1,1)>=0) && (eig.D(2,2)>=0))
                    {
                        float val =(eig.D(0,0) + eig.D(1,1))/2;
                        eig.D(0,0)=val;
                        eig.D(1,1)=val;
                        outlier_image->SetPixel(ind,1);
                    }
                    else
                    {
                        if( (eig.D(0,0)<0) && (eig.D(1,1)<=0) && (eig.D(2,2)<=0))
                        {
                            eig.D(0,0)=eig.D(1,1)=eig.D(2,2)=0;
                            outlier_image->SetPixel(ind,1);
                        }
                        if( (eig.D(0,0)<0) && (eig.D(1,1)<0) && (eig.D(2,2)>=0))
                        {
                            float val =(eig.D(0,0) + eig.D(1,1)+eig.D(2,2))/3;
                            if(val <0)
                            {
                                eig.D(0,0)=eig.D(1,1)=eig.D(2,2)=0;
                                outlier_image->SetPixel(ind,1);
                            }
                            else
                            {
                                eig.D(0,0)=eig.D(1,1)=eig.D(2,2)=val;
                                outlier_image->SetPixel(ind,1);
                            }
                        }
                    }
                }
                double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
                double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
                double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);
                

                double FA=0;
                if(denom!=0)
                    FA= sqrt( 1.5*nom/denom);
                vnl_vector<double> e1= eig.get_eigenvector(2);
                                 
                FAimage->SetPixel(ind,FA);
            }   
        }   
    }


    if(filter)
    {
        typedef itk::ImageDuplicator<ImageType3D> DupType;
        DupType::Pointer dup = DupType::New();
        dup->SetInputImage(FAimage);
        dup->Update();
        ImageType3D::Pointer FAimage2= dup->GetOutput();

        for(int k=0;k<imsize[2];k++)
        {
            ind[2]=k;
            for(int j=0;j<imsize[1];j++)
            {
                ind[1]=j;
                for(int i=0;i<imsize[0];i++)
                {
                    ind[0]=i;

                    if(outlier_image->GetPixel(ind)!=0)
                    {
                        std::vector<float> signs;


                        ImageType3D::SizeType size;
                        ImageType3D::IndexType start;

                        start[2]= std::max(0,(int)ind[2]-1);
                        size[2]=   std::min((int)imsize[2]-1,(int)ind[2]+1)-start[2]+1;

                        start[1]= std::max(0,(int)ind[1]-1);
                        size[1]=   std::min((int)imsize[1]-1,(int)ind[1]+1)-start[1]+1;

                        start[0]= std::max(0,(int)ind[0]-1);
                        size[0]=   std::min((int)imsize[0]-1,(int)ind[0]+1)-start[0]+1;

                        ImageType3D::RegionType reg(start,size);

                        itk::ImageRegionIteratorWithIndex<ImageType3D> it(FAimage2,reg);
                        it.GoToBegin();
                        while(!it.IsAtEnd())
                        {
                            signs.push_back(it.Get());
                            ++it;
                        }

                        float val = median(signs);
                        FAimage->SetPixel(ind,val);
                    }
                }
            }
        }
    }
    
       
    
       
        
    std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_FA.nii");
       
    
    
    typedef itk::ImageFileWriter<ImageType3D> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(FAimage);
    writer->Update();

                    
    
    

    
    return EXIT_SUCCESS;
}
