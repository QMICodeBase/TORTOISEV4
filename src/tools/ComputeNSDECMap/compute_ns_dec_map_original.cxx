

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
    


void HSV2RGB(double H,double S, double V, double &R, double &G,double &B)
{

    double C= V*S;

    double Hm  =  (H/60.);
    double X = C*(1-fabs((fmod(Hm,2) -1)));
    double m= V-C;

    double Rp, Gp,Bp;

    if(H>=0 && H<60)
    {
        Rp=C;
        Gp=X;
        Bp=0;
    }
    if(H>=60 && H<120)
    {
        Rp=X;
        Gp=C;
        Bp=0;
    }
    if(H>=120 && H<180)
    {
        Rp=0;
        Gp=C;
        Bp=X;
    }
    if(H>=180 && H<240)
    {
        Rp=0;
        Gp=X;
        Bp=C;
    }
    if(H>=240 && H<300)
    {
        Rp=X;
        Gp=0;
        Bp=C;
    }
    if(H>=300 && H<360)
    {
        Rp=C;
        Gp=0;
        Bp=X;
    }

    R =    ((Rp+m));
    G =    ((Gp+m));
    B =    ((Bp+m));




}

int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<<"Usage:   ComputeDECMapNS full_path_to_tensor_image"<<std::endl;
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
    
    
    
    typedef itk::RGBPixel< unsigned char > RGBPixelType;
    typedef itk::Image< RGBPixelType, 3 > RGBImageType;
    RGBImageType::Pointer DECimage = RGBImageType::New();
    DECimage->SetRegions(reg);
    DECimage->Allocate();
    DECimage->SetOrigin(norig);
    DECimage->SetSpacing(nspc);
    DECimage->SetDirection(ndir);
    
    double color_par_0=0.35  ;
    double color_par_1=0.8    ;
    double color_par_2=0.7    ;
    double color_par_3=0.5 ;
    double p3=color_par_3;
    
    
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
                
                
                vnl_symmetric_eigensystem<double>  eig(curr_tens);

                double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
                double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
                double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);
                
                
                double FA= sqrt( 1.5*nom/denom);
                if(FA>1)
                    FA=1;

                vnl_vector<double> e1= eig.get_eigenvector(2);
                e1.normalize();

                if(e1[2]<0)
                    e1= e1*-1;


                double xylen= sqrt( e1[0]*e1[0] + e1[1]*e1[1]);
                double phiv= xylen;

                if(xylen > 0.000001)
                    phiv= asin(e1[1]/xylen);
                if(e1[0]<0)
                    phiv= 3.141592 - phiv;
                phiv+= 2* 3.141592;

                double phired= -3.141592/2;
                double thetav= asin(xylen);
                double vals= 1;

                double satur=sin(p3*thetav)/sin(p3*3.141592/2.);
                double hue2=(phiv-phired)*180./3.141592;
                double hue=  fmod(hue2+360.,360.);




               double H=hue;
               double S= satur;
               double V=vals;


        /*        double theta= atan2(e1[1],e1[0]);
                if(theta <0)
                    theta+= 2*3.141592;

                double phi = acos(e1[2]);

                double H= 180.*theta/3.141592;
                double S= 2*phi/3.141592;
                double V= FA;
*/

                double R,G,B;

                HSV2RGB(H,S,V,R,G,B);



                
                RGBPixelType pix;
                pix[0]= (char)(R*FA*255);
                pix[1]= (char)(G*FA*255);
                pix[2]= (char)(B*FA*255);
                
                DECimage->SetPixel(ind,pix);
            }   
        }   
    }
    
       
    
       
        
         std::string filename(argv[1]);
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_DECNS.nii");
       
    
    
    typedef itk::ImageFileWriter<RGBImageType> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(DECimage);
    writer->Update();

                    
    
    

    
    return 1;
}
