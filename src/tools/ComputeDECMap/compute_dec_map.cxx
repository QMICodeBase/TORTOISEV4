

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <math.h>
using namespace std;
#include "compute_dec_map_parser.h"
#include "itkDiffusionTensor3D.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageDuplicator.h"
#include "../ComputeLIMap/compute_li_map.h"
#include <math.h>



typedef itk::DiffusionTensor3D<RealType> TensorPixelType;
typedef itk::Vector<RealType,6> VectorPixelType;
    
typedef itk::Image<itk::DiffusionTensor3D<RealType>,3>         TensorImageType;
typedef itk::Image<VectorPixelType,3>         VectorImageType;
    


double  ShiftBlue(double &ered, double &egreen, double &eblue,double color_par_0 ,double color_par_2)
{
    double bnorm = eblue*0;
    double bfact = eblue*0;
    double totint = ered+egreen+eblue+0.0000001;
    bnorm = eblue/totint;

    double xblue =1./3.;
    double modpar_B = color_par_2*color_par_0;
    bfact = modpar_B*(bnorm-xblue)/(1.-xblue);
    if(bfact < 0)
        bfact = 0;
    double bfact2 =1.-bfact;
    double rfact = 0.25*modpar_B*(bnorm-xblue)/(1.-xblue);
    if(rfact < 0)
        rfact = 0;
    double rfact2 = 1.-rfact;
    double rvl = bfact*eblue+bfact2*ered;
    double gvl = bfact*eblue+bfact2*egreen;
    double bvl = eblue;
    gvl = rfact*rvl+rfact2*gvl;
    bvl = rfact*rvl+rfact2*bvl;

    ered = rvl;
    egreen = gvl;
    eblue = bvl;


    return ered;
}

double  AdjustGreenEqualInt(double &ered, double &egreen, double &eblue,double p1,double p2,double percbeta)
{
    double maxval = std::max(std::max(ered,egreen),eblue);
    maxval = std::max(maxval,0.0000001);
    ered = ered/maxval;
    egreen = egreen/maxval;
    eblue = eblue/maxval;
    double thrd = 1./3.;
    double c1 = thrd-p1/25.;
    double c2 = thrd+p1/4.;
    double c3 = 1.-c1-c2;
    double leql = 0.7;

    double totval = (c1*ered+c2*egreen+(1.-c2-percbeta)*eblue)/pow(leql,(1./percbeta));
    if(totval < 1)
        totval=1.;

    ered = ered/(p2*totval+(1-p2));
    egreen = egreen/(p2*totval+(1-p2));
    eblue = eblue/(p2*totval+(1-p2));

    return ered;
}

double  ScaleIntensity(double &ered, double &egreen, double &eblue, double scalf,double color_scalexp,double percbeta)
{
    ered = ered*pow(scalf,color_scalexp/percbeta);
    egreen = egreen*pow(scalf,color_scalexp/percbeta);
    eblue = eblue*pow(scalf,color_scalexp/percbeta);


    return ered;
}

double  GammaCorrect(double &ered, double &egreen, double &eblue,double gammac_fact )
{
    double exr = 1./gammac_fact;
    double exg = 1./gammac_fact;
    double exb = 1./gammac_fact;

    ered = pow(ered,exr);
    egreen = pow(egreen,exg);
    eblue = pow(eblue,exb);

    return ered;
}

inline bool exists_test3 (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


int main( int argc , char * argv[] )
{


    ComputeDecMap_PARSER *parser = new ComputeDecMap_PARSER(argc,argv);
    
    
    std::string currdir;
    std::string nm(parser->getInputImageName());

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




    ImageType3D::Pointer A0_image=nullptr;
    {
        std::string filename(parser->getInputImageName());
        std::string::size_type idx=filename.rfind("DT.");
        std::string basename= filename.substr(mypos+1,idx-mypos-1);
        std::string A0name=currdir + basename + std::string("AM.nii");

        if(exists_test3(A0name))
        {
            typedef itk::ImageFileReader<ImageType3D> ImageType3DReaderType;
            ImageType3DReaderType::Pointer imager2= ImageType3DReaderType::New();
            imager2->SetFileName(A0name);
            imager2->Update();
            A0_image= imager2->GetOutput();
        }
        else
        {
            A0_image= ImageType3D::New();
            A0_image->SetRegions(reg);
            A0_image->Allocate();
            A0_image->SetOrigin(norig);
            A0_image->SetSpacing(nspc);
            A0_image->SetDirection(ndir);


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


                        double TR = curr_tens(0,0)+curr_tens(1,1)+curr_tens(2,2);
                        A0_image->SetPixel(ind,TR);
                    }
                }
            }
        }
    }


    ImageType3D::Pointer li_map=nullptr;
    if(!parser->getUseFA())
        li_map=compute_li_map(image4D,A0_image);


    
    typedef itk::RGBPixel< unsigned char > RGBPixelType;
    typedef itk::Image< RGBPixelType, 3 > RGBImageType;
    RGBImageType::Pointer DECimage = RGBImageType::New();
    DECimage->SetRegions(reg);
    DECimage->Allocate();
    DECimage->SetOrigin(norig);
    DECimage->SetSpacing(nspc);
    DECimage->SetDirection(ndir);
    RGBPixelType zero;
    zero.Fill(0);
    DECimage->FillBuffer(zero);


    double color_lattmin= parser->getLatticeIndexMin();
    double color_lattmax= parser->getLatticeIndexMax();

    
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
                curr_tens(1,0)=curr_tens(0,1);
                index[3]=4;
                curr_tens(0,2)=image4D->GetPixel(index);
                curr_tens(2,0)=curr_tens(0,2);
                index[3]=5;
                curr_tens(1,2)=image4D->GetPixel(index);
                curr_tens(2,1)=curr_tens(1,2);

                if(curr_tens(0,0)+curr_tens(1,1)+curr_tens(2,2)< 0.1)
                    continue;




                if( !isfinite(curr_tens(0,0)) ||
                    !isfinite(curr_tens(0,1)) ||
                    !isfinite(curr_tens(0,2)) ||
                    !isfinite(curr_tens(1,1)) ||
                    !isfinite(curr_tens(1,2)) ||
                    !isfinite(curr_tens(2,2)))
                    continue;
                

                
                vnl_symmetric_eigensystem<double>  eig(curr_tens);

                double LI=0;
                if(li_map)
                    LI=li_map->GetPixel(ind);
                else
                {
                    double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
                    double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
                    double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);

                    if(denom!=0)
                        LI= sqrt( 1.5*nom/denom);
                }


                double scal= (LI-color_lattmin)/(color_lattmax-color_lattmin);
                if(scal >1)
                    scal=1;
                if(scal<0.)
                    scal=0;


                vnl_vector<double> e1= eig.get_eigenvector(2);
                
                RGBPixelType pix;

                double ered= (fabs(e1[0]));
                double egreen= (fabs(e1[1]));
                double eblue= (fabs(e1[2]));


                ered = ShiftBlue(ered, egreen, eblue,parser->getColorParameter0(),parser->getColorParameter2());
                ered = AdjustGreenEqualInt(ered, egreen, eblue,parser->getColorParameter1(),parser->getColorParameter2(),parser->getPercentBeta());
                ered = ScaleIntensity(ered,egreen,eblue,scal,parser->getScaleXp(),parser->getPercentBeta());
                ered = GammaCorrect(ered, egreen, eblue,parser->getGammaFactor());

                pix[0]= (char)(fabs(ered)*255);
                pix[1]= (char)(fabs(egreen)*255);
                pix[2]= (char)(fabs(eblue)*255);

                DECimage->SetPixel(ind,pix);

            }   
        }   
    }
    
       
    
       
        
    std::string filename(parser->getInputImageName());
    std::string::size_type idx=filename.rfind('.');
    std::string basename= filename.substr(mypos+1,idx-mypos-1);
    std::string output_name=currdir + basename + std::string("_DEC.nii");
       
    
    
    typedef itk::ImageFileWriter<RGBImageType> WriterType;
    WriterType::Pointer writer= WriterType::New();
    writer->SetFileName(output_name);
    writer->SetInput(DECimage);
    writer->Update();


    
    return EXIT_SUCCESS;
}
