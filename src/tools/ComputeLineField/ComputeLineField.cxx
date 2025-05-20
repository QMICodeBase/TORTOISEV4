

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <math.h>
using namespace std;

#include "itkDiffusionTensor3D.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageDuplicator.h"
#include <math.h>
#include "defines.h"



//typedef itk::Vector<RealType,6> VectorPixelType;
    
//typedef itk::Image<VectorPixelType,3>         VectorImageType;
    

int main( int argc , char * argv[] )
{
    if(argc<2)
    {
        std::cout<<"Usage: ComputeLineField tensor_img  slice_z (optional) slice_y (optional) slice_x (optional)"<<std::endl;
        return EXIT_FAILURE;
    }

    const unsigned int mag_factor=21;



    std::string nm=argv[1];
    typedef itk::ImageFileReader<ImageType4D> ImageType4DReaderType;
    ImageType4DReaderType::Pointer imager= ImageType4DReaderType::New();
    imager->SetFileName(nm);
    imager->Update();
    ImageType4D::Pointer image4D= imager->GetOutput();



    ImageType4D::SizeType imsize= image4D->GetLargestPossibleRegion().GetSize();
    int sl_z= imsize[2]/2;
    if(argc>2)
    {
        sl_z= atoi(argv[2]);
    }
    int sl_y= imsize[1]/2;
    if(argc>3)
    {
        sl_y= atoi(argv[3]);
    }
    int sl_x= imsize[0]/2;
    if(argc>4)
    {
        sl_x= atoi(argv[4]);
    }


    int sls[3]={sl_x,sl_y,sl_z};

    for(int d=0;d<3;d++)
    {
        ImageType3D::SizeType nsize;
        nsize[0]=imsize[0]*mag_factor;
        nsize[1]=imsize[1]*mag_factor;
        nsize[2]=imsize[2]*mag_factor;
        nsize[d]=1;

        ImageType3D::IndexType start; start.Fill(0);
        ImageType3D::RegionType reg(start,nsize);

        ImageType3D::PointType norig;
        norig[0]=image4D->GetOrigin()[0];
        norig[1]=image4D->GetOrigin()[1];
        norig[2]=image4D->GetOrigin()[2];
        norig[d]=0;

        ImageType3D::SpacingType nspc;
        nspc[0]=image4D->GetSpacing()[0];
        nspc[1]=image4D->GetSpacing()[1];
        nspc[2]=image4D->GetSpacing()[2];

        ImageType3D::DirectionType ndir;
        ImageType4D::DirectionType dir = image4D->GetDirection();
        ndir(0,0)=dir(0,0);ndir(0,1)=dir(0,1);ndir(0,2)=dir(0,2);
        ndir(1,0)=dir(1,0);ndir(1,1)=dir(1,1);ndir(1,2)=dir(1,2);
        ndir(2,0)=dir(2,0);ndir(2,1)=dir(2,1);ndir(2,2)=dir(2,2);

        typedef itk::RGBPixel< unsigned char > RGBPixelType3D;
        typedef itk::Image< RGBPixelType3D, 3 > RGBImageType3D;
        RGBImageType3D::Pointer field_img = RGBImageType3D::New();
        field_img->SetRegions(reg);
        field_img->Allocate();
        field_img->SetOrigin(norig);
        field_img->SetSpacing(nspc);
        field_img->SetDirection(ndir);
        RGBPixelType3D zero;
        zero.Fill(0);
        field_img->FillBuffer(zero);

        int starts[3]={0};
        starts[d]=sls[d];
        int ends[3];
        ends[0]=imsize[0];
        ends[1]=imsize[1];
        ends[2]=imsize[2];
        ends[d]=sls[d]+1;



        for(int k=starts[2];k<ends[2];k++)
        {
            ImageType3D::IndexType ind3;
            ImageType4D::IndexType index;
            ind3[2]=k;
            index[2]=k;
            for(int j=starts[1];j<ends[1];j++)
            {
                ind3[1]=j;
                index[1]=j;
                for(int i=starts[0];i<ends[0];i++)
                {
                    ind3[0]=i;
                    index[0]=i;


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

                    vnl_symmetric_eigensystem<double>  eig(curr_tens);

                    if(eig.D(0,0) < 0)
                        eig.D(0,0)=0;
                    if(eig.D(1,1) < 0)
                        eig.D(1,1)=0;
                    if(eig.D(2,2) < 0)
                        eig.D(2,2)=0;


                    double FA=0;
                    double mn = (eig.D(0,0)+ eig.D(1,1)+ eig.D(2,2))/3.;
                    double nom = (eig.D(0,0)-mn)*(eig.D(0,0)-mn)+ (eig.D(1,1)-mn)*(eig.D(1,1)-mn)+(eig.D(2,2)-mn)*(eig.D(2,2)-mn);
                    double denom= eig.D(0,0)*eig.D(0,0)+eig.D(1,1)*eig.D(1,1)+eig.D(2,2)*eig.D(2,2);

                    if(denom!=0)
                        FA= sqrt( 1.5*nom/denom);

                    int starts2[3];
                    starts2[0]= i*mag_factor;
                    starts2[1]= j*mag_factor;
                    starts2[2]= k*mag_factor;
                    starts2[d]=0;

                    int ends2[3];
                    ends2[0]= (i+1)*mag_factor;
                    ends2[1]= (j+1)*mag_factor;
                    ends2[2]= (k+1)*mag_factor;
                    ends2[d]=1;

                    for(int i2=starts2[0]; i2<ends2[0];i2++)
                    {
                        RGBImageType3D::IndexType rind3;
                        rind3[0]=i2;

                        for(int j2=starts2[1]; j2<ends2[1];j2++)
                        {
                            rind3[1]=j2;
                            for(int k2=starts2[2]; k2<ends2[2];k2++)
                            {
                                rind3[2]=k2;
                                RGBImageType3D::PixelType pix;
                                unsigned char s[3];

                                s[0]=(unsigned char)floor(255*fabs(FA));
                                s[1]=(unsigned char)floor(255*fabs(FA));
                                s[2]=(unsigned char)floor(255*fabs(FA));

                                field_img->SetPixel(rind3,s);
                            }
                        }
                    }

                    if(FA<0.3)
                        continue;

                    vnl_vector<double> evec= eig.get_eigenvector(2);
                    unsigned char s[3];
                    s[0]=(unsigned char)floor(255*fabs(evec[0]));
                    s[1]=(unsigned char)floor(255*fabs(evec[1]));
                    s[2]=(unsigned char)floor(255*fabs(evec[2]));

                    RGBImageType3D::IndexType rind3;
                    rind3[0]=(starts2[0]+ends2[0])/2;
                    rind3[1]=(starts2[1]+ends2[1])/2;
                    rind3[2]=(starts2[2]+ends2[2])/2;
                    field_img->SetPixel(rind3,s);


                    for(int gr1=-2;gr1<2;gr1++)
                    {
                        for(int gr2=-2;gr2<2;gr2++)
                        {
                            for(int gr3=-2;gr3<2;gr3++)
                            {
                                double aa= rind3[0]+gr1;
                                double bb= rind3[1]+gr2;
                                double cc= rind3[2]+gr3;
                                for(int step=0;step<mag_factor*FA/2;step++)
                                {
                                    aa+= evec[0];
                                    bb+= evec[1];
                                    cc+= evec[2];

                                    RGBImageType3D::IndexType nind3;
                                    nind3[0]= (int)round(aa);
                                    if(nind3[0]<0)
                                        nind3[0]=0;
                                    if(nind3[0]>field_img->GetLargestPossibleRegion().GetSize()[0]-1)                            nind3[0]=field_img->GetLargestPossibleRegion().GetSize()[0]-1;

                                    nind3[1]= (int)round(bb);
                                    if(nind3[1]<0)
                                        nind3[1]=0;
                                    if(nind3[1]>field_img->GetLargestPossibleRegion().GetSize()[1]-1)
                                        nind3[1]=field_img->GetLargestPossibleRegion().GetSize()[1]-1;
                                    nind3[2]= (int)round(cc);
                                    if(nind3[2]<0)
                                        nind3[2]=0;
                                    if(nind3[2]>field_img->GetLargestPossibleRegion().GetSize()[2]-1)
                                        nind3[2]=field_img->GetLargestPossibleRegion().GetSize()[2]-1;

                                    nind3[d]=0;
                                    field_img->SetPixel(nind3,s);
                                }
                                aa= rind3[0]+gr1;
                                bb= rind3[1]+gr2;
                                cc= rind3[2]+gr3;
                                for(int step=0;step<mag_factor*FA/2;step++)
                                {
                                    aa-= evec[0];
                                    bb-= evec[1];
                                    cc-= evec[2];

                                    RGBImageType3D::IndexType nind3;
                                    nind3[0]= (int)round(aa);
                                    if(nind3[0]<0)
                                        nind3[0]=0;
                                    if(nind3[0]>field_img->GetLargestPossibleRegion().GetSize()[0]-1)                            nind3[0]=field_img->GetLargestPossibleRegion().GetSize()[0]-1;

                                    nind3[1]= (int)round(bb);
                                    if(nind3[1]<0)
                                        nind3[1]=0;
                                    if(nind3[1]>field_img->GetLargestPossibleRegion().GetSize()[1]-1)
                                        nind3[1]=field_img->GetLargestPossibleRegion().GetSize()[1]-1;
                                    nind3[2]= (int)round(cc);
                                    if(nind3[2]<0)
                                        nind3[2]=0;
                                    if(nind3[2]>field_img->GetLargestPossibleRegion().GetSize()[2]-1)
                                        nind3[2]=field_img->GetLargestPossibleRegion().GetSize()[2]-1;
                                    nind3[d]=0;
                                    field_img->SetPixel(nind3,s);
                                }

                            }

                        }

                    }


                } //for i
            } //for j
        } //for k

        std::string mdir;
        if(d==0)
            mdir="x";
        if(d==1)
            mdir="y";
        if(d==2)
            mdir="z";


        typedef itk::Image< RGBPixelType3D, 2 > RGBImageType2D;


        RGBImageType3D::SizeType rsz=field_img->GetLargestPossibleRegion().GetSize();
        rsz[d]=0;
        RGBImageType3D::IndexType rstart; rstart.Fill(0);
        RGBImageType3D::RegionType rr(rstart,rsz);

        using FilterType = itk::ExtractImageFilter<RGBImageType3D, RGBImageType2D>;
        auto filter = FilterType::New();
        filter->InPlaceOn();
        filter->SetDirectionCollapseToSubmatrix();
        filter->SetExtractionRegion(rr);
        filter->SetInput(field_img);
        filter->Update();
        auto output= filter->GetOutput();

        std::string oname = nm.substr(0, nm.rfind(".nii")) + "_linefield_" + mdir + ".png";

        using PNGWriter = itk::ImageFileWriter<RGBImageType2D>;
        PNGWriter::Pointer wr= PNGWriter::New();
        wr->SetInput(output);
        wr->SetFileName(oname);
        wr->Update();
    } //for d

    
    return EXIT_SUCCESS;
}
