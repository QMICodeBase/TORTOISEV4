#ifndef _WRITE3DIMAGETo4DFILE_HXX
#define _WRITE3DIMAGETo4DFILE_HXX



#include <string>
#include "itkNiftiImageIO.h"
#include <stdio.h>
//#include "itkNiftiImageIOHeader.h"
//#include "itkNiftiImageIOHeaderFactory.h"


template<typename PixelType>
void write_3D_image_to_4D_file(typename itk::Image<PixelType,3>::Pointer img, std::string filename, int curr_vol, int total_vols )
{
    using ImageType3D=itk::Image<PixelType,3>;
    using ImageType4D=itk::Image<PixelType,4>;

    FILE *file;
    int file_exists;

    file=fopen(filename.c_str(),"r");
    if (file==NULL)
        file_exists=0;
    else
    {
        file_exists=1;
        fclose(file);
    }


    bool mismatch=0;
    if(file_exists)
    {
        itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
        myio->SetFileName(filename);
        myio->ReadImageInformation();

        if(myio->GetDimensions(0)!=img->GetLargestPossibleRegion().GetSize()[0])
            mismatch=1;
        if(myio->GetDimensions(1)!=img->GetLargestPossibleRegion().GetSize()[1])
            mismatch=1;
        if(myio->GetDimensions(2)!=img->GetLargestPossibleRegion().GetSize()[2])
            mismatch=1;

        if(fabs(myio->GetSpacing(0) - img->GetSpacing()[0]) >0.001)
            mismatch=1;
        if(fabs(myio->GetSpacing(1) - img->GetSpacing()[1]) >0.001)
            mismatch=1;
        if(fabs(myio->GetSpacing(2) - img->GetSpacing()[2]) >0.001)
            mismatch=1;
        if(myio->GetDimensions(3)!=total_vols)
            mismatch=1;


        std::vector<double> temp = myio->GetDirection(0);
        if(temp.size()!=4)
           mismatch=1;

        typename ImageType4D::DirectionType dir4;
        std::vector< std::vector< double > > directionIO;
        for ( unsigned int k = 0; k < 4; k++ )
        {
            directionIO.push_back( myio->GetDirection(k) );
        }
        for ( unsigned int i = 0; i < 4; i++ )
        {
            std::vector<double> axis = directionIO[i];
            for ( unsigned j = 0; j < 4; j++ )
            {
                dir4[j][i] = axis[j];
            }
        }
        typename ImageType3D::DirectionType dirm;
        dirm(0,0)=dir4(0,0);dirm(0,1)=dir4(0,1);dirm(0,2)=dir4(0,2);
        dirm(1,0)=dir4(1,0);dirm(1,1)=dir4(1,1);dirm(1,2)=dir4(1,2);
        dirm(2,0)=dir4(2,0);dirm(2,1)=dir4(2,1);dirm(2,2)=dir4(2,2);

        double diff_sq=0;
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                double diff =(dirm(i,j)-img->GetDirection()(i,j));
                diff_sq+= diff*diff;
            }
        }
        diff_sq= sqrt(diff_sq);
        if(diff_sq>0.1)
            mismatch=1;


        auto datatype = myio->GetComponentType();
        if( (std::is_same<PixelType, float>::value) && (datatype!= itk::IOComponentEnum::FLOAT) )
            mismatch=1;
        if( (std::is_same<PixelType, char>::value) && (datatype!= itk::IOComponentEnum::CHAR) )
            mismatch=1;
    }

    if(!file_exists || mismatch)
    {
        typename ImageType4D::PointType orig;
        orig[0]=img->GetOrigin()[0];
        orig[1]=img->GetOrigin()[1];
        orig[2]=img->GetOrigin()[2];
        orig[3]=0;

        typename ImageType4D::SpacingType spc;
        spc[0]=img->GetSpacing()[0];
        spc[1]=img->GetSpacing()[1];
        spc[2]=img->GetSpacing()[2];
        spc[3]=1;

        typename ImageType4D::IndexType start; start.Fill(0);
        typename ImageType4D::SizeType sz;
        sz[0]=img->GetLargestPossibleRegion().GetSize()[0];
        sz[1]=img->GetLargestPossibleRegion().GetSize()[1];
        sz[2]=img->GetLargestPossibleRegion().GetSize()[2];
        sz[3]=total_vols;
        typename ImageType4D::RegionType reg(start,sz);

        typename ImageType4D::DirectionType dir;
        dir.SetIdentity();
        dir(0,0)= img->GetDirection()(0,0);dir(0,1)= img->GetDirection()(0,1);dir(0,2)= img->GetDirection()(0,2);
        dir(1,0)= img->GetDirection()(1,0);dir(1,1)= img->GetDirection()(1,1);dir(1,2)= img->GetDirection()(1,2);
        dir(2,0)= img->GetDirection()(2,0);dir(2,1)= img->GetDirection()(2,1);dir(2,2)= img->GetDirection()(2,2);

        typename ImageType4D::Pointer new_nifti= ImageType4D::New();
        new_nifti->SetOrigin(orig);
        new_nifti->SetSpacing(spc);
        new_nifti->SetDirection(dir);
        new_nifti->SetRegions(reg);
        new_nifti->Allocate();



       // itk::ObjectFactoryBase::RegisterFactory(itk::NiftiImageIOHeaderFactory::New());
       // itk::NiftiImageIOHeader::Pointer myio2 = itk::NiftiImageIOHeader::New();
        typedef itk::ImageFileWriter<ImageType4D> WriterType;
        typename WriterType::Pointer wr= WriterType::New();
       // wr->SetImageIO(myio2);
        wr->SetFileName(filename);
        wr->SetInput(new_nifti);
        wr->Update();



        long long int HEADER_SIZE=352;
        long long int BYTE_SIZEM= img->GetLargestPossibleRegion().GetSize()[0];
        BYTE_SIZEM *= img->GetLargestPossibleRegion().GetSize()[1];
        BYTE_SIZEM *= img->GetLargestPossibleRegion().GetSize()[2];
        BYTE_SIZEM *= sizeof(typename ImageType3D::PixelType)*total_vols;
        BYTE_SIZEM += HEADER_SIZE;

        FILE *fp= fopen(filename.c_str(),"r+b");
        if(!fp)
        {
            std::cout<<"Could not open file " << filename << " for writing... Exiting..."<<std::endl;
            exit(0);
        }
        #ifdef WIN32
            int ppos = _fseeki64(fp, BYTE_SIZEM - 1, SEEK_SET);
        #else
            int ppos= fseeko(fp, BYTE_SIZEM-1,SEEK_SET);
        #endif

        if(ppos!=0)
        {
            std::cout<<"Could not seek file " << filename << " . Exiting..."<<std::endl;
            exit(0);
        }
        char dum= 0;
        fwrite(&dum,sizeof(char),1,fp);
        fclose(fp);
    }

    long long int HEADER_SIZE=352;
    long long int BYTE_SIZEM= img->GetLargestPossibleRegion().GetSize()[0];
    BYTE_SIZEM = BYTE_SIZEM*(int)(img->GetLargestPossibleRegion().GetSize()[1]);
    BYTE_SIZEM = BYTE_SIZEM*(int)(img->GetLargestPossibleRegion().GetSize()[2]);
    BYTE_SIZEM= BYTE_SIZEM*sizeof(typename ImageType3D::PixelType)*curr_vol+ HEADER_SIZE;

    FILE *fp= fopen(filename.c_str(),"r+b");
    if(!fp)
    {
        std::cout<<"Could not open file " << filename << " for writing... Exiting..."<<std::endl;
        exit(0);
    }

#ifdef WIN32
        int ppos = _fseeki64(fp, BYTE_SIZEM, SEEK_SET);
#else
        int ppos = fseeko(fp, BYTE_SIZEM, SEEK_SET);
#endif

    if(ppos!=0)
    {
        std::cout<<"Could not seek file " << filename << " . Exiting..."<<std::endl;
        exit(0);
    }


    long Npixels= (long)img->GetLargestPossibleRegion().GetSize()[0] * (long)img->GetLargestPossibleRegion().GetSize()[1] * (long)img->GetLargestPossibleRegion().GetSize()[2];

    fwrite(img->GetBufferPointer(),sizeof(typename ImageType3D::PixelType),Npixels,fp);
    fclose(fp);

}



template void write_3D_image_to_4D_file<float>(itk::Image<float,3>::Pointer, std::string, int, int);
template void write_3D_image_to_4D_file<char>(itk::Image<char,3>::Pointer, std::string, int, int);


#endif
