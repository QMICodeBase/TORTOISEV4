#ifndef _READ3DVOLUMEFROM4D_CXX
#define _READ3DVOLUMEFROM4D_CXX


#include "read_3Dvolume_from_4D.h"
#include <string>
#include "itkNiftiImageIO.h"
#include <stdio.h>
#include "itkImportImageFilter.h"


void onifti_swap_4bytes( size_t n , void *ar )    /* 4 bytes at a time */
{
   size_t ii ;
   unsigned char * cp0 = (unsigned char *)ar, * cp1, * cp2 ;
   unsigned char tval ;

   for( ii=0 ; ii < n ; ii++ ){
       cp1 = cp0; cp2 = cp0+3;
       tval = *cp1;  *cp1 = *cp2;  *cp2 = tval;
       cp1++;  cp2--;
       tval = *cp1;  *cp1 = *cp2;  *cp2 = tval;
       cp0 += 4;
   }
   return ;
}

ImageType3D::Pointer read_3D_volume_from_4D(std::string fname, int vol_id)
{

//    ReaderType::Pointer reader= ReaderType::New();

    itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
    myio->SetFileName(fname);
    myio->ReadImageInformation();

    ImageType3D::SizeType sz;
    sz[0]= myio->GetDimensions(0);
    sz[1]= myio->GetDimensions(1);
    sz[2]= myio->GetDimensions(2);

    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,sz);

    ImageType3D::PointType orig;
    orig[0]= myio->GetOrigin(0);
    orig[1]= myio->GetOrigin(1);
    orig[2]= myio->GetOrigin(2);

    ImageType3D::SpacingType spc;
    spc[0]=myio->GetSpacing(0);
    spc[1]=myio->GetSpacing(1);
    spc[2]=myio->GetSpacing(2);

    ImageType4D::DirectionType dir4;

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

    ImageType3D::DirectionType dir;
    dir(0,0)=dir4(0,0);dir(0,1)=dir4(0,1);dir(0,2)=dir4(0,2);
    dir(1,0)=dir4(1,0);dir(1,1)=dir4(1,1);dir(1,2)=dir4(1,2);
    dir(2,0)=dir4(2,0);dir(2,1)=dir4(2,1);dir(2,2)=dir4(2,2);

  //  dir.SetIdentity();

    float off;
    int sz_hdr;
    FILE *fp = fopen(fname.c_str(),"rb");

    if(fread(&sz_hdr,sizeof(int),1,fp)==0)
    {
        std::cout<<"Could not read file " << fname << std::endl;
    }

    short datatype;
    fseek(fp,70,SEEK_SET);
    if(fread(&datatype,sizeof(short),1,fp)==0)
    {
        std::cout<<"Could not read file " << fname << std::endl;
    }

    fseek(fp,108,SEEK_SET);
    if(fread(&off,sizeof(float),1,fp)==0)
    {
        std::cout<<"Could not read file " << fname << std::endl;
    }


    bool change_endian=0;
    if(sz_hdr!=348)
        change_endian=1;

    if(change_endian)
        onifti_swap_4bytes(1,&off);


     long long int Npixelspervolume= sz[0]*sz[1]*sz[2];
     long long int bytepix=sizeof(float);


     long long int vol_id2= vol_id;
    long long int off2= off;


     long long int seek = off2 + Npixelspervolume* bytepix *vol_id2;

	 int ppos=0;

#if (BOOST_OS_CYGWIN || BOOST_OS_WINDOWS) 
#include <windows.h>
		ppos= _fseeki64(fp, seek, SEEK_SET);
#else
		 ppos= fseeko(fp,seek,SEEK_SET);
#endif    

     
     if(ppos!=0)
     {
         std::cout<< "error seeking file"<<std::endl;
     }

     float *data= new float[Npixelspervolume];
     if(fread(data,sizeof(float),Npixelspervolume,fp)!=Npixelspervolume)
     {
         std::cout<<"Could not read file " << fname << std::endl;
     }
     fclose(fp);
     if(change_endian)
         onifti_swap_4bytes(Npixelspervolume,data);



     typedef itk::ImportImageFilter< float, 3 >   ImportFilterType;
     ImportFilterType::Pointer importFilter = ImportFilterType::New();
     importFilter->SetRegion( reg );
     importFilter->SetOrigin(orig);
     importFilter->SetSpacing(spc);
     importFilter->SetDirection(dir);;
     importFilter->SetImportPointer(data,Npixelspervolume,false);      
     importFilter->Update();

     return importFilter->GetOutput();
}


ImageType3DBool::Pointer read_3D_volume_from_4DBool(std::string fname, int vol_id)
{
    itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
    myio->SetFileName(fname);
    myio->ReadImageInformation();

    ImageType3DBool::SizeType sz;
    sz[0]= myio->GetDimensions(0);
    sz[1]= myio->GetDimensions(1);
    sz[2]= myio->GetDimensions(2);

    ImageType3DBool::IndexType start; start.Fill(0);
    ImageType3DBool::RegionType reg(start,sz);

    ImageType3DBool::PointType orig;
    orig[0]= myio->GetOrigin(0);
    orig[1]= myio->GetOrigin(1);
    orig[2]= myio->GetOrigin(2);

    ImageType3DBool::SpacingType spc;
    spc[0]=myio->GetSpacing(0);
    spc[1]=myio->GetSpacing(1);
    spc[2]=myio->GetSpacing(2);

    std::vector<double> temp = myio->GetDirection(0);
    if(temp.size()<4)
    {
        std::cout<<"File "<<fname << " not a 4D image file.. Exiting"<<std::endl;
        exit(0);
    }

    ImageType4DBool::DirectionType dir4;

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

    ImageType3DBool::DirectionType dir;
    dir(0,0)=dir4(0,0);dir(0,1)=dir4(0,1);dir(0,2)=dir4(0,2);
    dir(1,0)=dir4(1,0);dir(1,1)=dir4(1,1);dir(1,2)=dir4(1,2);
    dir(2,0)=dir4(2,0);dir(2,1)=dir4(2,1);dir(2,2)=dir4(2,2);


    float off;
    int sz_hdr;
    FILE *fp = fopen(fname.c_str(),"rb");

    if(fread(&sz_hdr,sizeof(int),1,fp)==0)
    {
        std::cout<<"Could not read file " << fname << std::endl;
    }

    short datatype;
    fseek(fp,70,SEEK_SET);
    if(fread(&datatype,sizeof(short),1,fp)==0)
    {
        std::cout<<"Could not read file " << fname << std::endl;
    }

    fseek(fp,108,SEEK_SET);
    if(fread(&off,sizeof(float),1,fp)==0)
    {
        std::cout<<"Could not read file " << fname << std::endl;
    }


    bool change_endian=0;
    if(sz_hdr!=348)
        change_endian=1;

    if(change_endian)
        onifti_swap_4bytes(1,&off);


     long long int Npixelspervolume= sz[0]*sz[1]*sz[2];
     long long int bytepix=sizeof(ImageType3DBool::PixelType);


     long long int vol_id2= vol_id;
    long long int off2= off;


     long long int seek = off2 + Npixelspervolume* bytepix *vol_id2;

#ifdef WIN32
         int ppos = _fseeki64(fp, seek, SEEK_SET);
#else
         int ppos = fseeko(fp, seek, SEEK_SET);
#endif
     if(ppos!=0)
     {
         std::cout<< "error seeking file"<<std::endl;
     }

     ImageType3DBool::PixelType *data= new ImageType3DBool::PixelType[Npixelspervolume];
     if(fread(data,sizeof(ImageType3DBool::PixelType),Npixelspervolume,fp)!=Npixelspervolume)
     {
         std::cout<<"Could not read file " << fname << std::endl;
     }
     fclose(fp);



     ImageType3DBool::Pointer out_img= ImageType3DBool::New();
     out_img->SetRegions(reg);
     out_img->SetSpacing(spc);
     out_img->SetDirection(dir);
     out_img->SetOrigin(orig);
     out_img->GetPixelContainer()->SetImportPointer(data,Npixelspervolume,true);


     return out_img;
}




#endif
