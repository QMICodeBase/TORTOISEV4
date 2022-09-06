#include "defines.h"
#include "../tools/EstimateMAPMRI/MAPMRIModel.h"
#include "../utilities/read_bmatrix_file.h"

typedef itk::Vector<float,6> DTType;



int main(int argc, char*argv[])
{

    if(argc==1)
    {
        std::cout<<"Usage:  SynthesizeDWIsFromMAPMRI path_to_mapmri_coeffs path_to_dt path_to_a0 small_delta big_delta  path_to_desired_bmatrix"<<std::endl;
        return EXIT_FAILURE;
    }



    std::string mapname(argv[1]);
    std::string dtname(argv[2]);
    std::string A0name(argv[3]);
    float small_delta = atof(argv[4]);
    float big_delta = atof(argv[5]);
    std::string bmatname(argv[6]);
    

    ImageType3D::Pointer A0_img= readImageD<ImageType3D>(A0name);
    ImageType4D::Pointer dt_img4d= readImageD<ImageType4D>(dtname);
    
    DTImageType::Pointer dt_img= DTImageType::New();
    dt_img->SetRegions(A0_img->GetLargestPossibleRegion());
    dt_img->Allocate();
    dt_img->SetSpacing(A0_img->GetSpacing());
    dt_img->SetDirection(A0_img->GetDirection());
    dt_img->SetOrigin(A0_img->GetOrigin());

    itk::ImageRegionIteratorWithIndex<DTImageType> it(dt_img,dt_img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        DTImageType::IndexType ind3 = it.GetIndex();
        ImageType4D::IndexType ind4;
        ind4[0]=ind3[0];
        ind4[1]=ind3[1];
        ind4[2]=ind3[2];

        ind4[3]=0;
        double Dxx= dt_img4d->GetPixel(ind4)/1000000.;
        ind4[3]=1;
        double Dyy= dt_img4d->GetPixel(ind4)/1000000.;
        ind4[3]=2;
        double Dzz= dt_img4d->GetPixel(ind4)/1000000.;
        ind4[3]=3;
        double Dxy= dt_img4d->GetPixel(ind4)/1000000.;
        ind4[3]=4;
        double Dxz= dt_img4d->GetPixel(ind4)/1000000.;
        ind4[3]=5;
        double Dyz= dt_img4d->GetPixel(ind4)/1000000.;

        DTType  dt_vec;
        dt_vec[0]=Dxx;
        dt_vec[1]=Dxy;
        dt_vec[2]=Dxz;
        dt_vec[3]=Dyy;
        dt_vec[4]=Dyz;
        dt_vec[5]=Dzz;
        it.Set(dt_vec);

        ++it;
    }

    typedef itk::ImageFileReader<MAPImageType> VecReaderType;
    VecReaderType::Pointer reader = VecReaderType::New();
    reader->SetFileName(mapname);
    reader->Update();
    MAPImageType::Pointer mapmri_image= reader->GetOutput();

    MAPImageType::IndexType ind_temp; ind_temp.Fill(0);
    MAPImageType::PixelType vec= mapmri_image->GetPixel(ind_temp);
    int ncoeffs= vec.Size();
    int MAP_ORDER;
    switch(ncoeffs)
    {
           case 7:
               MAP_ORDER=2;
               break;
           case 22:
               MAP_ORDER=4;
               break;
           case 50:
               MAP_ORDER=6;
               break;
           case 95:
               MAP_ORDER=8;
               break;
           case 161:
               MAP_ORDER=10;
               break;
           default:
               std::cout<<"MAPMRI number of coefficients do not match any order. Exiting..."<<std::endl;
               return EXIT_FAILURE;
    }



    MAPMRIModel mapmri_estimator;
    mapmri_estimator.SetMAPMRIDegree(MAP_ORDER);
    mapmri_estimator.SetDTImg(dt_img);
    mapmri_estimator.SetA0Image(A0_img);
    mapmri_estimator.SetSmallDelta(small_delta);
    mapmri_estimator.SetBigDelta(big_delta);
    mapmri_estimator.SetOutput(mapmri_image);
    mapmri_estimator.ComputeEigenImages();

    vnl_matrix<double> Bmatrix = read_bmatrix_file(bmatname);


    ImageType4D::Pointer new_dwis=ImageType4D::New();
    ImageType4D::IndexType start; start.Fill(0);
    ImageType4D::SizeType sz;
    sz[0]=A0_img->GetLargestPossibleRegion().GetSize()[0];
    sz[1]=A0_img->GetLargestPossibleRegion().GetSize()[1];
    sz[2]=A0_img->GetLargestPossibleRegion().GetSize()[2];
    sz[3]= Bmatrix.rows();
    ImageType4D::RegionType reg(start,sz);
    new_dwis->SetRegions(reg);
    new_dwis->Allocate();
    new_dwis->SetSpacing(dt_img4d->GetSpacing());    
    new_dwis->SetOrigin(dt_img4d->GetOrigin());
    new_dwis->SetDirection(dt_img4d->GetDirection());    
    new_dwis->FillBuffer(0.);

#pragma omp parallel for
    for(int vol_id=0;vol_id<Bmatrix.rows();vol_id++)
    {
        ImageType3D::Pointer dwi =mapmri_estimator.SynthesizeDWI( Bmatrix.get_row(vol_id) );

        itk::ImageRegionIteratorWithIndex<ImageType3D> it2(dwi,dwi->GetLargestPossibleRegion());
        it2.GoToBegin();
        while(!it2.IsAtEnd())
        {
            ImageType3D::IndexType ind3= it2.GetIndex();
            ImageType4D::IndexType ind4;
            ind4[0]=ind3[0];
            ind4[1]=ind3[1];
            ind4[2]=ind3[2];
            ind4[3]=vol_id;

            new_dwis->SetPixel(ind4, it2.Get());

            ++it2;
        }

    }

    mapname = mapname.substr(0,mapname.find(".nii")) + "_synth.nii";
    writeImageD<ImageType4D>(new_dwis,mapname);


}

