
#include "../EstimateTensor/DTIModel.h"
#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "estimate_mapmri_parser.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "TORTOISE.h"
#include "MAPMRIModel.h"
#include "itkCastImageFilter.h"

int main(int argc, char *argv[])
{

    TORTOISE t;

    EstimateMAPMRI_PARSER *parser= new EstimateMAPMRI_PARSER(argc,argv);


    std::string input_name = parser->getInputImageName();
    std::string bmtxt_name= input_name.substr(0,input_name.rfind(".nii"))+".bmtxt";
    vnl_matrix<double> Bmatrix= read_bmatrix_file(bmtxt_name);
    vnl_vector<double> bvals= Bmatrix.get_column(0)+Bmatrix.get_column(3)+Bmatrix.get_column(5);
    int Nvols= Bmatrix.rows();

    std::vector<ImageType3D::Pointer> final_data;
    final_data.resize(Nvols);
    std::vector<ImageType3D::Pointer> weight_imgs;

    if(parser->getInclusionImg()!="")
        weight_imgs.resize(Nvols);

    for(int v=0;v<Nvols;v++)
    {
        final_data[v]= read_3D_volume_from_4D(input_name,v);
        if(parser->getInclusionImg()!="")
        {
            ImageType3DBool::Pointer inc_img=read_3D_volume_from_4DBool(parser->getInclusionImg(),v);
            using FilterType = itk::CastImageFilter<ImageType3DBool, ImageType3D>;
            auto filter = FilterType::New();
            filter->SetInput(inc_img);
            filter->Update();
            weight_imgs[v]=filter->GetOutput();
        }
    }

    ImageType3D::Pointer mask_image=nullptr;
    if(parser->getMaskImageName()!="")
    {
        if(fs::exists(parser->getMaskImageName()))
        {            
            mask_image=readImageD<ImageType3D>(parser->getMaskImageName());
        }
        else
        {
            std::cout<<"Mask image does NOT exist!!. Fitting tensors to the entire image"<<std::endl;
        }
    }


    std::vector<int> dummy;    
    {
        double bval_cut =parser->getBValCutoff();
        for(int i=0;i<Bmatrix.rows();i++)
        {
            double bval= Bmatrix(i,0)+ Bmatrix(i,3)+Bmatrix(i,5);
            if(bval<=1.05*bval_cut)
                dummy.push_back(i);
        }
    }

    ImageType3D::Pointer A0_img=nullptr;
    DTImageType::Pointer dti_img=nullptr;
    if(parser->getDTIImageName()!="")
    {
        A0_img=readImageD<ImageType3D>(parser->getA0ImageName());
        ImageType4D::Pointer dt_image4d= readImageD<ImageType4D>(parser->getDTIImageName());

        DTImageType::IndexType start; start.Fill(0);
        DTImageType::SizeType sz;
        sz[0]= dt_image4d->GetLargestPossibleRegion().GetSize()[0];
        sz[1]= dt_image4d->GetLargestPossibleRegion().GetSize()[1];
        sz[2]= dt_image4d->GetLargestPossibleRegion().GetSize()[2];
        DTImageType::RegionType reg(start,sz);

        DTImageType::SpacingType spc;
        spc[0]=dt_image4d->GetSpacing()[0];
        spc[1]=dt_image4d->GetSpacing()[1];
        spc[2]=dt_image4d->GetSpacing()[2];

        DTImageType::PointType orig;
        orig[0]=dt_image4d->GetOrigin()[0];
        orig[1]=dt_image4d->GetOrigin()[1];
        orig[2]=dt_image4d->GetOrigin()[2];

        DTImageType::DirectionType dir;
        dir.SetIdentity();
        dir(0,0)= dt_image4d->GetDirection()(0,0);dir(0,1)= dt_image4d->GetDirection()(0,1);dir(0,2)= dt_image4d->GetDirection()(0,2);
        dir(1,0)= dt_image4d->GetDirection()(1,0);dir(1,1)= dt_image4d->GetDirection()(1,1);dir(1,2)= dt_image4d->GetDirection()(1,2);
        dir(2,0)= dt_image4d->GetDirection()(2,0);dir(2,1)= dt_image4d->GetDirection()(2,1);dir(2,2)= dt_image4d->GetDirection()(2,2);

        dti_img= DTImageType::New();
        dti_img->SetRegions(reg);
        dti_img->Allocate();
        dti_img->SetSpacing(spc);
        dti_img->SetOrigin(orig);
        dti_img->SetDirection(dir);

        itk::ImageRegionIteratorWithIndex<DTImageType> it(dti_img,dti_img->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            DTImageType::IndexType index= it.GetIndex();

            ImageType4D::IndexType ind4d;
            ind4d[0]=index[0];
            ind4d[1]=index[1];
            ind4d[2]=index[2];

            DTImageType::PixelType dti_vec;
            ind4d[3]=0;
            dti_vec[0] = dt_image4d->GetPixel(ind4d )/1000000.;
            ind4d[3]=1;
            dti_vec[3] =dt_image4d->GetPixel(ind4d)/1000000.;
            ind4d[3]=2;
            dti_vec[5] =dt_image4d->GetPixel(ind4d)/1000000.;
            ind4d[3]=3;
            dti_vec[1] =dt_image4d->GetPixel(ind4d)/1000000.;
            ind4d[3]=4;
            dti_vec[2] =dt_image4d->GetPixel(ind4d)/1000000.;
            ind4d[3]=5;
            dti_vec[4] =dt_image4d->GetPixel(ind4d)/1000000.;

            it.Set(dti_vec);
            ++it;
        }
    }



    std::vector<ImageType4D::Pointer > dummyv;
    if(!A0_img || !dti_img)
    {
        DTIModel dti_estimator;
        dti_estimator.SetBmatrix(Bmatrix);
        dti_estimator.SetDWIData(final_data);
        dti_estimator.SetWeightImage(weight_imgs);
        dti_estimator.SetVoxelwiseBmatrix(dummyv);
        dti_estimator.SetMaskImage(mask_image);
        dti_estimator.SetVolIndicesForFitting(dummy);
        dti_estimator.SetFittingMode("WLLS");
        dti_estimator.PerformFitting();
        if(!A0_img)
            A0_img=dti_estimator.GetA0Image();
        if(!dti_img)
            dti_img=dti_estimator.GetOutput();
    }


        // MAPMRI FITTING
    const unsigned int FINAL_STAGE_MAPMRI_DEGREE=parser->getMAPMRIOrder();
    MAPMRIModel mapmri_estimator;
    float small_delta=parser->getSmallDelta();
    float big_delta=parser->getBigDelta();
    if(small_delta==0 && big_delta==0)
    {
        std::cout<<"Small and big delta not entered. Computing heuristic values."<<std::endl;

        double max_bval= bvals.max_value();
        //If the small and big deltas are unknown, just make a guesstimate
        //using the max bvalue and assumed gradient strength
        double gyro= 267.51532*1E6;
        double G= 40*1E-3;  //well most scanners are either 40 mT/m or 80mT/m.

        G*=2;

        double temp= max_bval/gyro/gyro/G/G/2.*1E6;
        // assume that big_delta = 3 * small_delta
        // deltas are in miliseconds
        small_delta= pow(temp,1./3.)*1000.;
        big_delta= small_delta*3;
        std::cout<< "small_delta: "<<small_delta<<" big_delta: "<<big_delta<<std::endl;
    }

    mapmri_estimator.SetMAPMRIDegree(FINAL_STAGE_MAPMRI_DEGREE);
    mapmri_estimator.SetDTImg(dti_img);
    mapmri_estimator.SetA0Image(A0_img);
    mapmri_estimator.SetBmatrix(Bmatrix);
    mapmri_estimator.SetDWIData(final_data);
    mapmri_estimator.SetWeightImage(weight_imgs);
    mapmri_estimator.SetVoxelwiseBmatrix(dummyv);
    mapmri_estimator.SetMaskImage(mask_image);
    mapmri_estimator.SetVolIndicesForFitting(dummy);
    mapmri_estimator.SetSmallDelta(small_delta);
    mapmri_estimator.SetBigDelta(big_delta);
    mapmri_estimator.PerformFitting();
    mapmri_estimator.ComputeEigenImages();

    auto mapmri_image = mapmri_estimator.GetOutput();
    MAPMRIModel::EValImageType::Pointer eval_image= mapmri_estimator.getEvalImage();


    double tdiff= big_delta-small_delta/3.;
    itk::ImageRegionIteratorWithIndex<MAPMRIModel::EValImageType> it(eval_image,eval_image->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        MAPMRIModel::EValType v= it.Get();
        v[0]= sqrt(v[0]*2000.*tdiff);
        v[1]= sqrt(v[1]*2000.*tdiff);
        v[2]= sqrt(v[2]*2000.*tdiff);
        it.Set(v);
        ++it;
    }


    std::string full_base_name=input_name.substr(0, input_name.find(".nii"));
    std::string map_name= full_base_name + std::string("_mapmri.nii");
    std::string uvec_name= full_base_name + std::string("_uvec.nii");



    int MAP_DEGREE=FINAL_STAGE_MAPMRI_DEGREE;
    int N_MAPMRI_COEFFS =((((MAP_DEGREE/2)+1)*((MAP_DEGREE/2)+2)*(4*(MAP_DEGREE/2)+3))/6);
    ImageType3D::SizeType sz= A0_img->GetLargestPossibleRegion().GetSize();


    typedef itk::VectorImage<float,3> WrImageType;
    WrImageType::Pointer mapmri_vec= WrImageType::New();
    mapmri_vec->SetRegions(mapmri_image->GetLargestPossibleRegion());
    mapmri_vec->SetSpacing(mapmri_image->GetSpacing());
    mapmri_vec->SetOrigin(mapmri_image->GetOrigin());
    mapmri_vec->SetDirection(mapmri_image->GetDirection());
    mapmri_vec->SetNumberOfComponentsPerPixel(N_MAPMRI_COEFFS);
    long numberOfPixels= (long)sz[0]*sz[1]*sz[2];
    mapmri_vec->GetPixelContainer()->SetImportPointer(mapmri_image->GetBufferPointer(), numberOfPixels,    false);

    typedef itk::ImageFileWriter< WrImageType> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetInput(mapmri_vec);
    wr->SetFileName(map_name);
    wr->Update();



    writeImageD<MAPMRIModel::EValImageType>(eval_image,uvec_name);

}

