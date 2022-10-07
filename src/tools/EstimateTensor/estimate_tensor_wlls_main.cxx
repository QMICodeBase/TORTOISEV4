#include "estimate_tensor_wlls.h"
#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "estimate_tensor_wlls_parser.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "DTIModel.h"
#include "itkNiftiImageIO.h"


vnl_matrix<double> read_bvecs_bvals(std::string bvals_file, std::string bvecs_file, int Nvols )
{
    vnl_matrix<double> bvecs(3,Nvols);
    vnl_matrix<double> bvecs_transposed(Nvols,3);
    bool use_transposed_bvecs=false;
    vnl_vector<double> bvals(Nvols);
    vnl_matrix<double> Bmatrix(Nvols,6);


    std::ifstream infileb(bvals_file.c_str());
    infileb>>bvals;
    infileb.close();

    std::string line;
    std::ifstream infile(bvecs_file.c_str());
    int nlines=0;
    while (std::getline(infile, line))
    {
        line.erase(remove(line.begin(), line.end(), ' '), line.end());
        if(line.length()>1)
            nlines++;
    }
    infile.close();
    if(nlines>3)
        use_transposed_bvecs=true;

    std::ifstream infile2(bvecs_file.c_str());
    if(use_transposed_bvecs)
    {
        infile2>>bvecs_transposed;
        bvecs= bvecs_transposed.transpose();
    }
    else
        infile2>>bvecs;
    infile2.close();


    for(int i=0;i<Nvols;i++)
    {
        vnl_matrix<double> vec= bvecs.get_n_columns(i,1);
        double nrm= sqrt(vec(0,0)*vec(0,0) + vec(1,0)*vec(1,0) + vec(2,0)*vec(2,0) );
        if(nrm > 1E-3)
        {
            vec(0,0)/=nrm;
            vec(1,0)/=nrm;
            vec(2,0)/=nrm;
        }

        vnl_matrix<double> mat = bvals[i] * vec * vec.transpose();
        Bmatrix(i,0)=mat(0,0);
        Bmatrix(i,1)=2*mat(0,1);
        Bmatrix(i,2)=2*mat(0,2);
        Bmatrix(i,3)=mat(1,1);
        Bmatrix(i,4)=2*mat(1,2);
        Bmatrix(i,5)=mat(2,2);
    }

    return Bmatrix;
}


int main(int argc, char *argv[])
{

    EstimateTensorWLLS_PARSER *parser= new EstimateTensorWLLS_PARSER(argc,argv);

    std::string input_name = parser->getInputImageName();
    std::string bmtxt_name= input_name.substr(0,input_name.rfind(".nii"))+".bmtxt";

    vnl_matrix<double> Bmatrix;
    if(!fs::exists(bmtxt_name))
    {
        std::string bvecs_name= input_name.substr(0,input_name.rfind(".nii"))+".bvec";
        std::string bvals_name= input_name.substr(0,input_name.rfind(".nii"))+".bval";

        itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
        myio->SetFileName(input_name);
        myio->ReadImageInformation();
        int Nvols= myio->GetDimensions(3);
        Bmatrix= read_bvecs_bvals( bvals_name,  bvecs_name, Nvols );
    }
    else
    {
        Bmatrix= read_bmatrix_file(bmtxt_name);
    }



    int Nvols= Bmatrix.rows();

    std::vector<ImageType3D::Pointer> final_data;
    final_data.resize(Nvols);
    std::vector<ImageType3D::Pointer> inclusion_imgs;

    if(parser->getInclusionImg()!="")
        inclusion_imgs.resize(Nvols);

    for(int v=0;v<Nvols;v++)
    {
        final_data[v]= read_3D_volume_from_4D(input_name,v);
        if(parser->getInclusionImg()!="")
            inclusion_imgs[v]= read_3D_volume_from_4D(parser->getInclusionImg(),v);
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


    std::vector<int> bindices;
    {
        double bval_cut =parser->getBValCutoff();
        for(int i=0;i<Bmatrix.rows();i++)
        {
            double bval= Bmatrix(i,0)+ Bmatrix(i,3)+Bmatrix(i,5);
            if(bval<=1.05*bval_cut)
                bindices.push_back(i);
        }
    }

    std::string regresion_mode = parser->getRegressionMode();

    std::vector<ImageType3D::Pointer> graddev_img;
    std::vector<  std::vector< ImageType3D::Pointer> > vbmat_img;


    if(parser->getUseVoxelwiseBmat())
    {
        std::string Lname= input_name.substr(0,input_name.rfind(".nii")) + "_graddev.nii";
        if(fs::exists(Lname))
        {
            graddev_img.resize(9);
            for(int v=0;v<9;v++)
            {
                graddev_img[v]=read_3D_volume_from_4D(Lname,v);
            }
        }
        else
        {
            std::string vbmat_name= input_name.substr(0,input_name.rfind(".nii")) + "_vbmat.nii";
            if(fs::exists(vbmat_name))
            {
                vbmat_img.resize(Nvols);
                for(int v=0; v< Nvols;v++)
                {
                    vbmat_img[v].resize(6);
                    for(int bi=0;bi<6;bi++)
                    {
                        vbmat_img[v][bi]= read_3D_volume_from_4D(vbmat_name, v*6+bi);
                    }
                }
            }
            else
            {
                std::cout<<"Voxelwise bmat or grad_dev use requested but these files are not present. Disregarding this option."<<std::endl;
            }
        }
    }




    std::vector<int> dummyv;
    DTIModel dti_estimator;
    dti_estimator.SetBmatrix(Bmatrix);
    dti_estimator.SetDWIData(final_data);
    dti_estimator.SetWeightImage(inclusion_imgs);
    dti_estimator.SetVoxelwiseBmatrix(vbmat_img);
    dti_estimator.SetGradDev(graddev_img);
    dti_estimator.SetMaskImage(mask_image);
    dti_estimator.SetVolIndicesForFitting(bindices);
    dti_estimator.SetFittingMode(regresion_mode);
    dti_estimator.PerformFitting();


    ImageType3D::Pointer A0_image=dti_estimator.GetA0Image();
    DTImageType::Pointer dt_image= dti_estimator.GetOutput();

    typedef ImageType4D DTImageType4D;

    DTImageType4D::IndexType start; start.Fill(0);
    DTImageType4D::SizeType sz;
    sz[0]= dt_image->GetLargestPossibleRegion().GetSize()[0];
    sz[1]= dt_image->GetLargestPossibleRegion().GetSize()[1];
    sz[2]= dt_image->GetLargestPossibleRegion().GetSize()[2];
    sz[3]= 6;
    DTImageType4D::RegionType reg(start,sz);

    DTImageType4D::PointType orig;
    orig[0]=dt_image->GetOrigin()[0];
    orig[1]=dt_image->GetOrigin()[1];
    orig[2]=dt_image->GetOrigin()[2];
    orig[3]=0;


    DTImageType4D::SpacingType spc;
    spc[0]=dt_image->GetSpacing()[0];
    spc[1]=dt_image->GetSpacing()[1];
    spc[2]=dt_image->GetSpacing()[2];
    spc[3]=1;

    DTImageType4D::DirectionType dir;
    dir.SetIdentity();
    dir(0,0)= dt_image->GetDirection()(0,0);dir(0,1)= dt_image->GetDirection()(0,1);dir(0,2)= dt_image->GetDirection()(0,2);
    dir(1,0)= dt_image->GetDirection()(1,0);dir(1,1)= dt_image->GetDirection()(1,1);dir(1,2)= dt_image->GetDirection()(1,2);
    dir(2,0)= dt_image->GetDirection()(2,0);dir(2,1)= dt_image->GetDirection()(2,1);dir(2,2)= dt_image->GetDirection()(2,2);


    DTImageType4D::Pointer dt_image4d= DTImageType4D::New();
    dt_image4d->SetRegions(reg);
    dt_image4d->Allocate();
    dt_image4d->SetOrigin(orig);
    dt_image4d->SetDirection(dir);
    dt_image4d->SetSpacing(spc);



    itk::ImageRegionIteratorWithIndex<DTImageType> it(dt_image,dt_image->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        DTImageType::IndexType index= it.GetIndex();
        DTType  dt_vec= it.Get();
        DTImageType4D::IndexType ind4d;
        ind4d[0]=index[0];
        ind4d[1]=index[1];
        ind4d[2]=index[2];

        ind4d[3]=0;
        dt_image4d->SetPixel(ind4d, dt_vec[0]*1000000.);
        ind4d[3]=1;
        dt_image4d->SetPixel(ind4d, dt_vec[3]*1000000.);
        ind4d[3]=2;
        dt_image4d->SetPixel(ind4d, dt_vec[5]*1000000.);
        ind4d[3]=3;
        dt_image4d->SetPixel(ind4d, dt_vec[1]*1000000.);
        ind4d[3]=4;
        dt_image4d->SetPixel(ind4d, dt_vec[2]*1000000.);
        ind4d[3]=5;
        dt_image4d->SetPixel(ind4d, dt_vec[4]*1000000.);
        ++it;
    }

    std::string ext;
    if(regresion_mode=="WLLS")
        ext="_L1";
    if(regresion_mode=="NLLS")
        ext="_N1";
    if(regresion_mode=="RESTORE")
        ext="_R1";
    if(regresion_mode=="DIAG")
        ext="_N0";
    if(regresion_mode=="N2")
        ext="_N2";
    if(regresion_mode=="NT2")
        ext="_NT2";



    std::string full_base_name=input_name.substr(0, input_name.find(".nii"));
    std::string DT_name= full_base_name + ext+  std::string("_DT.nii");
    std::string AM_name= full_base_name + ext+  std::string("_AM.nii");

    writeImageD<ImageType3D>(A0_image,AM_name);
    writeImageD<ImageType4D>(dt_image4d,DT_name);
    if(parser->getWriteCSImg())
    {
        std::string CS_name= full_base_name + ext+  std::string("_CS.nii");
        if(dti_estimator.getCSImg())
            writeImageD<ImageType3D>(dti_estimator.getCSImg(),CS_name);
    }

    if(dti_estimator.getVFImg())
    {
        std::string VF_name= full_base_name + ext+  std::string("_VF.nii");
        writeImageD<ImageType3D>(dti_estimator.getVFImg(),VF_name);

        ImageType3D::Pointer VF_image = dti_estimator.getVFImg();
        ImageType3D::Pointer one_m_VF_img= ImageType3D::New();
        one_m_VF_img->SetRegions(VF_image->GetLargestPossibleRegion());
        one_m_VF_img->Allocate();
        one_m_VF_img->SetSpacing(VF_image->GetSpacing());
        one_m_VF_img->SetDirection(VF_image->GetDirection());
        one_m_VF_img->SetOrigin(VF_image->GetOrigin());
        one_m_VF_img->FillBuffer(0);
        itk::ImageRegionIteratorWithIndex<ImageType3D> it2(one_m_VF_img,one_m_VF_img->GetLargestPossibleRegion());
        it2.GoToBegin();
        while(!it2.IsAtEnd())
        {
            ImageType3D::IndexType index = it2.GetIndex();
            DTType  dt_vec= dt_image->GetPixel(index);
            if(dt_vec[0]+dt_vec[3]+dt_vec[5] > 1E-8)
            {
                it2.Set(1-VF_image->GetPixel(index));
            }
            ++it2;
        }

        std::string OMVF_name= full_base_name + ext+  std::string("_OMVF.nii");
        writeImageD<ImageType3D>(one_m_VF_img,OMVF_name);
    }
    if(dti_estimator.getVFImg2())
    {
        std::string VF_name= full_base_name + ext+  std::string("_VF2.nii");
        writeImageD<ImageType3D>(dti_estimator.getVFImg2(),VF_name);

    }


    if(dti_estimator.getFlowImg())
    {
        DTImageType::Pointer flow_img= dti_estimator.getFlowImg();

        DTImageType4D::Pointer flow_img4d= DTImageType4D::New();
        flow_img4d->SetRegions(reg);
        flow_img4d->Allocate();
        flow_img4d->SetOrigin(orig);
        flow_img4d->SetDirection(dir);
        flow_img4d->SetSpacing(spc);


        itk::ImageRegionIteratorWithIndex<DTImageType> it(flow_img,flow_img->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            DTImageType::IndexType index= it.GetIndex();
            DTType  dt_vec= it.Get();
            DTImageType4D::IndexType ind4d;
            ind4d[0]=index[0];
            ind4d[1]=index[1];
            ind4d[2]=index[2];

            ind4d[3]=0;
            flow_img4d->SetPixel(ind4d, dt_vec[0]*1000000.);
            ind4d[3]=1;
            flow_img4d->SetPixel(ind4d, dt_vec[3]*1000000.);
            ind4d[3]=2;
            flow_img4d->SetPixel(ind4d, dt_vec[5]*1000000.);
            ind4d[3]=3;
            flow_img4d->SetPixel(ind4d, dt_vec[1]*1000000.);
            ind4d[3]=4;
            flow_img4d->SetPixel(ind4d, dt_vec[2]*1000000.);
            ind4d[3]=5;
            flow_img4d->SetPixel(ind4d, dt_vec[4]*1000000.);
            ++it;
        }

        std::string flow_name= full_base_name + ext+  std::string("_DT2.nii");
        writeImageD<ImageType4D>(flow_img4d,flow_name);

    }


}

