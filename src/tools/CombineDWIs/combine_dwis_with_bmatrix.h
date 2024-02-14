#ifndef __COMBINEDWISWITHBMATRIX_H
#define __COMBINEDWISWITHBMATRIX_H


#include "defines.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "../utilities/extract_3Dvolume_from_4D.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../tools/TORTOISEBmatrixToFSLBVecs/tortoise_bmatrix_to_fsl_bvecs.h"

void CombineDWIsWithBMatrix(std::vector<std::string> nii_names, std::string output_name)
{

    int Nimgs= nii_names.size();

    int tot_Nvols=0;
    for(int ni=0;ni<Nimgs;ni++)
    {
        std::string nii_name = nii_names[ni];
        itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
        myio->SetFileName(nii_name);
        myio->ReadImageInformation();
        int Nvols= myio->GetDimensions(3);
        tot_Nvols+=Nvols;
    }

    vnl_matrix<double> tot_Bmatrix(tot_Nvols,6);
    std::cout<<"Total volumes: "<< tot_Nvols<<std::endl;

    int vols_so_far=0;
    bool all_data_have_vbmat=true;
    for(int ni=0;ni<Nimgs;ni++)
    {
        std::string nii_name = nii_names[ni];
        ImageType4D::Pointer img = readImageD<ImageType4D>(nii_name) ;
        int Nvols = img->GetLargestPossibleRegion().GetSize()[3];

        std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii"))+".bmtxt";

        vnl_matrix<double> Bmatrix= read_bmatrix_file(bmtxt_name);
        tot_Bmatrix.update(Bmatrix,vols_so_far,0);

        for(int v=0;v<Nvols;v++)
        {
            ImageType3D::Pointer vol = extract_3D_volume_from_4D(img,v);
            write_3D_image_to_4D_file<float>(vol,output_name,vols_so_far+v,tot_Nvols);
        }
        vols_so_far+=Nvols;


        std::string vbmat_name = nii_name.substr(0,nii_name.rfind(".nii")) + "_vbmat.nii";
        if(!fs::exists(vbmat_name))
            all_data_have_vbmat=false;
    }


    if(all_data_have_vbmat)
    {
        std::string output_vbmat_name = output_name.substr(0,output_name.rfind(".nii"))+"_vbmat.nii";
        vols_so_far=0;
        for(int ni=0;ni<Nimgs;ni++)
        {
            std::string nii_name = nii_names[ni];
            std::string vbmat_name = nii_name.substr(0,nii_name.rfind(".nii")) + "_vbmat.nii";
            std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii"))+".bmtxt";
            vnl_matrix<double> Bmatrix= read_bmatrix_file(bmtxt_name);
            int Nvols = Bmatrix.rows();

            for(int v=0;v<6*Nvols;v++)
            {
                ImageType3D::Pointer vol = read_3D_volume_from_4D(vbmat_name,v);
                write_3D_image_to_4D_file<float>(vol,output_vbmat_name,vols_so_far+v,6*tot_Nvols);
            }
            vols_so_far+=6*Nvols;
        }
    }




    std::string bmat_name= output_name.substr(0,output_name.rfind(".nii")) + ".bmtxt";
    std::string bvals_fname= output_name.substr(0,output_name.rfind(".nii")) + ".bvals";
    std::string bvecs_fname= output_name.substr(0,output_name.rfind(".nii")) + ".bvecs";
    std::ofstream outfile(bmat_name);
    outfile<<tot_Bmatrix;
    outfile.close();

    vnl_matrix<double> bvecs(3,tot_Nvols);
    vnl_matrix<double> bvals= tortoise_bmatrix_to_fsl_bvecs(tot_Bmatrix, bvecs);


    std::ofstream bvecs_file(bvecs_fname.c_str());
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<tot_Nvols;j++)
        {
            if(bvals(0,j)==0)
                bvecs_file<<"0 ";
            else
                bvecs_file<< bvecs(i,j)<< " ";
        }
        bvecs_file<<std::endl;
    }
    bvecs_file.close();


    std::ofstream bvals_file(bvals_fname.c_str());


        for(int j=0;j<tot_Nvols;j++)
        {
            bvals_file<< bvals(0,j)<< " ";
        }
    bvals_file.close();


}

#endif
