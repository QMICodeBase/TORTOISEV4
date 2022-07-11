#include <string>
#include <vector>
#include "defines.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "../utilities/extract_3Dvolume_from_4D.h"
#include "../tools/TORTOISEBmatrixToFSLBVecs/tortoise_bmatrix_to_fsl_bvecs.h"






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
    if(argc < 5 || (argc-2)%3 !=0)
    {
        std::cout<< "Usage: CombineDWIs output_nifti_filename nifti1_filename bvals1_filename bvecs1_filename  .......... niftiN_filename bvalsN_filename bvecsN_filename"<<std::endl;
        return EXIT_FAILURE;
    }

    int Nimgs= (argc-2)/3;

    std::string output_name= argv[1];
    if(output_name.find(".gz")!=std::string::npos)
    {
        output_name= output_name.substr(0,output_name.rfind(".gz")) + ".nii";
    }

    fs::path output_path(output_name);
    if(!fs::exists(output_path.parent_path()))
            fs::create_directories(output_path.parent_path());


    int tot_Nvols=0;
    for(int ni=0;ni<Nimgs;ni++)
    {
        std::string nii_name = argv[ni*3+2];
        itk::NiftiImageIO::Pointer myio = itk::NiftiImageIO::New();
        myio->SetFileName(nii_name);
        myio->ReadImageInformation();
        int Nvols= myio->GetDimensions(3);
        tot_Nvols+=Nvols;
    }

    vnl_matrix<double> tot_Bmatrix(tot_Nvols,6);
    std::cout<<"Total volumes: "<< tot_Nvols<<std::endl;


    int vols_so_far=0;
    for(int ni=0;ni<Nimgs;ni++)
    {
        std::string nii_name = argv[ni*3+2];
        ImageType4D::Pointer img = readImageD<ImageType4D>(nii_name) ;
        int Nvols = img->GetLargestPossibleRegion().GetSize()[3];

        vnl_matrix<double> Bmatrix= read_bvecs_bvals(argv[ni*3+3], argv[ni*3+4], Nvols);
        tot_Bmatrix.update(Bmatrix,vols_so_far,0);

        for(int v=0;v<Nvols;v++)
        {
            ImageType3D::Pointer vol = extract_3D_volume_from_4D(img,v);
            write_3D_image_to_4D_file<float>(vol,output_name,vols_so_far+v,tot_Nvols);
        }
        vols_so_far+=Nvols;
    }

    std::string bmat_name= output_name.substr(0,output_name.rfind(".nii")) + ".bmtxt";
    std::string bvals_fname= output_name.substr(0,output_name.rfind(".nii")) + ".bvals";
    std::string bvecs_fname= output_name.substr(0,output_name.rfind(".nii")) + ".bvecs";
    std::ofstream outfile(bmat_name);
    outfile<<tot_Bmatrix;
    outfile.close();

    vnl_matrix<double> bvecs;
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
