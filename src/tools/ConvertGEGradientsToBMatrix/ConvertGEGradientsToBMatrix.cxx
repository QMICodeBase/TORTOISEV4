#ifndef __CONVERTGEGRADIENTSTOBMATRIX_CXX
#define __CONVERTGEGRADIENTSTOBMATRIX_CXX


#include "defines.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/math_utilities.h"

vnl_matrix<double> read_gradients_file(std::string fname)
{
     std::ifstream infile(fname.c_str());
     int Nvolumes;
     infile>>Nvolumes;
     vnl_matrix<double> grads;
     grads.set_size(Nvolumes,3);
     infile>>grads;
     infile.close();

     return grads;
}


int main(int argc, char* argv[])
{
    if(argc<4)
    {
        std::cout<<"Tool to convert GE tensor.dat format to a TORTOISE BMatrix."<<std::endl;
        std::cout<<"The gradients should be in read/phase/slice coordinate system and scaled with the square root of b-value"<<std::endl;
        std::cout<<"The json file corresponding to the DWI NIFTI file should exist."<<std::endl;
        std::cout<<"The first line of the grad file should contain the total number of volumes."<<std::endl;
        std::cout<<"USAGE:"<<std::endl;
        std::cout<<"ConvertGEGradientsToBMatrix dwi_nifti_file grad_file max_bval"<<std::endl;
        return EXIT_FAILURE;
    }


    std::string grad_name=argv[2];
    std::string dwi_name= argv[1];
    float max_bval= atof(argv[3]);
    std::string json_name = dwi_name.substr(0, dwi_name.rfind(".nii")) +".json";

    vnl_matrix<double> grads= read_gradients_file(grad_name);


    json dwi_json;
    std::ifstream json_stream(json_name);
    json_stream >> dwi_json;
    json_stream.close();

    std::string PE_string = dwi_json["PhaseEncodingDirection"];

    int Nvols=grads.rows();
    vnl_matrix<double> Bmatrix(Nvols,6);

    for(int v=0;v<Nvols;v++)
    {
        vnl_vector<double> curr_vec(3);
        curr_vec[0]=-grads(v,0);
        curr_vec[1]=grads(v,1);
        curr_vec[2]=grads(v,2);

        double mag= curr_vec.magnitude();
        float bval =mag*mag*max_bval;
        if(mag!=0)
        {
            curr_vec[0]=curr_vec[0]/mag;
            curr_vec[1]=curr_vec[1]/mag;
            curr_vec[2]=curr_vec[2]/mag;
        }

        Bmatrix(v,0)=    bval * curr_vec[0] * curr_vec[0];
        Bmatrix(v,1)= 2* bval * curr_vec[0] * curr_vec[1];
        Bmatrix(v,2)= 2* bval * curr_vec[0] * curr_vec[2];
        Bmatrix(v,3)=    bval * curr_vec[1] * curr_vec[1];
        Bmatrix(v,4)= 2* bval * curr_vec[1] * curr_vec[2];
        Bmatrix(v,5)=    bval * curr_vec[2] * curr_vec[2];
    }

    // So far we have not considered the phase encoding direction, image acquisition plane, and nifti header.
    //Let's handle these one by one

    ImageType3D::Pointer img = read_3D_volume_from_4D(dwi_name,0);


    vnl_vector<double> i_vec(3,0), j_vec(3,0), k_vec(3,0);
    i_vec[0]=1;
    j_vec[1]=1;
    k_vec[2]=1;
    vnl_vector<double> slice_phys_vec= img->GetDirection().GetVnlMatrix() * k_vec;

    int sliceDir=3;
    if(  (fabs(slice_phys_vec[0])>fabs(slice_phys_vec[1])) && (fabs(slice_phys_vec[0])>fabs(slice_phys_vec[2])) )
        sliceDir=1;
    if(  (fabs(slice_phys_vec[1])>fabs(slice_phys_vec[0])) && (fabs(slice_phys_vec[1])>fabs(slice_phys_vec[2])) )
        sliceDir=2;


    bool col=true;
    if(PE_string.find("j")==std::string::npos)
        col=false;

    if(!col)
     {
         vnl_vector<double> Bxx= Bmatrix.get_column(0);
         vnl_vector<double> Bxy= Bmatrix.get_column(1);
         vnl_vector<double> Bxz= Bmatrix.get_column(2);
         vnl_vector<double> Byy= Bmatrix.get_column(3);
         vnl_vector<double> Byz= Bmatrix.get_column(4);
         vnl_vector<double> Bzz= Bmatrix.get_column(5);

         Bmatrix.set_column(0, Byy);
         Bmatrix.set_column(1, -Bxy);
         Bmatrix.set_column(2, -Byz);
         Bmatrix.set_column(3, Bxx);
         Bmatrix.set_column(4, Bxz);
         Bmatrix.set_column(5, Bzz);
     }



    if (abs(sliceDir) == 1) //SAGITTAL
    {
        vnl_vector<double> Bxx= Bmatrix.get_column(0);
        vnl_vector<double> Bxy= Bmatrix.get_column(1);
        vnl_vector<double> Bxz= Bmatrix.get_column(2);
        vnl_vector<double> Byy= Bmatrix.get_column(3);
        vnl_vector<double> Byz= Bmatrix.get_column(4);
        vnl_vector<double> Bzz= Bmatrix.get_column(5);

        Bmatrix.set_column(0, Bzz);
        Bmatrix.set_column(1, Bxz);
        Bmatrix.set_column(2, Byz);
        Bmatrix.set_column(3, Bxx);
        Bmatrix.set_column(4, Bxy);
        Bmatrix.set_column(5, Byy);
    }

    if (abs(sliceDir) == 2) //CORONAL
    {
        vnl_vector<double> Bxx= Bmatrix.get_column(0);
        vnl_vector<double> Bxy= Bmatrix.get_column(1);
        vnl_vector<double> Bxz= Bmatrix.get_column(2);
        vnl_vector<double> Byy= Bmatrix.get_column(3);
        vnl_vector<double> Byz= Bmatrix.get_column(4);
        vnl_vector<double> Bzz= Bmatrix.get_column(5);

        Bmatrix.set_column(0, Bxx);
        Bmatrix.set_column(1, Bxz);
        Bmatrix.set_column(2, Bxy);
        Bmatrix.set_column(3, Bzz);
        Bmatrix.set_column(4, Byz);
        Bmatrix.set_column(5, Byy);
    }

    vnl_matrix<double> eye(3,3);
    eye.set_identity();
    eye(1,1)=-1;

    vnl_vector<double> ivec_phys= img->GetDirection().GetVnlMatrix() * eye* i_vec;
    vnl_vector<double> jvec_phys= img->GetDirection().GetVnlMatrix() * eye * j_vec;
    vnl_vector<double> kvec_phys= img->GetDirection().GetVnlMatrix() * eye * k_vec;

    bool flip_i=false, flip_j=false, flip_k=false;

    {
        vnl_vector<double> vec_phys=ivec_phys;
        double mx_val = vec_phys.max_value();
        double mn_val = vec_phys.min_value();

        if( fabs(mx_val) < fabs(mn_val))
            if(sgn<double>(mn_val) ==-1)
                flip_i=true;
        if( fabs(mx_val) > fabs(mn_val))
            if(sgn<double>(mx_val) ==-1)
                flip_i=true;
    }
    {
        vnl_vector<double> vec_phys=jvec_phys;
        double mx_val = vec_phys.max_value();
        double mn_val = vec_phys.min_value();

        if( fabs(mx_val) < fabs(mn_val))
            if(sgn<double>(mn_val) ==-1)
                flip_j=true;
        if( fabs(mx_val) > fabs(mn_val))
            if(sgn<double>(mx_val) ==-1)
                flip_j=true;
    }
    {
        vnl_vector<double> vec_phys=kvec_phys;
        double mx_val = vec_phys.max_value();
        double mn_val = vec_phys.min_value();

        if( fabs(mx_val) < fabs(mn_val))
            if(sgn<double>(mn_val) ==-1)
                flip_k=true;
        if( fabs(mx_val) > fabs(mn_val))
            if(sgn<double>(mx_val) ==-1)
                flip_k=true;
    }



    if(flip_i)
    {
        vnl_vector<double> Bxy= Bmatrix.get_column(1);
        vnl_vector<double> Bxz= Bmatrix.get_column(2);

        Bmatrix.set_column(1, -Bxy);
        Bmatrix.set_column(2, -Bxz);
    }
    if(flip_j)
    {
        vnl_vector<double> Bxy= Bmatrix.get_column(1);
        vnl_vector<double> Byz= Bmatrix.get_column(4);

        Bmatrix.set_column(1, -Bxy);
        Bmatrix.set_column(4, -Byz);
    }
    if(flip_k)
    {
        vnl_vector<double> Bxz= Bmatrix.get_column(2);
        vnl_vector<double> Byz= Bmatrix.get_column(4);

        Bmatrix.set_column(2, -Bxz);
        Bmatrix.set_column(4, -Byz);
    }


    std::string bmtxt_name= dwi_name.substr(0,dwi_name.rfind(".nii"))+".bmtxt";

    std::ofstream outbmtxtfile(bmtxt_name);
    outbmtxtfile<<Bmatrix;
    outbmtxtfile.close();








    return EXIT_SUCCESS;
}


#endif
