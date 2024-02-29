#ifndef _DIFFUSIONMODEL_H
#define _DIFFUSIONMODEL_H


#include "defines.h"
#include "TORTOISE.h"


template< typename T >
class DiffusionModel
{
public:
    using OutputImageType =  T;


private:


public:
    DiffusionModel(){stream=TORTOISE::stream;}
    ~DiffusionModel()
    {
        this->output_img=nullptr;
        this->noise_img=nullptr;

        this->dwi_data.clear();
        this->weight_imgs.clear();
        this->voxelwise_Bmatrix.clear();
        this->graddev_img.clear();

        this->mask_img=nullptr;
        this->A0_img=nullptr;
    };

    void SetJson( json njson){my_json=njson;}
    void SetBmatrix( vnl_matrix<double> bmat){Bmatrix=bmat;}
    void SetDWIData(std::vector<ImageType3D::Pointer> ndata){dwi_data=ndata;}
    void SetWeightImage(std::vector<ImageType3D::Pointer> ninc){weight_imgs=ninc;}
    void SetVoxelwiseBmatrix(std::vector<std::vector<ImageType3D::Pointer> > vbmat){voxelwise_Bmatrix=vbmat;}
    void SetGradDev(std::vector<ImageType3D::Pointer> gd){graddev_img=gd;}
    void SetMaskImage(ImageType3D::Pointer mi){mask_img=mi;}
    void SetVolIndicesForFitting(std::vector<int> inf){indices_fitting=inf;}
    void SetA0Image(ImageType3D::Pointer ai){A0_img=ai;}
    ImageType3D::Pointer GetA0Image(){return A0_img;}
    typename OutputImageType::Pointer GetOutput(){return output_img;}
    void SetOutput(typename OutputImageType::Pointer oi){output_img=oi;}
    void SetFreeWaterDiffusivity(float fwd){free_water_diffusivity=fwd;}
    void SetNoiseImg(ImageType3D::Pointer ni){noise_img=ni;}

    std::vector<ImageType3D::Pointer> GetWeightImage()    {return weight_imgs;}


    vnl_matrix<double>  getCurrentBmatrix(ImageType3D::IndexType ind3,std::vector<int> curr_all_indices)
    {
        vnl_matrix<double> curr_Bmatrix(curr_all_indices.size(),6);

        vnl_matrix_fixed<double,3,3> L, B;
        if(graddev_img.size())
        {
            L(0,0)= graddev_img[0]->GetPixel(ind3); L(0,1)= graddev_img[1]->GetPixel(ind3); L(0,2)= graddev_img[2]->GetPixel(ind3);
            L(1,0)= graddev_img[3]->GetPixel(ind3); L(1,1)= graddev_img[4]->GetPixel(ind3); L(1,2)= graddev_img[5]->GetPixel(ind3);
            L(2,0)= graddev_img[6]->GetPixel(ind3); L(2,1)= graddev_img[7]->GetPixel(ind3); L(2,2)= graddev_img[8]->GetPixel(ind3);

            L=L.transpose();
        }


        for(int vol=0;vol<curr_Bmatrix.rows();vol++)
        {
            int vol_id= curr_all_indices[vol];

            if(graddev_img.size())
            {
                B(0,0)= Bmatrix(vol_id,0); B(0,1)= Bmatrix(vol_id,1)/2;  B(0,2)= Bmatrix(vol_id,2)/2;
                B(1,0)= Bmatrix(vol_id,1)/2; B(1,1)= Bmatrix(vol_id,3);  B(1,2)= Bmatrix(vol_id,4)/2;
                B(2,0)= Bmatrix(vol_id,2)/2; B(2,1)= Bmatrix(vol_id,4)/2;  B(2,2)= Bmatrix(vol_id,5);

                B= L * B * L.transpose();

                curr_Bmatrix(vol,0)= B(0,0);
                curr_Bmatrix(vol,1)= 2*B(0,1);
                curr_Bmatrix(vol,2)= 2*B(0,2);
                curr_Bmatrix(vol,3)= B(1,1);
                curr_Bmatrix(vol,4)= 2*B(1,2);
                curr_Bmatrix(vol,5)= B(2,2);
            }
            else
            {
                if(voxelwise_Bmatrix.size())
                {
                    curr_Bmatrix(vol,0)= voxelwise_Bmatrix[vol_id][0]->GetPixel(ind3);
                    curr_Bmatrix(vol,1)= voxelwise_Bmatrix[vol_id][1]->GetPixel(ind3);
                    curr_Bmatrix(vol,2)= voxelwise_Bmatrix[vol_id][2]->GetPixel(ind3);
                    curr_Bmatrix(vol,3)= voxelwise_Bmatrix[vol_id][3]->GetPixel(ind3);
                    curr_Bmatrix(vol,4)= voxelwise_Bmatrix[vol_id][4]->GetPixel(ind3);
                    curr_Bmatrix(vol,5)= voxelwise_Bmatrix[vol_id][5]->GetPixel(ind3);
                }
                else
                {
                    curr_Bmatrix(vol,0)=Bmatrix(vol_id,0);
                    curr_Bmatrix(vol,1)=Bmatrix(vol_id,1);
                    curr_Bmatrix(vol,2)=Bmatrix(vol_id,2);
                    curr_Bmatrix(vol,3)=Bmatrix(vol_id,3);
                    curr_Bmatrix(vol,4)=Bmatrix(vol_id,4);
                    curr_Bmatrix(vol,5)=Bmatrix(vol_id,5);
                }
            }
        }
        return curr_Bmatrix;
    }





    virtual void PerformFitting(){};
    virtual ImageType3D::Pointer SynthesizeDWI(vnl_vector<double> bmat_vec){return nullptr;};





protected:

    json my_json;
    vnl_matrix<double> Bmatrix;

    ImageType3D::Pointer noise_img{nullptr};

    std::vector<ImageType3D::Pointer> dwi_data;
    std::vector<ImageType3D::Pointer> weight_imgs;
    std::vector<std::vector<ImageType3D::Pointer> > voxelwise_Bmatrix;
    std::vector<ImageType3D::Pointer> graddev_img;

    ImageType3D::Pointer mask_img{nullptr};
    std::vector<int> indices_fitting;

    ImageType3D::Pointer A0_img{nullptr};
    typename OutputImageType::Pointer output_img{nullptr};
    float free_water_diffusivity;


    TORTOISE::TeeStream *stream;

};



#endif
