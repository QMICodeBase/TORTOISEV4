#ifndef _DRBUDDIDIFFEO_H
#define _DRBUDDIDIFFEO_H


#include "drbuddi_structs.h"
#include "defines.h"
#include "TORTOISE_parser.h"
#include "TORTOISE.h"

#include "itkDisplacementFieldTransform.h"

#ifdef USECUDA
    #include "cuda_image.h"
#endif

class DRBUDDI_Diffeo
{    
    public:

    using DisplacementFieldTransformType= TORTOISE::DisplacementFieldTransformType;

    #ifdef USECUDA
        using CurrentFieldType = CUDAIMAGE;
        using CurrentImageType = CUDAIMAGE;
        using PhaseEncodingVectorType = float3;
    #else
        using CurrentFieldType = DisplacementFieldType;
        using CurrentImageType = ImageType3D;
        using PhaseEncodingVectorType = vnl_vector<double>;
    #endif



    DRBUDDI_Diffeo()
    {
#ifdef DRBUDDIALONE
    this->stream= &(std::cout);
#else
    this->stream= TORTOISE::stream;
#endif
    }
    ~DRBUDDI_Diffeo(){};

#ifdef USECUDA
    DisplacementFieldType::Pointer getDefFINV(){return def_FINV->CudaImageToITKField();}
    DisplacementFieldType::Pointer getDefMINV(){return def_MINV->CudaImageToITKField();}

    void SetStructuralImages(std::vector<ImageType3D::Pointer> si)
    {
        for(int i=0;i<si.size();i++)
        {
            CurrentImageType::Pointer str_img=CUDAIMAGE::New();
            str_img->SetImageFromITK(si[i]);
            structural_imgs.push_back(str_img);
        }
    }

    void SetB0UpImage(ImageType3D::Pointer img)
    {
        this->b0_up_img=CUDAIMAGE::New();
        this->b0_up_img->SetImageFromITK(img);
    }
    void SetB0DownImage(ImageType3D::Pointer img)
    {
        this->b0_down_img=CUDAIMAGE::New();
        this->b0_down_img->SetImageFromITK(img);
    }
    void SetFAUpImage(ImageType3D::Pointer img)
    {
        this->FA_up_img=CUDAIMAGE::New();
        this->FA_up_img->SetImageFromITK(img);
    }
    void SetFADownImage(ImageType3D::Pointer img)
    {
        this->FA_down_img=CUDAIMAGE::New();
        this->FA_down_img->SetImageFromITK(img);
    }
    void SetUpPEVector(vnl_vector<double> pe)
    {
        up_phase_vector.x=pe[0];up_phase_vector.y=pe[1];up_phase_vector.z=pe[2];
    }
    void SetDownPEVector(vnl_vector<double> pe)
    {
        down_phase_vector.x=pe[0];down_phase_vector.y=pe[1];down_phase_vector.z=pe[2];
    }
#else
    DisplacementFieldType::Pointer getDefFINV(){return def_FINV;}
    DisplacementFieldType::Pointer getDefMINV(){return def_MINV;}

    void SetB0UpImage(ImageType3D::Pointer img){b0_up_img=img;}
    void SetB0DownImage(ImageType3D::Pointer img){b0_down_img=img;}
    void SetFAUpImage(ImageType3D::Pointer img){ImageType3D::Pointer FA_up_img=img;}
    void SetFADownImage(ImageType3D::Pointer img){ImageType3D::Pointer FA_down_img=img;}
    void SetStructuralImages(std::vector<ImageType3D::Pointer> si){structural_imgs=si;}

    void SetUpPEVector(vnl_vector<double> pe) {up_phase_vector=pe;}
    void SetDownPEVector(vnl_vector<double> pe) {down_phase_vector=pe;}
#endif

    void SetStagesFromExternal(std::vector<DRBUDDIStageSettings> st){stages=st;}
    void SetParser(DRBUDDI_PARSERBASE *prs){parser=prs;};


private:            //Subfunctions the main processing functions use
    void SetImagesForMetrics();
    void SetUpStages();
    void SetDefaultStages();
    

public:                    //Main processing functions
    void Process();




private:                    //Main processing functions





private:          //class member variables

#ifdef USECUDA
    CUDAIMAGE::Pointer b0_up_img{nullptr},b0_down_img{nullptr};
    CUDAIMAGE::Pointer FA_up_img{nullptr},FA_down_img{nullptr};
    std::vector<CUDAIMAGE::Pointer> structural_imgs;

    CUDAIMAGE::Pointer def_FINV{nullptr};
    CUDAIMAGE::Pointer def_MINV{nullptr};

#else
    ImageType3D::Pointer b0_up_img{nullptr},b0_down_img{nullptr};
    ImageType3D::Pointer FA_up_img{nullptr},FA_down_img{nullptr};
    std::vector<ImageType3D::Pointer> structural_imgs;

    DisplacementFieldType::Pointer def_FINV{nullptr};
    DisplacementFieldType::Pointer def_MINV{nullptr};

#endif

    std::vector<DRBUDDIStageSettings> stages;

#ifdef DRBUDDIALONE
    std::ostream  *stream;
#else
    TORTOISE::TeeStream *stream;
#endif
    DRBUDDI_PARSERBASE *parser{nullptr};

    PhaseEncodingVectorType up_phase_vector, down_phase_vector;


};



#endif

