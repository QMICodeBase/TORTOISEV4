#ifndef _DRTAMASDIFFEO_H
#define _DRTAMASDIFFEO_H


#include "drtamas_structs.h"
#include "defines.h"
#include "TORTOISE.h"

#include "DRTAMAS_parser.h"


#include "cuda_image.h"
#include "../cuda_src/cuda_image_utilities.h"


class DRTAMAS_Diffeo
{    
    public:


    using CurrentFieldType = CUDAIMAGE;
    using CurrentImageType = CUDAIMAGE;    


    DRTAMAS_Diffeo()
    {
        this->stream= &(std::cout);
    }

    ~DRTAMAS_Diffeo(){};

    DisplacementFieldType::Pointer getDef()
    {
        CurrentFieldType::Pointer def_F= InvertField(def_FINV);
        CurrentFieldType::Pointer def =  ComposeFields(def_F,this->def_MINV);


        DisplacementFieldType::Pointer disp=def->CudaImageToITKField();
        return disp;
    }

    void SetFixedStructurals(std::vector<ImageType3D::Pointer> si)
    {
        for(int i=0;i<si.size();i++)
        {
            CurrentImageType::Pointer str_img=CUDAIMAGE::New();            
            str_img->SetImageFromITK(si[i]);            
            this->fixed_structurals.push_back(str_img);
        }
    }
    void SetMovingStructurals(std::vector<ImageType3D::Pointer> si)
    {
        for(int i=0;i<si.size();i++)
        {
            CurrentImageType::Pointer str_img=CUDAIMAGE::New();
            str_img->SetImageFromITK(si[i]);
            this->moving_structurals.push_back(str_img);
        }
    }

    void SetFixedTensor(DTMatrixImageType::Pointer tens)
    {
        this->fixed_tensor = CUDAIMAGE::New();
        this->fixed_tensor->SetTImageFromITK(tens);
    }

    void SetMovingTensor(DTMatrixImageType::Pointer tens)
    {
        this->moving_tensor = CUDAIMAGE::New();
        this->moving_tensor->SetTImageFromITK(tens);
    }




    void SetStagesFromExternal(std::vector<DRTAMASStageSettings> st){stages=st;}
    void SetParser(DRTAMAS_PARSER *prs){parser=prs;};


private:            //Subfunctions the main processing functions use
    void SetImagesForMetrics();
    void SetUpStages();
    void SetDefaultStages();
    

public:                    //Main processing functions
    void Process();

  //  std::string GetRegistrationMethodType(){return parser->getRegistrationMethodType();}



private:                    //Main processing functions





private:          //class member variables

    CUDAIMAGE::Pointer fixed_tensor{nullptr};
    CUDAIMAGE::Pointer moving_tensor{nullptr};
    std::vector<CUDAIMAGE::Pointer> fixed_structurals;
    std::vector<CUDAIMAGE::Pointer> moving_structurals;

    CUDAIMAGE::Pointer def_FINV{nullptr};
    CUDAIMAGE::Pointer def_MINV{nullptr};


    std::vector<DRTAMASStageSettings> stages;


    std::ostream  *stream;    
    DRTAMAS_PARSER *parser{nullptr};

};



#endif

