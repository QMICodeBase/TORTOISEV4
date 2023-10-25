#ifndef _FINALDATA_H
#define _FINALDATA_H

#include "TORTOISE.h"
#include "../tools/gradnonlin/gradcal.h"

class FINALDATA
{    
    using OkanQuadraticTransformType= TORTOISE::OkanQuadraticTransformType;
    using DisplacementFieldTransformType= TORTOISE::DisplacementFieldTransformType;
    using DisplacementFieldType=TORTOISE::DisplacementFieldType;
    using RigidTransformType = TORTOISE::RigidTransformType;
    using CompositeTransformType= TORTOISE::CompositeTransformType;
    using BaseTransformType= itk::Transform<double, 3, 3 >;

public:

    FINALDATA(std::string orig_upname, json orig_upjson, std::string orig_downname, json orig_downjson)
    {
        epi_trans[0]=nullptr;
        epi_trans[1]=nullptr;
        Nvols[0]=0;
        Nvols[1]=0;


        data_names.push_back(orig_upname);
        jsons.push_back(orig_upjson);
        data_names.push_back(orig_downname);
        jsons.push_back(orig_downjson);


        for(int d=0;d<2;d++)
        {
            if(data_names[d]!="")
            {
                std::string json_PE= jsons[d]["PhaseEncodingDirection"];      //get phase encoding direction
                if(json_PE.find("j")!=std::string::npos)
                    PE_strings[d]="vertical";
                else
                    if(json_PE.find("i")!=std::string::npos)
                        PE_strings[d]="horizontal";
                    else
                        PE_strings[d]="slice";
            }
        }



        std::string nname= fs::path(data_names[0]).filename().string();
        std::string basename = nname.substr(0, nname.find(".nii"));
        temp_folder= fs::path(data_names[0]).parent_path().string()  + "/" +  basename + std::string("_temp_proc");
        stream=TORTOISE::stream;
    }

    void SetTempFolder(std::string tf){temp_folder=tf;}
    void SetParser(TORTOISE_PARSER *p){parser=p;}
    void SetOutputName(std::string nm){output_name=nm;}


    ~FINALDATA(){};  

private:            //Subfunctions the main processing functions use        

    void ReadOrigTransforms();

    CompositeTransformType::Pointer GenerateCompositeTransformForVolume(ImageType3D::Pointer ref_img, int PE, int vol);
    template <typename ImageType>
    typename ImageType::Pointer ChangeImageHeaderToDP(typename ImageType::Pointer img);


    ImageType3D::Pointer GenerateStructurals();
    ImageType3D::Pointer GenerateFirstStructural();
    void  GenerateFinalData(std::vector< std::vector<ImageType3D::Pointer> > dwis);
    std::vector< std::vector<ImageType3D::Pointer> >  GenerateTransformedInterpolatedData();
    void GenerateGradNonlinOutput();
    ImageType3D::Pointer ComputeDetImgFromAllTransExceptStr(ImageType3D::Pointer ref_vol,int vol_id,int PE);


    ImageType3D::Pointer UnObliqueImage(ImageType3D::Pointer img);

    std::vector<ImageType3D::Pointer> ComputeVBMatImgFromCoeffs(int UPDOWN);
    std::vector<ImageType3D::Pointer> ComputeLImgFromCoeffs();
    std::vector<ImageType3D::Pointer> ComputeVBMatImgFromField(int UPDOWN);
    std::vector<ImageType3D::Pointer> ComputeLImgFromField();
    InternalMatrixType  pixel_bmatrix(const GradCoef &E, ImageType3D::PointType point,const vnl_vector<double> &norms);


    std::vector<ImageType3D::Pointer>  ComputeS2VInverse(int PE);




public:                    //Main processing functions
    void Generate();



private:
    RigidTransformType::Pointer b0_t0_str_trans{nullptr};
    RigidTransformType::Pointer b0down_t0_b0up_trans{nullptr};

    DisplacementFieldType::Pointer epi_trans[2];
    DisplacementFieldType::Pointer gradwarp_field{nullptr};
    DisplacementFieldType::Pointer gradwarp_field_forward{nullptr};

    std::vector<OkanQuadraticTransformType::Pointer>  dwi_transforms[2];
    std::vector<std::vector<OkanQuadraticTransformType::Pointer> > s2v_transformations[2];


    std::vector<ImageType3D::Pointer> native_weight_img[2];
    std::vector<ImageType3D::Pointer> native_native_synth_img[2];

    std::vector<double> drift_params[2];


    int Nvols[2];

    typedef std::pair<double,int> mypair;
    static bool comparator ( const mypair& l, const mypair& r)
    {
        return l.first < r.first;
    }




private:

    std::string PE_strings[2];
    ImageType3D::Pointer template_structural{nullptr};

    std::vector<std::string> data_names;
    std::vector<json> jsons;
    std::string temp_folder;
    std::string output_name;

    TORTOISE::TeeStream *stream{nullptr};
    TORTOISE_PARSER *parser{nullptr};

};



#endif

