#ifndef _DIFFPREP_H
#define _DIFFPREP_H


#include "TORTOISE.h"
#include "mecc_settings.h"


class DIFFPREP
{
public:
    using CompositeTransformType= TORTOISE::CompositeTransformType;
    using OkanQuadraticTransformType= TORTOISE::OkanQuadraticTransformType;
    using DisplacementFieldTransformType= TORTOISE::DisplacementFieldTransformType;


public:
    DIFFPREP(){};
    DIFFPREP(std::string data_name, json mjs);
    ~DIFFPREP(){};


protected:
    vnl_matrix<int> ParseJSONForSliceTiming(json cjson);
    void GetSmallBigDelta(float &small_delta,float &big_delta);


    template<typename ImageType>
    typename ImageType::Pointer QuadratictransformImage(typename ImageType::Pointer img,CompositeTransformType::Pointer trans,std::string interp_method,float default_val);

    template <typename ImageType>
    typename ImageType::Pointer ChangeImageHeaderToDP(typename ImageType::Pointer img);


    ImageType3D::Pointer dmc_make_target(ImageType3D::Pointer img,ImageType3D::Pointer  mask);
    std::vector<float> choose_range(ImageType3D::Pointer b0_img,ImageType3D::Pointer curr_vol, ImageType3D::Pointer b0_mask_img);
    void ClassicMotionEddyCorrectAllDWIs(ImageType3D::Pointer target, std::vector<ImageType3D::Pointer> dwis);
    void SynthMotionEddyCorrectAllDWIs(std::vector<ImageType3D::Pointer> target_imgs, std::vector<ImageType3D::Pointer> source_imgs);


    std::vector<ImageType3D::Pointer>  ReplaceOutliers( std::vector<ImageType3D::Pointer> native_native_synth_dwis,  std::vector<ImageType3D::Pointer> raw_dwis,std::vector<int> shells,vnl_vector<double> bvals,ImageType3D::Pointer TR_map);
    void EM(std::vector< std::vector<float> >  logRMS_shell, std::vector< std::vector<float> > &per_shell_inliers, std::vector< std::vector<float> > &per_shell_outliers ,std::vector<float> &Pin_per_shell,std::vector<float> &medians,std::vector<float> &MADs);

private:            //Subfunctions the main processing functions use

    std::vector<ImageType3D::Pointer> TransformRepolData(std::string nii_filename, vnl_matrix<double> &rot_Bmatrix, std::vector<ImageType3DBool::Pointer> &final_inclusion_imgs);

    bool  CheckVolumeInclusion(ImageType3DBool::Pointer inc_vol);
    ImageType3D::Pointer  ComputeMedianB0Img(std::vector<ImageType3D::Pointer> dwis,vnl_vector<double> bvals);


private:                    //Main processing functions
    void ProcessData();
    void PadAndWriteImage();
    void SetBoId();
    void DPCreateMask();
    void WriteOutputFiles();
    void MotionAndEddy();


protected:

    int Nvols;
    ImageType3D::Pointer b0_mask_img{nullptr};    
    std::vector<ImageType3D::Pointer> eddy_s2v_replaced_synth_dwis;
    std::vector<ImageType3D::Pointer> native_native_synth_dwis;


    MeccSettings *mecc_settings;

    json my_json;
    std::string nii_name;
    vnl_matrix<double> Bmatrix;

    TORTOISE::TeeStream *stream;

private:

    std::vector<CompositeTransformType::Pointer>  dwi_transforms;
    std::vector<std::vector<OkanQuadraticTransformType::Pointer> > s2v_transformations;
    std::vector<ImageType3D::Pointer> native_weight_img;

    int b0_vol_id;
    std::string PE_string;


};



#endif

