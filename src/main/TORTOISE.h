#ifndef _TORTOISE_H
#define _TORTOISE_H


#include <string>
#include "defines.h"

#include "TORTOISE_parser.h"
#include "OmpThreadBase.h"


#include "itkCompositeTransform.h"
#include "itkDisplacementFieldTransform.h"
#include "itkEuler3DTransform.h"
#include "itkAffineTransform.h"
#include "itkOkanQuadraticTransform.h"
#include "../utilities/TORTOISE_Utilities.h"


#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/tee.hpp>

#include <fstream>
#include <iostream>


class TORTOISE: public OMPTHREADBASE
{
public:

    using OkanQuadraticTransformType=itk::OkanQuadraticTransform<CoordType,3,3>;
    using DisplacementFieldTransformType= itk::DisplacementFieldTransform<CoordType,3> ;
    using DisplacementFieldType= DisplacementFieldTransformType::DisplacementFieldType;
    using CompositeTransformType= itk::CompositeTransform<CoordType,3>;
    using RigidTransformType= itk::Euler3DTransform<CoordType>;
    using AffineTransformType= itk::AffineTransform<CoordType,3>;

    using Tee=boost::iostreams::tee_device<std::ostream, std::ofstream>;
    using TeeStream= boost::iostreams::stream<Tee>;
private:

    enum STEPS
    {
        Import,Denoising,Gibbs,MotionEddy,Drift,EPI,StructuralAlignment,FinalData,Unknown
    };

    STEPS ConvertStringToStep(std::string str)
    {
        std::string str2=str;
        std::transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

        if(str2=="import")
            return STEPS::Import;
        if(str2=="denoising")
            return STEPS::Denoising;
        if(str2=="gibbs")
            return STEPS::Gibbs;
        if(str2=="motioneddy")
            return STEPS::MotionEddy;
        if(str2=="epi")
            return STEPS::EPI;
        if(str2=="structuralalignment")
            return STEPS::StructuralAlignment;
        if(str2=="drift")
            return STEPS::Drift;
        if(str2=="finaldata")
            return STEPS::FinalData;

        return STEPS::Unknown;
    }

    struct NIFTI_BMAT
    {
        std::string nii_name;
        std::string bmtxt_name;
        std::string json_name;
    };



private:
    void FillReportJson();


private:


    bool CheckIfInputsOkay();
    void Process();

    void LoadDefaultSettings();
    void UpdateSettingsFromCommandLine();
    void WriteFinalSettings();


    void CheckAndCopyInputData();
    void DenoiseData(std::string input_name,double &b0_noise_mean, double &b0_noise_std);
    void GibbsUnringData(std::string input_name,float PF,std::string PE);
    void EPICorrectData();
    void AlignB0ToReorientation();
    void DriftCorrect(std::string nii_name);



public:

    TORTOISE()
    {       
        TORTOISE::stream = nullptr;

        SetNMaxCores(getNCores());
        int nc=getNCores();
        SetNAvailableCores(nc);
        omp_set_num_threads(GetNAvailableCores());

        std::vector<uint> thread_array;
        thread_array.resize(GetNAvailableCores());
        for(int t=0;t<thread_array.size();t++)
            thread_array[t]=0;
        TORTOISE::SetThreadArray(thread_array);
    }


    TORTOISE(int argc, char * argv[]);
    ~TORTOISE()
    {
    }

    void entryPoint();
    static void * staticEntryPoint(void * c){((TORTOISE*)c)->entryPoint();return NULL;};

    void start()
    {
        pthread_create(&thread, NULL, TORTOISE::staticEntryPoint, this);
    }






private:

    pthread_t thread;
    TORTOISE_PARSER *parser;


    std::string temp_proc_folder;
    std::string output_name;
    std::vector<NIFTI_BMAT>  proc_infos;
    std::vector<json> my_jsons;

    std::vector<ImageType3D::Pointer> final_data;
    ImageType3D::Pointer final_mask{nullptr};

    json processing_report_json;

public:
    static TeeStream* stream;
    static std::string executable_folder;



};



#endif
