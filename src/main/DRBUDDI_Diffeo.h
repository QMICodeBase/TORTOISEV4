#ifndef _DRBUDDIDIFFEO_H
#define _DRBUDDIDIFFEO_H


#include "drbuddi_structs.h"
#include "defines.h"
#include "TORTOISE_parser.h"
#include "TORTOISE.h"

#include "itkDisplacementFieldTransform.h"

#ifdef USECUDA
    #include "cuda_image.h"
#else
    #include "itkImageDuplicator.h"
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
    DisplacementFieldType::Pointer getDefFINV()
    {
        DisplacementFieldType::Pointer disp=def_FINV->CudaImageToITKField();
        disp->SetDirection(orig_dir);
        itk::ImageRegionIterator<DisplacementFieldType> it(disp,disp->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            DisplacementFieldType::PixelType pix= it.Get();
            vnl_vector<double> vec=orig_dir.GetVnlMatrix()*new_dir.GetTranspose()* pix.GetVnlVector();
            pix[0]=vec[0];
            pix[1]=vec[1];
            pix[2]=vec[2];
            it.Set(pix);
        }
        return disp;

    }

    DisplacementFieldType::Pointer getDefMINV()
    {
        DisplacementFieldType::Pointer disp=def_MINV->CudaImageToITKField();
        disp->SetDirection(orig_dir);
        itk::ImageRegionIterator<DisplacementFieldType> it(disp,disp->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            DisplacementFieldType::PixelType pix= it.Get();
            vnl_vector<double> vec=orig_dir.GetVnlMatrix()*new_dir.GetTranspose()* pix.GetVnlVector();
            pix[0]=vec[0];
            pix[1]=vec[1];
            pix[2]=vec[2];
            it.Set(pix);
        }
        return disp;
    }

    void SetStructuralImages(std::vector<ImageType3D::Pointer> si)
    {
        for(int i=0;i<si.size();i++)
        {
            CurrentImageType::Pointer str_img=CUDAIMAGE::New();
            si[i]->SetDirection(new_dir);
            str_img->SetImageFromITK(si[i]);
            si[i]->SetDirection(orig_dir);
            structural_imgs.push_back(str_img);
        }
    }

    void SetB0UpImage(ImageType3D::Pointer img)
    {
        orig_dir = img->GetDirection();
        new_dir=orig_dir;
        new_dir.Fill(0);
        for(int r=0;r<3;r++)
        {

            float mx=-2;
            int mx_id=-1;
            for(int c=0;c<3;c++)
            {
                if(fabs(orig_dir(r,c))>mx)
                {
                    mx=fabs(orig_dir(r,c));
                    mx_id=c;
                }
            }
            if(orig_dir(r,mx_id)>0)
                new_dir(r,mx_id)=1;
            else
                new_dir(r,mx_id)=-1;
        }
        img->SetDirection(new_dir);
        this->b0_up_img=CUDAIMAGE::New();
        this->b0_up_img->SetImageFromITK(img);
        img->SetDirection(orig_dir);
    }
    void SetB0DownImage(ImageType3D::Pointer img)
    {
        img->SetDirection(new_dir);
        this->b0_down_img=CUDAIMAGE::New();
        this->b0_down_img->SetImageFromITK(img);
        img->SetDirection(orig_dir);
    }
    void SetFAUpImage(ImageType3D::Pointer img)
    {
        img->SetDirection(new_dir);
        this->FA_up_img=CUDAIMAGE::New();
        this->FA_up_img->SetImageFromITK(img);
        img->SetDirection(orig_dir);
    }
    void SetFADownImage(ImageType3D::Pointer img)
    {
        img->SetDirection(new_dir);
        this->FA_down_img=CUDAIMAGE::New();
        this->FA_down_img->SetImageFromITK(img);
        img->SetDirection(orig_dir);
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
    DisplacementFieldType::Pointer getDefFINV()
    {
        DisplacementFieldType::Pointer disp=def_FINV;
        disp->SetDirection(orig_dir);
        itk::ImageRegionIterator<DisplacementFieldType> it(disp,disp->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            DisplacementFieldType::PixelType pix= it.Get();
            vnl_vector<double> vec=orig_dir.GetVnlMatrix()*new_dir.GetTranspose()* pix.GetVnlVector();
            pix[0]=vec[0];
            pix[1]=vec[1];
            pix[2]=vec[2];
            it.Set(pix);
        }
        return disp;
    }
    DisplacementFieldType::Pointer getDefMINV()
    {
        DisplacementFieldType::Pointer disp=def_MINV;
        disp->SetDirection(orig_dir);
        itk::ImageRegionIterator<DisplacementFieldType> it(disp,disp->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            DisplacementFieldType::PixelType pix= it.Get();
            vnl_vector<double> vec=orig_dir.GetVnlMatrix()*new_dir.GetTranspose()* pix.GetVnlVector();
            pix[0]=vec[0];
            pix[1]=vec[1];
            pix[2]=vec[2];
            it.Set(pix);
        }
        return disp;
    }

    void SetB0UpImage(ImageType3D::Pointer img2)
    {
        orig_dir = img2->GetDirection();
        new_dir=orig_dir;
        new_dir.Fill(0);
        for(int r=0;r<3;r++)
        {

            float mx=-2;
            int mx_id=-1;
            for(int c=0;c<3;c++)
            {
                if(fabs(orig_dir(r,c))>mx)
                {
                    mx=fabs(orig_dir(r,c));
                    mx_id=c;
                }
            }
            if(orig_dir(r,mx_id)>0)
                new_dir(r,mx_id)=1;
            else
                new_dir(r,mx_id)=-1;
        }
        typedef itk::ImageDuplicator<ImageType3D> DupType;
        DupType::Pointer dup= DupType::New();
        dup->SetInputImage(img2);
        dup->Update();
        ImageType3D::Pointer img= dup->GetOutput();
        img->SetDirection(new_dir);

        b0_up_img=img;
    }
    void SetB0DownImage(ImageType3D::Pointer img2)
    {
        typedef itk::ImageDuplicator<ImageType3D> DupType;
        DupType::Pointer dup= DupType::New();
        dup->SetInputImage(img2);
        dup->Update();
        ImageType3D::Pointer img= dup->GetOutput();
        img->SetDirection(new_dir);
        b0_down_img=img;
    }
    void SetFAUpImage(ImageType3D::Pointer img2)
    {
        typedef itk::ImageDuplicator<ImageType3D> DupType;
        DupType::Pointer dup= DupType::New();
        dup->SetInputImage(img2);
        dup->Update();
        ImageType3D::Pointer img= dup->GetOutput();
        img->SetDirection(new_dir);
        FA_up_img=img;
    }
    void SetFADownImage(ImageType3D::Pointer img2)
    {
        typedef itk::ImageDuplicator<ImageType3D> DupType;
        DupType::Pointer dup= DupType::New();
        dup->SetInputImage(img2);
        dup->Update();
        ImageType3D::Pointer img= dup->GetOutput();
        img->SetDirection(new_dir);
        FA_down_img=img;
    }
    void SetStructuralImages(std::vector<ImageType3D::Pointer> si)
    {
        structural_imgs=si;
        for(int s=0;s<structural_imgs.size();s++ )
        {
            typedef itk::ImageDuplicator<ImageType3D> DupType;
            DupType::Pointer dup= DupType::New();
            dup->SetInputImage(si[s]);
            dup->Update();
            ImageType3D::Pointer img= dup->GetOutput();
            img->SetDirection(new_dir);
            structural_imgs[s]=img;
        }
    }

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

    std::string GetRegistrationMethodType(){return parser->getRegistrationMethodType();}



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

    ImageType3D::DirectionType orig_dir,new_dir;



};



#endif

