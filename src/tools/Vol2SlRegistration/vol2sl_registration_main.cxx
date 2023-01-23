#include "defines.h"
#include "register_dwi_to_slice.h"
#ifdef USECUDA
    #include "../cuda_src/register_dwi_to_slice_cuda.h"
#endif
#include "itkOkanQuadraticTransform.h"

#include <iostream>
#include <fstream>

using OkanQuadraticTransformType=itk::OkanQuadraticTransform<double,3,3>;

std::vector<float> choose_range(ImageType3D::Pointer b0_img,ImageType3D::Pointer curr_vol, ImageType3D::Pointer b0_mask_img)
{
     std::vector<float> fixed_signal;
     std::vector<float> moving_signal;

     float moving_max=-1E10;
     float moving_min = 1E10;

     itk::ImageRegionIteratorWithIndex<ImageType3D> it(b0_mask_img,b0_mask_img->GetLargestPossibleRegion());
     it.GoToBegin();
     while(!it.IsAtEnd())
     {
         ImageType3D::IndexType index=it.GetIndex();
          if(it.Get()!=0)
          {
              fixed_signal.push_back(b0_img->GetPixel(index));
              moving_signal.push_back(curr_vol->GetPixel(index));
          }
          if(curr_vol->GetPixel(index)> moving_max)
              moving_max=curr_vol->GetPixel(index);
          if(curr_vol->GetPixel(index)< moving_min)
              moving_min=curr_vol->GetPixel(index);

         ++it;
     }

     std::sort (fixed_signal.begin(), fixed_signal.end());
     std::sort (moving_signal.begin(), moving_signal.end());

     float koeff=0.005;
     int nb= fixed_signal.size();

     int ind= (nb-1) - koeff*nb;

     std::vector<float> lim_arr;
     lim_arr.resize(4);
     lim_arr[0]=0.1;
     lim_arr[1]= fixed_signal[ind];
     //lim_arr[2]=0.1;
    // lim_arr[3]= moving_signal[ind];
     lim_arr[2]=moving_min;
     lim_arr[3]= moving_max;

     return lim_arr;
}


vnl_matrix<int> ParseJSONForSliceTiming(json cjson)
{
    //Use the Slice timings in the JSON file to convert them into FSL style slspec (slice specifications).

    std::vector<float> slice_timing= cjson["SliceTiming"];

    std::vector<size_t> idx(slice_timing.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),  [&slice_timing](size_t i1, size_t i2) {return slice_timing[i1] < slice_timing[i2];});

    int nslices= slice_timing.size();
    float first_t= slice_timing[0];
    int MB=0;
    for(int s=0;s<nslices;s++)
        if(slice_timing[s]==first_t)
            MB++;

    vnl_matrix<int> slspec(nslices/MB,MB);
    for(int s=0;s<nslices; s++)
    {
        int r = s/MB;
        int c = s%MB;
        slspec(r,c)= idx[s];
    }
    return slspec;
}

int main(int argc, char*argv[])
{

    if(argc==1)
    {
        std::cout<<"Usage:  Vol2SLRegistration synth_image  slice_image3d  json_file  mask_image do_qudratic(optional.0/1.  default:1) "<<std::endl;
        return EXIT_FAILURE;
    }


    json my_json;
    std::ifstream jstream(argv[3]);
    jstream >> my_json;
    jstream.close();

    std::string PE_string;
    std::string json_PE= my_json["PhaseEncodingDirection"];      //get phase encoding direction
    if(json_PE.find("j")!=std::string::npos)
        PE_string="vertical";
    else
        if(json_PE.find("i")!=std::string::npos)
            PE_string="horizontal";
        else
            PE_string="slice";


    vnl_matrix<int> slspec= ParseJSONForSliceTiming(my_json);

    ImageType3D::Pointer target= readImageD<ImageType3D>(argv[2]);
    ImageType3D::Pointer native_synth_img= readImageD<ImageType3D>(argv[1]);
    ImageType3D::Pointer b0_mask_img= readImageD<ImageType3D>(argv[4]);
    bool do_q=true;
    if(argc>5)
        do_q= (bool)(atoi(argv[5]));
    
        
    int Nexc= slspec.rows();
    int MB=slspec.columns();
    
    std::vector<float> signal_ranges = choose_range(target, native_synth_img,b0_mask_img);
    
    std::vector<OkanQuadraticTransformType::Pointer> s2v_transformations;


    std::cout<<"ranges: " << signal_ranges[0]<< " " << signal_ranges[1]<< " " << signal_ranges[2]<< " " << signal_ranges[3]<< " " <<std::endl;

//    #ifdef USECUDA
  //         VolumeToSliceRegistration_cuda(target, native_synth_img,slspec,signal_ranges,s2v_transformations,do_q,PE_string);
  //  #else
           VolumeToSliceRegistration(target, native_synth_img,slspec,signal_ranges,s2v_transformations,do_q,PE_string);
   // #endif
    
        
    ImageType3D::Pointer native_native_synth_dwis = ImageType3D::New();
    native_native_synth_dwis->SetRegions(target->GetLargestPossibleRegion());
    native_native_synth_dwis->Allocate();
    native_native_synth_dwis->SetSpacing(target->GetSpacing());
    native_native_synth_dwis->SetOrigin(target->GetOrigin());
    native_native_synth_dwis->SetDirection(target->GetDirection());


    ImageType3D::SizeType sz= native_native_synth_dwis->GetLargestPossibleRegion().GetSize();
    ImageType3D::Pointer curr_vol = target;

    typedef itk::LinearInterpolateImageFunction<ImageType3D,double> InterpolatorType;
    InterpolatorType::Pointer interp = InterpolatorType::New();
    interp->SetInputImage(native_synth_img);

    for(int k=0;k<sz[2];k++)
    {
        ImageType3D::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;

                ImageType3D::PointType pt;
                curr_vol->TransformIndexToPhysicalPoint(ind3,pt);
                ImageType3D::PointType pt_trans=s2v_transformations[k]->TransformPoint(pt);

                ImageType3D::PixelType interp_val =0;
                if(interp->IsInsideBuffer(pt_trans))
                    interp_val=interp->Evaluate(pt_trans);
                native_native_synth_dwis->SetPixel(ind3,interp_val);
            }
        }
    }

    for(int k=0;k<sz[2];k++)
    {
        std::cout<< "k: "<< k << " P: "<< s2v_transformations[k]->GetParameters()<<std::endl;
    }



    {
        std::string nm(argv[1]);
        std::string oname = nm.substr(0,nm.rfind(".nii")) +std::string("_synth_reg.nii");
        writeImageD<ImageType3D>(native_native_synth_dwis,oname);
    }
    {
       ImageType3D::Pointer sl_img_reg= ForwardTransformImage(target, s2v_transformations);
       std::string nm(argv[2]);
       std::string oname = nm.substr(0,nm.rfind(".nii")) +std::string("_reg.nii");
       writeImageD<ImageType3D>(sl_img_reg,oname);
    }


}


