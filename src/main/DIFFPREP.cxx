#ifndef _DIFFPREP_CXX
#define _DIFFPREP_CXX

#include "DIFFPREP.h"
#include "../utilities/read_bmatrix_file.h"
#include "registration_settings.h"
#include "../tools/PadImage/pad_image.hxx"
#include "../tools/SelectBestB0/select_best_b0.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../tools/ResampleDWIs/resample_dwis.h"

#include "../tools/EstimateTensor/DTIModel.h"
#include "../tools/EstimateMAPMRI/MAPMRIModel.h"


#include "register_dwi_to_b0.hxx"
#ifdef USECUDA
    #include "register_dwi_to_b0_cuda.h"
    #include "register_dwi_to_slice_cuda.h"
#endif


#include "create_mask.h"
#include "../utilities/math_utilities.h"
#include "../tools/ConvertQuadraticTransformToDisplacementField/convert_eddy_trans_to_field.hxx"
#include "register_dwi_to_slice.h"

#include "../tools/RotateBMatrix/rotate_bmatrix.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"


#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <random>



#include "itkRegionOfInterestImageFilter.h"
#include "itkInvertDisplacementFieldImageFilter.h"

#define  SLICE_NOTCONSIDER 2


DIFFPREP::DIFFPREP(std::string data_name, json mjs)
{
    this->stream= TORTOISE::stream;        //for logging

    my_json=mjs;
    nii_name=data_name;


    std::string json_PE= my_json["PhaseEncodingDirection"];      //get phase encoding direction
    if(json_PE.find("j")!=std::string::npos)
        PE_string="vertical";
    else
        if(json_PE.find("i")!=std::string::npos)
            PE_string="horizontal";
        else
            PE_string="slice";

    this->Bmatrix= read_bmatrix_file(nii_name.substr(0,nii_name.rfind(".nii"))+std::string(".bmtxt"));
    this->Nvols = this->Bmatrix.rows();


    //between volume motion& eddy currents transformations
    //not initializing s2v transformations because they will be option and data dependent.
    // so will be initialized later if needed.
    dwi_transforms.resize(Nvols);
    for(int vol=0;vol<Nvols;vol++)
        dwi_transforms[vol]= CompositeTransformType::New();



    //Mecc settings file describe many registration parameters for motion& eddy correction.
    //They actually correspond to files in the settings folder.
    this->mecc_settings=nullptr;
    std::string mecc_choice= RegistrationSettings::get().getValue<std::string>(std::string("correction_mode"));
    if(mecc_choice!="off")
    {

        std::string mecc_filename1= TORTOISE::executable_folder +  std::string("/../settings/mecc_settings/")+ mecc_choice + std::string(".mec");
        if(!fs::exists(mecc_filename1))
        {
            (*stream)<<"Mecc settings file " << mecc_filename1 << " could not be found. Using quadratic settings instead."<<std::endl;
            mecc_filename1= TORTOISE::executable_folder +  std::string("/../settings/mecc_settings/quadratic.mec");
        }
        this->mecc_settings = new MeccSettings(mecc_filename1);
    }

    ProcessData();
}


void DIFFPREP::ProcessData()
{
   // PadAndWriteImage();           //padding image to prevent issues with very narrow FoV data

    SetBoId();            // Set our reference b0 image.

    DPCreateMask();         // This mask will be used for many purposes

    MotionAndEddy();      // Main motion & eddy-currents & slice-2-volume and & outlier replacement correction

    WriteOutputFiles();  //Write The necessary files for further TORTOISE processing
}


vnl_matrix<int> DIFFPREP::ParseJSONForSliceTiming(json cjson)
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

void DIFFPREP::GetSmallBigDelta(float &small_delta,float &big_delta)
{
    vnl_vector<double> bvals = Bmatrix.get_column(0) + Bmatrix.get_column(3)+ Bmatrix.get_column(5);
    double max_bval= bvals.max_value();

    if(my_json["BigDelta"]==json::value_t::null || my_json["SmallDelta"]==json::value_t::null)
    {
        //If the small and big deltas are unknown, just make a guesstimate
        //using the max bvalue and assumed gradient strength

        double gyro= 267.51532*1E6;
        double G= 40*1E-3;  //well most scanners are either 40 mT/m or 80mT/m.
        if(my_json["ManufacturersModelName"]!=json::value_t::null)
        {
            std::string scanner_model=my_json["ManufacturersModelName"];
            if(scanner_model.find("Prisma")!=std::string::npos)
                G= 80*1E-3;
        }

        double temp= max_bval/gyro/gyro/G/G/2.*1E6;

        // assume that big_delta = 3 * small_delta
        // deltas are in miliseconds
        small_delta= pow(temp,1./3.)*1000.;
        big_delta= small_delta*3;
        my_json["BigDelta"]= big_delta;
        my_json["SmallDelta"]= small_delta;

        std::string json_name = this->nii_name.substr(0,this->nii_name.rfind(".nii"))+".json";
        std::ofstream out_json(json_name);
        out_json << std::setw(4) << this->my_json << std::endl;
        out_json.close();
    }
    else
    {
        //If the small and big deltas are entered by the user , use them.
        big_delta = my_json["BigDelta"];
        small_delta = my_json["SmallDelta"];
    }
}



template <typename ImageType>
typename ImageType::Pointer DIFFPREP::ChangeImageHeaderToDP(typename ImageType::Pointer img)
{

    std::string rot_center= RegistrationSettings::get().getValue<std::string>(std::string("rot_eddy_center"));

    /*********************************************************************************
     We are doing this for several reasons.
     1:  We have to operate on Read/Phase/Slice coordinate system, not x/y/z like ITK.
         So we set the image direction to identity.
         Yes this causes an inconsistency between images and transformations so
         we have to be really careful everytime we register/transform an image.
     2:  Eddy currents do not affect the scanner isocenter (besides a translation which is accounted for in motion correction).
         If the image header is correct and the image coordinate (0,0,0) is indeed the scanner isocenter
         we should use that one.
         But if the header is wrong, we can use the closest thing which is the center voxel of the image.
    **************************************************************************************/

    // do not want to touch the original image so we duplicate it
    using DupType= itk::ImageDuplicator<ImageType>;
    typename DupType::Pointer dup= DupType::New();
    dup->SetInputImage(img);
    dup->Update();
    typename ImageType::Pointer nimg= dup->GetOutput();

    typename ImageType::DirectionType id_dir;     id_dir.SetIdentity();
    nimg->SetDirection(id_dir);
    typename ImageType::PointType new_orig;

    if(rot_center=="isocenter")
    {
        // The center is the isocenter.
        // If we werent changing the Direction matrix, we would not have to do ANYTHING here.
        // But we are, so keep  the same location as the (0,0,0) coordinate with the new Id direction matrix.

        vnl_matrix<double> Sinv(3,3,0);
        Sinv(0,0)= 1./img->GetSpacing()[0];
        Sinv(1,1)= 1./img->GetSpacing()[1];
        Sinv(2,2)= 1./img->GetSpacing()[2];
        vnl_matrix<double> S(3,3,0);
        S(0,0)= img->GetSpacing()[0];
        S(1,1)= img->GetSpacing()[1];
        S(2,2)= img->GetSpacing()[2];

        vnl_vector<double> indo= Sinv*img->GetDirection().GetTranspose() * (-1.*img->GetOrigin().GetVnlVector());   //this is the continuous index (i,j,k) of the isocenter
        vnl_vector<double> new_orig_v= -S*indo;
        new_orig[0]=new_orig_v[0];
        new_orig[1]=new_orig_v[1];
        new_orig[2]=new_orig_v[2];
    }
    else
    {
        //Make the rotation and eddy center the image center voxel.
        new_orig[0]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[0])-1)/2. * img->GetSpacing()[0];
        new_orig[1]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[1])-1)/2. * img->GetSpacing()[1];
        new_orig[2]=   - ((int)(img->GetLargestPossibleRegion().GetSize()[2])-1)/2. * img->GetSpacing()[2];

    }
    nimg->SetOrigin(new_orig);

    return nimg;
}




template <typename ImageType>
typename ImageType::Pointer DIFFPREP::QuadratictransformImage(typename ImageType::Pointer img,CompositeTransformType::Pointer trans,std::string interp_method,float default_val)
{
    typename ImageType::Pointer nimg= ChangeImageHeaderToDP<ImageType>(img);

    int NITK= TORTOISE::GetAvailableITKThreadFor();

    using ResampleImageFilterType= itk::ResampleImageFilter<ImageType, ImageType> ;
    typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetOutputParametersFromImage(nimg);
    resampleFilter->SetInput(nimg);
    resampleFilter->SetTransform(trans);
    resampleFilter->SetDefaultPixelValue(default_val);
    resampleFilter->SetNumberOfWorkUnits(NITK);
    resampleFilter->Update();    
    typename ImageType::Pointer resampled_img=resampleFilter->GetOutput();

    resampled_img->SetOrigin(img->GetOrigin());
    resampled_img->SetDirection(img->GetDirection());

    return resampled_img;
}


ImageType3D::Pointer DIFFPREP::dmc_make_target(ImageType3D::Pointer img,ImageType3D::Pointer  mask)
{
    double fct=0.9;

    std::vector<float> new_res,factors;
    new_res.resize(3);
    new_res[0]= img->GetSpacing()[0]/fct;
    new_res[1]= img->GetSpacing()[1]/fct;
    new_res[2]= img->GetSpacing()[2]/fct;


    ImageType3D::Pointer transformed_img= resample_3D_image(img, new_res,factors, "BSPCubic");
    ImageType3D::Pointer transformed_mask= resample_3D_image(mask, new_res,factors, "NN");


    int minx= transformed_mask->GetLargestPossibleRegion().GetSize()[0]+5;
    int maxx=-1;
    int miny= transformed_mask->GetLargestPossibleRegion().GetSize()[1]+5;
    int maxy=-1;
    int minz= transformed_mask->GetLargestPossibleRegion().GetSize()[2]+5;
    int maxz=-1;


    itk::ImageRegionIteratorWithIndex<ImageType3D>  it(transformed_mask,transformed_mask->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType index=it.GetIndex();
        float val= it.Get();
        if(val>0.5)
        {
            if(index[0]<minx)
                minx=index[0];
            if(index[1]<miny)
                miny=index[1];
            if(index[2]<minz)
                minz=index[2];

            if(index[0]>maxx)
                maxx=index[0];
            if(index[1]>maxy)
                maxy=index[1];
            if(index[2]>maxz)
                maxz=index[2];
        }
        ++it;
    }

    if(minx==transformed_mask->GetLargestPossibleRegion().GetSize()[0]+5)
        minx=0;
    if(miny==transformed_mask->GetLargestPossibleRegion().GetSize()[1]+5)
        miny=0;
    if(minz==transformed_mask->GetLargestPossibleRegion().GetSize()[2]+5)
        minz=0;
    if(maxx==-1)
        maxx=(int)(transformed_mask->GetLargestPossibleRegion().GetSize()[0])-1;
    if(maxy==-1)
        maxy=(int)(transformed_mask->GetLargestPossibleRegion().GetSize()[1])-1;
    if(maxz==-1)
        maxz=(int)(transformed_mask->GetLargestPossibleRegion().GetSize()[2])-1;


    minx= std::max(minx-2,0);
    maxx= std::min(maxx+2, (int)(transformed_mask->GetLargestPossibleRegion().GetSize()[0])-1);
    miny= std::max(miny-2,0);
    maxy= std::min(maxy+2, (int)(transformed_mask->GetLargestPossibleRegion().GetSize()[1])-1);
    minz= std::max(minz-2,0);
    maxz= std::min(maxz+2, (int)(transformed_mask->GetLargestPossibleRegion().GetSize()[2])-1);


    ImageType3D::IndexType start;
    ImageType3D::SizeType sz;

    start[0]=minx;
    start[1]=miny;
    start[2]=minz;
    sz[0]= (maxx-minx+1);
    sz[1]= (maxy-miny+1);
    sz[2]= (maxz-minz+1);

    if(sz[0]+sz[1]+sz[2] < 0.3*(transformed_mask->GetLargestPossibleRegion().GetSize()[0]+transformed_mask->GetLargestPossibleRegion().GetSize()[1]+transformed_mask->GetLargestPossibleRegion().GetSize()[2]))
    {
        start.Fill(0);
        sz= transformed_mask->GetLargestPossibleRegion().GetSize();
    }
    ImageType3D::RegionType reg(start,sz);


    typedef itk::RegionOfInterestImageFilter<ImageType3D,ImageType3D> ExtractorType;
    ExtractorType::Pointer ext= ExtractorType::New();
    ext->SetInput(transformed_img);
    ext->SetRegionOfInterest(reg);
    ext->Update();
    ImageType3D::Pointer final_img= ext->GetOutput();


    long msk_cnt=0;
    itk::ImageRegionIteratorWithIndex<ImageType3D>  itm(mask,mask->GetLargestPossibleRegion());
    itm.GoToBegin();
    while(!itm.IsAtEnd())
    {
        if(itm.Get()!=0)
            msk_cnt++;
        ++itm;
    }

    long npixels= (long)(mask->GetLargestPossibleRegion().GetSize()[0])*(long)(mask->GetLargestPossibleRegion().GetSize()[1])*(long)(mask->GetLargestPossibleRegion().GetSize()[2]);
    if(msk_cnt<0.02*npixels)
    {
        itm.GoToBegin();
        while(!itm.IsAtEnd())
        {
            itm.Set(1);
            ++itm;
        }
    }


    return final_img;
}




std::vector<float> DIFFPREP::choose_range(ImageType3D::Pointer b0_img,ImageType3D::Pointer curr_vol, ImageType3D::Pointer b0_mask_img)
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

void DIFFPREP::SynthMotionEddyCorrectAllDWIs(std::vector<ImageType3D::Pointer> target_imgs, std::vector<ImageType3D::Pointer> source_imgs)
{
    std::string correction_mode= RegistrationSettings::get().getValue<std::string>(std::string("correction_mode"));


    ImageType3D::Pointer mask2=ChangeImageHeaderToDP<ImageType3D>(this->b0_mask_img);
    ImageType3D::Pointer target_zero=ChangeImageHeaderToDP<ImageType3D>(target_imgs[0]);

     (*stream)<<"Done registering vol: " <<std::flush;

#ifdef USECUDA
    #pragma omp parallel for schedule(dynamic)
#else
    #pragma omp parallel for
#endif
    for( int vol=0; vol<Nvols;vol++)
    {
        TORTOISE::EnableOMPThread();

        if(vol == this->b0_vol_id || correction_mode=="off" )
        {
            itk::OkanQuadraticTransform<double>::Pointer id_trans=itk::OkanQuadraticTransform<double>::New();
            id_trans->SetPhase(this->PE_string);
            id_trans->SetIdentity();
            dwi_transforms[vol]->ClearTransformQueue();
            dwi_transforms[vol]->AddTransform(id_trans);
        }
        else
        {
            ImageType3D::Pointer target= target_imgs[vol];
            ImageType3D::Pointer target2=ChangeImageHeaderToDP<ImageType3D>(target);
            ImageType3D::Pointer target_target=dmc_make_target(target2,mask2);

            ImageType3D::Pointer curr_vol = ChangeImageHeaderToDP<ImageType3D>(source_imgs[vol]);
            std::vector<float> signal_ranges = choose_range(target, curr_vol,b0_mask_img);


            OkanQuadraticTransformType::Pointer curr_trans=nullptr;
            #ifdef USECUDA
                if(TORTOISE::ReserveGPU())
                {
                    curr_trans=  RegisterDWIToB0_cuda(target_target, curr_vol, this->PE_string, this->mecc_settings,true,signal_ranges);
                    TORTOISE::ReleaseGPU();
                }
                else
                    curr_trans=  RegisterDWIToB0(target_target, curr_vol, this->PE_string, this->mecc_settings,true,signal_ranges);
            #else
                    curr_trans=  RegisterDWIToB0(target_target, curr_vol, this->PE_string, this->mecc_settings,true,signal_ranges);
            #endif

            dwi_transforms[vol]->ClearTransformQueue();
            dwi_transforms[vol]->AddTransform(curr_trans);
        }
        #pragma omp critical
        {            
            (*stream)<<", "<<vol<<std::flush;
        }
        TORTOISE::DisableOMPThread();
    }
    (*stream)<<std::endl<<std::endl;
}


void DIFFPREP::ClassicMotionEddyCorrectAllDWIs(ImageType3D::Pointer target, std::vector<ImageType3D::Pointer> dwis)
{
    std::vector<ImageType3D::Pointer> target_imgs;
    target_imgs.resize(dwis.size());
    for(int v=0;v<dwis.size();v++)
        target_imgs[v]=target;

    SynthMotionEddyCorrectAllDWIs(target_imgs,dwis);
}


void DIFFPREP::EM(std::vector< std::vector<float> >  logRMS_shell, std::vector< std::vector<float> > &per_shell_inliers, std::vector< std::vector<float> > &per_shell_outliers, std::vector<float> &Pin_per_shell,std::vector<float> &medians,std::vector<float> &MADs )
{
    int Nshells= logRMS_shell.size();

  //  #pragma omp parallel for
    for(int sh=0; sh<Nshells;sh++ )
    {
        per_shell_inliers[sh].resize(2);
        per_shell_outliers[sh].resize(2);


        Eigen::VectorXf res (logRMS_shell[sh].size());
        for(int i=0;i<logRMS_shell[sh].size();i++)
        {
            res[i]=logRMS_shell[sh][i];
            if(logRMS_shell[sh][i] > 1E-10)
                logRMS_shell[sh][i]=log(logRMS_shell[sh][i]);
            else
                logRMS_shell[sh][i]=-10;
        }



        float nzmin = res.redux([](float a, float b) {
          if (a > 0) return (b > 0) ? std::min(a,b) : a;
          else return (b > 0) ? b : std::numeric_limits<float>::infinity();
        });
        Eigen::VectorXf logres = res.array().max(nzmin).log();


        const float reg = 1e-6;
        const float tol = 1e-6;
        int niter=500;


        double prev_LL= -std::numeric_limits<float>::infinity();

        float shell_median= median(logRMS_shell[sh]);
        medians[sh]=shell_median;

        std::vector<float> temp;
        for(int i=0; i<logRMS_shell[sh].size();i++ )
        {
            temp.push_back(fabs(logres[i] - shell_median));
        }
        float MAD= median(temp);
        float MAD_mean= std::accumulate(temp.begin(), temp.end(), 0.0)/temp.size();
        float sigma= 1.4826018 *MAD;
       // float sigma= 1.4826018 *MAD;
        MADs[sh]=sigma;




        // Ideally, we should have one or two clusters: inliers vs inliers and outliers
        // if RMS is log normally distributed
        // with most data that is the case
        // however, some data for some reason do not follow this pattern
        // for such data the outlier detection can make things worse
        // so instead of just 2 clusters, we use K=4 clusters for EM
        // then keep merging them until we have covered at least 80% of data points
        // which would 99% of the time give us 2 clusters
        // this approach is more robust than just starting with K=2 clusters for EM


        int Km = RegistrationSettings::get().getValue<int>("outlier_EM_clusters");
        std::vector<double> M,S,P;        
        //for(int K=1;K<=4;K++)
        {

            M.resize(Km);
            S.resize(Km);
            P.resize(Km);

            int mid_id = (Km-1)/2;
            for(int c=0;c<mid_id;c++)
            {
                M[c]= shell_median - (mid_id-c);
                S[c]=sigma - 0.05*(mid_id-c);
                P[c]= 0.1/ (Km-1);
            }
            M[mid_id]=shell_median;
            S[mid_id]=sigma;
            P[mid_id]=0.9 + 0.1*(Km==1);
            for(int c=mid_id+1;c<Km;c++)
            {
                M[c]= shell_median + (c-mid_id);
                S[c]=sigma + 0.05*(c-mid_id);
                P[c]= 0.1/ (Km-1);
            }

            float_t ll, ll0 = -std::numeric_limits<float>::infinity();

            for (int n = 0; n < niter; n++)
            {
                EigenVecType tt= EigenVecType::Constant(logres.size(),0);

                std::vector<EigenVecType> R; R.resize(Km);
                std::vector<EigenVecType> w; w.resize(Km);
                for(int k=0;k<Km;k++)
                {
                    R[k]= log_gaussian(logres,  M[k], S[k]);
                    if(P[k] <=0)
                        P[k]=1E-10;
                    R[k] = R[k].array() + std::log(P[k]);

                    tt= tt.array() + R[k].array().exp();
                }

                float nzmin2 = tt.redux([](float a, float b) {
                  if (a > 0) return (b > 0) ? std::min(a,b) : a;
                  else return (b > 0) ? b : std::numeric_limits<float>::infinity();
                });
                tt = tt.array().max(nzmin2);


                EigenVecType log_prob_norm = Eigen::log(tt.array());
                for(int k=0;k<Km;k++)
                {
                    R[k]-=log_prob_norm;
                    w[k]= R[k].array().exp()+ std::numeric_limits<float>::epsilon();
                    P[k]= w[k].mean();
                    M[k]= average(logres, w[k]);
                    S[k] = std::sqrt(average((logres.array() - M[k]).square(), w[k]) + reg);
                }
                ll= log_prob_norm.mean();

              if (std::fabs(ll - ll0) < tol)
                  break;
              ll0 = ll;
            } //for n

        } //for K


        (*stream)<<std::endl<<"Shell "<<sh<<" residuals: "<< " median: " << shell_median <<" MAD: " << MAD << std::endl<<"Clusters:"<<std::endl;
        for(int kk=0;kk<Km;kk++)
        {
            (*stream)<<"mu: " << M[kk] << " std: " << S[kk] << " P: " <<P[kk]<<std::endl;
        }



        //This is where we decide which clusters are for inlinears which are for outliers
        double tot_P=0;
        int k=0;
        while(tot_P<0.75)
        {
            tot_P+=P[k];
            k++;
        }

        if(k==Km)
        {
            per_shell_inliers[sh][0]= shell_median;
            per_shell_inliers[sh][1]= sigma ;
            per_shell_outliers[sh][0]= shell_median+2;
            per_shell_outliers[sh][1]= sigma +0.2;
            Pin_per_shell[sh]=1;
        }
        else
        {
            /*
            per_shell_inliers[sh][0]= M[k-1];
            per_shell_inliers[sh][1]= S[k-1] ;
            */

            // I admit, I am doing this in an INCREDIBLY ineffcieint and LAZY way.
            //Sorry.
            std::vector<double> dummy_data;
            for(int kk=0;kk<k;kk++)
            {
                std::random_device rd;
                std::mt19937 e2(rd());
                std::normal_distribution<> dist(M[kk],S[kk]);

                int N= std::round(100000*P[kk]);
                for(int n=0;n<N;n++)
                {
                    double vl = dist(e2);
                    dummy_data.push_back(vl);
                }
            }
            double sum = std::accumulate(dummy_data.begin(), dummy_data.end(), 0.0);
            double mean = sum / dummy_data.size();
            double sq_sum = std::inner_product(dummy_data.begin(), dummy_data.end(), dummy_data.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / dummy_data.size() - mean * mean);

            Pin_per_shell[sh]=tot_P;
            per_shell_inliers[sh][0]= mean;
            per_shell_inliers[sh][1]= stdev;
            per_shell_outliers[sh][0]= M[k];
            per_shell_outliers[sh][1]= S[k];
        }

    } //for sh
}




std::vector<ImageType3D::Pointer> DIFFPREP::ReplaceOutliers( std::vector<ImageType3D::Pointer> native_native_synth_dwis,  std::vector<ImageType3D::Pointer> raw_dwis,std::vector<int> shells,vnl_vector<double> bvals)
{

    int aggressive_level = RegistrationSettings::get().getValue<int>("outlier_replacement_mode");

    std::vector<int> shell_ids;     shell_ids.resize(Nvols);

    std::vector< std::vector<float> >  logRMS_shell;
    logRMS_shell.resize(shells.size());

    //The first few and last few slices are not touched


    ImageType3D::SizeType sz =native_native_synth_dwis[0]->GetLargestPossibleRegion().GetSize();

    //RMS holder per volume per slice
    std::vector< std::vector< double> > all_RMS;
    all_RMS.resize(Nvols);

    std::vector< ImageType3D::Pointer> weight_img;
    weight_img.resize(Nvols);

    std::vector<int> volumes_per_shell;  volumes_per_shell.resize(shells.size(),0);
    for(int vol=0;vol<Nvols; vol++)
    {
        for(int s=0;s<shells.size();s++)
        {
            if(fabs(bvals[vol] -shells[s])<10)
            {
                volumes_per_shell[s]++;
                break;
            }
        }
    }



    #pragma omp parallel for
    for(int vol=0;vol<Nvols; vol++)
    {
        TORTOISE::EnableOMPThread();
        all_RMS[vol].resize(sz[2],0.);

        //Outliers are computed shell by shell so
        //Get shell id of the volume.
        int shell_id=-1;
        for(int s=0;s<shells.size();s++)
        {
            if(fabs(bvals[vol] -shells[s])<10)
            {
                shell_id=s;
                shell_ids[vol]=shell_id;
                break;
            }
        }


        using DupType =itk::ImageDuplicator<ImageType3D>;
        DupType::Pointer dup= DupType::New();
        dup->SetInputImage(native_native_synth_dwis[vol]);
        dup->Update();
        ImageType3D::Pointer resid_img=dup->GetOutput();
        resid_img->FillBuffer(0);



        // Compute the residuals between the estimated and raw signals
        // and put them into an image
        itk::ImageRegionIteratorWithIndex<ImageType3D> it(resid_img,resid_img->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            ImageType3D::IndexType ind3= it.GetIndex();
            if(b0_mask_img->GetPixel(ind3)>0)
            {
                float resid= native_native_synth_dwis[vol]->GetPixel(ind3) - raw_dwis[vol]->GetPixel(ind3) ;
                it.Set(resid);
            }
            ++it;
        }

        for(int k=SLICE_NOTCONSIDER;k<sz[2]-SLICE_NOTCONSIDER;k++)
        {
           double sm=0;
           long Npix=0;

           //Compute slice RMS for the current slice
           ImageType3D::IndexType ind3;
           ind3[2]=k;
           for(int j=0;j<sz[1];j++)
           {
               ind3[1]=j;
               for(int i=0;i<sz[0];i++)
               {
                   ind3[0]=i;
                   if(b0_mask_img->GetPixel(ind3))
                   {
                       Npix++;
                       double resid= resid_img->GetPixel(ind3);
                       sm+=resid*resid;
                   }
               }
           }
           float val =0;
           if(Npix)
               val=sqrt(sm/Npix);

           all_RMS[vol][k]=val;

           #pragma omp critical
           {
               // put the RMS value into the the corresponding shell's RMS array
               // we put the Actual RMS in.  the log of it will be taken in the function
               if(val!=0)
                   logRMS_shell[shell_id].push_back(val);
           }
       }
        TORTOISE::DisableOMPThread();
    } //for vol

    std::vector< std::vector<float> > per_shell_inliers;
    std::vector< std::vector<float> > per_shell_outliers;
    std::vector<float> Pin_per_shell;
    std::vector<float> medians, MADs;
    per_shell_inliers.resize(shells.size());
    per_shell_outliers.resize(shells.size());
    Pin_per_shell.resize(shells.size());
    medians.resize(shells.size());
    MADs.resize(shells.size());


    // Cluster the log RMS into two clusters with Expectation-Maximization
    // The lower RMS one will be inlier, the higher one will be outlier classes
    EM(logRMS_shell, per_shell_inliers, per_shell_outliers, Pin_per_shell,medians,MADs );


    vnl_matrix<double> meds_sl(shells.size(),sz[2],0),  MADs_sl(shells.size(),sz[2],0)  ;
    for(int sh=0;sh<shells.size();sh++)
    {
        for(int k=SLICE_NOTCONSIDER;k<sz[2]-SLICE_NOTCONSIDER;k++)
        {
            meds_sl(sh,k)= per_shell_inliers[sh][0];
            MADs_sl(sh,k)= per_shell_inliers[sh][1];
        }
    }


    float THR = RegistrationSettings::get().getValue<float>("outlier_prob");
    //Check for each shell and for each slice, if we have at least one inlier
    vnl_matrix<int> covereds(shells.size(),sz[2],0);
    for(int vol=0;vol<Nvols; vol++)
    {
        int si= shell_ids[vol];
        for(int k=SLICE_NOTCONSIDER;k<sz[2]-SLICE_NOTCONSIDER;k++)
        {
            double val = all_RMS[vol][k];
            if(val>0)
                val=log(val);
            else
                val=0;            
            double valCDF = ComputeResidProb(val,meds_sl(shell_ids[vol],k), MADs_sl(shell_ids[vol],k),aggressive_level);
            if(valCDF>THR)
                covereds(si,k)++;
        }
    }

    // each shell has to have 70% good data for a given slice
    // otherwise fitting synth goes crazy
    for(int sh=0;sh<shells.size();sh++)
    {
        for(int k=SLICE_NOTCONSIDER;k<sz[2]-SLICE_NOTCONSIDER;k++)
        {
            if(covereds(sh,k) < 0.7*volumes_per_shell[sh] )
            {
                if(per_shell_inliers[sh][0] + 3* per_shell_inliers[sh][1]< medians[sh]+3*MADs[sh])
                {
                    meds_sl(sh,k)= medians[sh];
                    MADs_sl(sh,k)= MADs[sh];
                }

                // first try with median/MAD
                covereds(sh,k)=0;
                for(int vol=0;vol<Nvols; vol++)
                {
                    if(shell_ids[vol]==sh)
                    {
                        double val = all_RMS[vol][k];
                        if(val>0)
                            val=log(val);
                        else
                            val=0;
                        double valCDF = ComputeResidProb(val,meds_sl(sh,k), MADs_sl(sh,k),aggressive_level);
                        if(valCDF>THR)
                            covereds(sh,k)++;
                    }
                }
                if(covereds(sh,k) < 0.7*volumes_per_shell[sh] )
                {
                    // median/MAD didnt work
                    // so we will increase median/MAD till we have 50% of data
                    bool enter=true;
                    meds_sl(sh,k)=per_shell_inliers[sh][0];
                    MADs_sl(sh,k)=per_shell_inliers[sh][1];
                    while(enter)
                    {
                        covereds(sh,k)=0;
                        meds_sl(sh,k)*=1.1;
                        MADs_sl(sh,k)*=1.05;
                        for(int vol=0;vol<Nvols; vol++)
                        {
                            if(shell_ids[vol]==sh)
                            {
                                double val = all_RMS[vol][k];
                                if(val>0)
                                    val=log(val);
                                else
                                    val=0;
                                double valCDF = ComputeResidProb(val,meds_sl(sh,k), MADs_sl(sh,k),aggressive_level);
                                if(valCDF>THR)
                                    covereds(sh,k)++;
                            }
                        }
                        if(covereds(sh,k) >= 0.7*volumes_per_shell[sh] )
                            enter=false;
                    }
                }
            }
        }
    }





    // Saving residuals to files for reporting
    std::string nname= fs::path(this->nii_name).filename().string();
    fs::path proc_path2=fs::path(this->nii_name).parent_path();
    std::string basename = nname.substr(0, nname.find(".nii"));
    fs::path shell_resid_dist_path = proc_path2 / (basename + std::string("_shell_resid_dist.txt"));
    {
        FILE *fp=fopen(shell_resid_dist_path.string().c_str(),"w");
        for(int sh=0;sh<per_shell_inliers.size();sh++)
        {
            fprintf(fp,"%f %f %f %f %f\n",per_shell_inliers[sh][0],per_shell_inliers[sh][1],per_shell_outliers[sh][0],per_shell_outliers[sh][1],Pin_per_shell[sh]);
        }
        fclose(fp);
    }
    fs::path slice_resids_path = proc_path2 / (basename + std::string("_slice_resids.txt"));
    {
        FILE *fp=fopen(slice_resids_path.string().c_str(),"w");
        for(int vol=0;vol<Nvols;vol++)
        {
            for(int k=0;k<sz[2];k++)
            {
                double val = all_RMS[vol][k];
                if(val>0)
                    val=log(val);
                else
                    val=0;
                fprintf(fp,"%f ",val);
            }
            fprintf(fp,"\n");
        }
        fclose(fp);
    }

    #pragma omp parallel for
    for(int vol=0;vol<Nvols;vol++)
    {
        TORTOISE::EnableOMPThread();

        weight_img[vol]= ImageType3D::New();
        weight_img[vol]->SetRegions(raw_dwis[0]->GetLargestPossibleRegion());
        weight_img[vol]->Allocate();
        weight_img[vol]->SetSpacing(raw_dwis[0]->GetSpacing());
        weight_img[vol]->SetOrigin(raw_dwis[0]->GetOrigin());
        weight_img[vol]->SetDirection(raw_dwis[0]->GetDirection());
        weight_img[vol]->FillBuffer(1);


        for(int k=SLICE_NOTCONSIDER;k<sz[2]-SLICE_NOTCONSIDER;k++)
        {
            ImageType3D::IndexType ind3;
            ind3[2]=k;

            double val = log(all_RMS[vol][k]);
            double valCDF = ComputeResidProb(val,meds_sl(shell_ids[vol],k), MADs_sl(shell_ids[vol],k),aggressive_level);

            for(int j=0;j<sz[1];j++)
            {
                ind3[1]=j;
                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    weight_img[vol]->SetPixel(ind3, valCDF);
                }
            }
        } //for k

        TORTOISE::DisableOMPThread();
    }//for vol


    return weight_img;
}




void DIFFPREP::MotionAndEddy()
{
    vnl_vector<double> bvals = Bmatrix.get_column(0) + Bmatrix.get_column(3)+ Bmatrix.get_column(5);
    double max_bval= bvals.max_value();

    float dti_bval_cutoff= RegistrationSettings::get().getValue<float>(std::string("dti_bval"));
    float mapmri_bval_cutoff= RegistrationSettings::get().getValue<float>(std::string("hardi_bval"));
    bool high_bval_present= max_bval > dti_bval_cutoff*1.2;
    bool is_human_brain = RegistrationSettings::get().getValue<bool>(std::string("is_human_brain"));
    bool slice_to_volume=  RegistrationSettings::get().getValue<bool>(std::string("s2v"));
    bool outlier_replacement= RegistrationSettings::get().getValue<bool>(std::string("repol"));
    int Nepoch=RegistrationSettings::get().getValue<int>(std::string("niter"));
    std::string correction_mode= RegistrationSettings::get().getValue<std::string>(std::string("correction_mode"));

    bool iterative=false;
    if( (is_human_brain && high_bval_present) || slice_to_volume || outlier_replacement)
        iterative=true;

    if(correction_mode=="off" && !slice_to_volume)
        iterative=false;

    if(iterative && ! slice_to_volume && !outlier_replacement)
        Nepoch=1;

    if(Nvols < 20 )
    {
        (*stream)<<"Not enough volumes in the data to perform s2v and repol. Disabling these options."<<std::endl;
        iterative=false;
        slice_to_volume=false;
        outlier_replacement=false;
    }    

    //READ all the volumes as 3D images into a vector
    std::vector<ImageType3D::Pointer> native_native_raw_dwis;
    for( int vol=0; vol<Nvols;vol++)
    {
        ImageType3D::Pointer img=read_3D_volume_from_4D(this->nii_name,vol);
        native_native_raw_dwis.push_back(img);
    }



    vnl_matrix<int> slspec;
    if(slice_to_volume)
    {
        if(this->my_json["SliceTiming"]!=json::value_t::null)
        {
            slspec= ParseJSONForSliceTiming(this->my_json);
        }
        else
        {
            int Nslice= native_native_raw_dwis[0]->GetLargestPossibleRegion().GetSize()[2];
            if(this->my_json["MultibandAccelerationFactor"]!=json::value_t::null)
            {
                int MB= this->my_json["MultibandAccelerationFactor"];
                int Nexc= Nslice/MB;
                slspec.set_size(Nexc,MB);
                slspec.fill(0);
                for(int k=0;k<Nslice;k++)
                {
                    int r= k % Nexc;
                    int c= k/ Nexc;
                    slspec(r,c)=k;
                }
            }
            else
            {
                slspec.set_size(Nslice,1);
                for(int k=0;k<Nslice;k++)
                    slspec(k,0)=k;
            }

        }

        s2v_transformations.resize(Nvols);
    }




    ///////////////////////////// FIRST PASS ///////////////////////////////////////////
     if(iterative)
         (*stream)<<"Performing FIRST PASS of iterative correction..."<<std::endl<<std::endl;

    {   //Regardless of being iterative or not, register all DWIs to the b=0 image first

        (*stream)<<"Registering DWIs..."<<std::endl;

        ImageType3D::Pointer b0_img = native_native_raw_dwis[this->b0_vol_id];
        ClassicMotionEddyCorrectAllDWIs(b0_img,native_native_raw_dwis);





    } // IF NOT ITERATIVE WE ARE DONE



     if(iterative)         //OHHH THIS IS GONNA BE A BIG ONE :)
     {
         const unsigned int  CORRECTION_STAGE_MAPMRI_DEGREE=4;

         //Get small delta and big delta from either input or estimate them
         float big_delta=0, small_delta=0;
         GetSmallBigDelta(small_delta,big_delta);

         // Shells and indices
         std::vector<int> shells;
         for(int v=0;v<Nvols;v++)
         {
             float curr_b= bvals[v];
             bool found=false;
             for(int v2=0;v2<shells.size();v2++)
             {
                 if(fabs(shells[v2] -curr_b)<50)
                 {
                     found=true;
                     break;
                 }
             }
             if(!found)
                 shells.push_back(round50(curr_b));
         }
         std::vector<int> low_b_indices,  not_low_b_indices,  low_DT_indices, MAPMRI_indices;
         for(int v=0;v<Nvols;v++)
         {
             if(bvals[v] <= dti_bval_cutoff)
                 low_b_indices.push_back(v);
             else
                 not_low_b_indices.push_back(v);

             if( bvals[v] <=mapmri_bval_cutoff)
                 low_DT_indices.push_back(v);
             else
                 MAPMRI_indices.push_back(v);
         }


         // ALL THE IMAGES THAT WILL BE USED
         std::vector<ImageType3D::Pointer> s2v_replaced_raw_dwis;
         std::vector<ImageType3D::Pointer> eddy_s2v_replaced_raw_dwis;         
         std::vector<ImageType3D::Pointer> s2v_replaced_synth_dwis;
         //std::vector<ImageType3D::Pointer> native_native_synth_dwis;
         std::vector<ImageType3D::Pointer> eddy_weight_img;



         s2v_replaced_raw_dwis.resize(Nvols);
         eddy_s2v_replaced_raw_dwis.resize(Nvols);
         this->eddy_s2v_replaced_synth_dwis.resize(Nvols);
         s2v_replaced_synth_dwis.resize(Nvols);
         this->native_native_synth_dwis.resize(Nvols);


         if(outlier_replacement)
         {
             this->native_weight_img.resize(Nvols);
             eddy_weight_img.resize(Nvols);

             //Create the native space voxelwise  inclusion images.
             //  Voxel value=1, means the voxel is good
             //  Voxel value=0, means the voxel is artifactual
             for(int vol=0;vol<Nvols;vol++)
             {
                 s2v_replaced_raw_dwis[vol]= native_native_raw_dwis[vol];
                 native_weight_img[vol]= ImageType3D::New();
                 native_weight_img[vol]->SetRegions(native_native_raw_dwis[vol]->GetLargestPossibleRegion());
                 native_weight_img[vol]->Allocate();
                 native_weight_img[vol]->SetSpacing(native_native_raw_dwis[vol]->GetSpacing());
                 native_weight_img[vol]->SetOrigin(native_native_raw_dwis[vol]->GetOrigin());
                 native_weight_img[vol]->SetDirection(native_native_raw_dwis[vol]->GetDirection());
                 native_weight_img[vol]->FillBuffer(1);
             }
         }

         for(int vol=0;vol<Nvols;vol++)
         {
             s2v_replaced_raw_dwis[vol]= native_native_raw_dwis[vol];
         }


         //EPOCHS
         for(int epoch=1;epoch<=Nepoch;epoch++)
         {
             (*stream)<<"EPOCH: " <<epoch<<std::endl;

             // Transform all DWIs with inter-volume motion & eddy-currents transformations
             #pragma omp parallel for
             for(int vol=0;vol<Nvols;vol++)
             {
                 TORTOISE::EnableOMPThread();

                 ImageType3D::Pointer curr_vol= s2v_replaced_raw_dwis[vol];

                 if(outlier_replacement)
                 {
                     ImageType3D::Pointer weight_vol= native_weight_img[vol];
                     eddy_weight_img[vol]= QuadratictransformImage<ImageType3D>(weight_vol,dwi_transforms[vol],"NN",1);
                     native_weight_img[vol]->FillBuffer(1);
                 }

                 eddy_s2v_replaced_raw_dwis[vol] = QuadratictransformImage<ImageType3D>(curr_vol,dwi_transforms[vol],"Linear",0);


                 TORTOISE::DisableOMPThread();
             }
             //All this is quite memory intensive. We should clear as soon as we can
             s2v_replaced_raw_dwis.clear(); s2v_replaced_raw_dwis.resize(Nvols);


             // Fit DTI and MAPMRI to the transformed  data
              // if less than 15 DTI regime volumes, use everything instead for DTI fit

             //DTI fitting                    
             if(low_DT_indices.size()<15)
                 low_DT_indices.resize(0);

             std::vector<std::vector<ImageType3D::Pointer> >dummyv;
             std::vector<int> dummy;
             DTIModel dti_estimator;
             dti_estimator.SetBmatrix(Bmatrix);
             dti_estimator.SetDWIData(eddy_s2v_replaced_raw_dwis);
             dti_estimator.SetWeightImage(eddy_weight_img);
             dti_estimator.SetVoxelwiseBmatrix(dummyv);
             dti_estimator.SetMaskImage(nullptr);
             dti_estimator.SetVolIndicesForFitting(low_DT_indices);
             dti_estimator.SetFittingMode("WLLS");
             dti_estimator.PerformFitting();
	                     

             // MAPMRI FITTING
             MAPMRIModel mapmri_estimator;
             if(MAPMRI_indices.size()>0)
             {
                 mapmri_estimator.SetMAPMRIDegree(CORRECTION_STAGE_MAPMRI_DEGREE);
                 mapmri_estimator.SetDTImg(dti_estimator.GetOutput());
                 mapmri_estimator.SetA0Image(dti_estimator.GetA0Image());
                 mapmri_estimator.SetBmatrix(Bmatrix);
                 mapmri_estimator.SetDWIData(eddy_s2v_replaced_raw_dwis);
                 mapmri_estimator.SetWeightImage(eddy_weight_img);
                 mapmri_estimator.SetVoxelwiseBmatrix(dummyv);
                 mapmri_estimator.SetMaskImage(b0_mask_img);
                 mapmri_estimator.SetVolIndicesForFitting(dummy);
                 mapmri_estimator.SetSmallDelta(small_delta);
                 mapmri_estimator.SetBigDelta(big_delta);
                 mapmri_estimator.PerformFitting();
             }
             //All this is quite memory intensive. We should clear as soon as we can
             eddy_s2v_replaced_raw_dwis.clear(); eddy_s2v_replaced_raw_dwis.resize(Nvols);
             if(outlier_replacement)
             {
                 eddy_weight_img.clear(); eddy_weight_img.resize(Nvols);
             }
             (*stream)<<"Synthesizing volumes... " <<std::endl;            

             //Start Processing each volume
             #pragma omp parallel for
             for(int vol=0;vol<Nvols;vol++)
             {
                 TORTOISE::EnableOMPThread();
                 int NITK= TORTOISE::GetAvailableITKThreadFor();

                 // Synthesize artificial volume with the same bvec/bval
                 ImageType3D::Pointer synth_img=nullptr;

                 /*
                 if( bvals[vol] <=mapmri_bval_cutoff)
                     synth_img= dti_estimator.SynthesizeDWI( Bmatrix.get_row(vol) );
                 else if(bvals[vol] >mapmri_bval_cutoff)
                     synth_img = mapmri_estimator.SynthesizeDWI( Bmatrix.get_row(vol) );
                 */
                 if(MAPMRI_indices.size()>0)
                     synth_img = mapmri_estimator.SynthesizeDWI( Bmatrix.get_row(vol) );
                 else
                     synth_img= dti_estimator.SynthesizeDWI( Bmatrix.get_row(vol) );
                 eddy_s2v_replaced_synth_dwis[vol]=synth_img;

                 // This synthesized image is on the space of the corrected data (for everything).
                 // We have to transform it ALL the way to the very native space,
                 //i.e., no inter-volume motion, no eddy-currents , no slice-to-volume motion
                 // in reverse order.
                 //Some transformations such as eddy's quadratic/cubic can not be analytically inverted
                 // So we have to convert them into a displacement field and invert it there numerically.


                 {  // curly bracket to automatically delete objects (yes me lazy)
                     DisplacementFieldType::Pointer distortion_field=  ConvertEddyTransformToField(dwi_transforms[vol],synth_img);
                     typedef itk::InvertDisplacementFieldImageFilter<DisplacementFieldType> InverterType;
                     InverterType::Pointer inverter = InverterType::New();
                     inverter->SetInput( distortion_field );
                     inverter->SetMaximumNumberOfIterations( 50 );
                     inverter->SetMeanErrorToleranceThreshold( 0.0004 );
                     inverter->SetMaxErrorToleranceThreshold( 0.04 );
                     inverter->SetNumberOfWorkUnits(NITK);
                     inverter->Update();

                     DisplacementFieldType::Pointer distortion_field_inv = inverter->GetOutput();
                     DisplacementFieldTransformType::Pointer disp_trans= DisplacementFieldTransformType::New();
                     disp_trans->SetDisplacementField(distortion_field_inv);

                     using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
                     ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
                     resampleFilter->SetOutputParametersFromImage(synth_img);
                     resampleFilter->SetInput(synth_img);
                     resampleFilter->SetTransform(disp_trans);
                     resampleFilter->SetNumberOfWorkUnits(NITK);
                     resampleFilter->Update();
                     ImageType3D::Pointer native_synth_img= resampleFilter->GetOutput();
                     s2v_replaced_synth_dwis[vol]= native_synth_img;
                     TORTOISE::ReleaseITKThreadFor();
                 }
                 TORTOISE::DisableOMPThread();
             }// for vol


             //slice to volume registration
             if(slice_to_volume)
             {
                 (*stream)<<"Done Slice-to-volume registering volume: " <<std::flush;

                 #ifdef USECUDA
                     #pragma omp parallel for schedule(dynamic)
                 #else
                     #pragma omp parallel for
                 #endif
                 for(int vol=0;vol<Nvols;vol++)
                 {
                     TORTOISE::EnableOMPThread();

                     ImageType3D::Pointer target= native_native_raw_dwis[vol];
                     ImageType3D::Pointer native_synth_img =s2v_replaced_synth_dwis[vol] ;

                     std::vector<float> signal_ranges = choose_range(target, native_synth_img,b0_mask_img);

                     //for the first few epochs we only do rigid registration for sv2
                     //for the last one, we also do quadratic
                     bool do_eddy=true;
                     if(epoch==1)
                         do_eddy=false;

                  //   #ifdef USECUDA
                  //      if(TORTOISE::ReserveGPU())
                   //     {
                   //         VolumeToSliceRegistration_cuda(target, native_synth_img,slspec,signal_ranges,s2v_transformations[vol],do_eddy,this->PE_string);
                   //         TORTOISE::ReleaseGPU();
                   //     }
                   //     else
                            VolumeToSliceRegistration(target, native_synth_img,slspec,signal_ranges,s2v_transformations[vol],do_eddy,this->PE_string);
                 //    #else
                //            VolumeToSliceRegistration(target, native_synth_img,slspec,signal_ranges,s2v_transformations[vol],do_eddy,this->PE_string);
                //     #endif

                     #pragma omp critical
                     {
                         (*stream)<<vol<<", "<<std::flush;
                     }
                     TORTOISE::DisableOMPThread();
                 } //for vol
             } //if s2v


             (*stream)<<std::endl<<std::endl;

             if(outlier_replacement)
             {
                 (*stream)<<"Replacing outliers..."<<std::endl;

                 #pragma omp parallel for
                 for(int vol=0;vol<Nvols;vol++)
                 {
                     TORTOISE::EnableOMPThread();

                     // Create the native_native (i.e. mot&eddy and s2v uncorrected)
                     // space image
                     native_native_synth_dwis[vol] = ImageType3D::New();
                     native_native_synth_dwis[vol]->SetRegions(native_native_raw_dwis[vol]->GetLargestPossibleRegion());
                     native_native_synth_dwis[vol]->Allocate();
                     native_native_synth_dwis[vol]->SetSpacing(native_native_raw_dwis[vol]->GetSpacing());
                     native_native_synth_dwis[vol]->SetOrigin(native_native_raw_dwis[vol]->GetOrigin());
                     native_native_synth_dwis[vol]->SetDirection(native_native_raw_dwis[vol]->GetDirection());

                     ImageType3D::SizeType sz= native_native_synth_dwis[vol]->GetLargestPossibleRegion().GetSize();
                     ImageType3D::Pointer curr_vol = native_native_raw_dwis[vol];

                     typedef itk::LinearInterpolateImageFunction<ImageType3D,double> InterpolatorType;
                     InterpolatorType::Pointer interp = InterpolatorType::New();
                     interp->SetInputImage(s2v_replaced_synth_dwis[vol]);


                     //further transform the mot&eddy uncorrected SYNTH image bach to s2v uncorrected space
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
                                 ImageType3D::PointType pt_trans=pt;
                                 if(slice_to_volume)
                                     pt_trans=s2v_transformations[vol][k]->TransformPoint(pt);

                                 ImageType3D::PixelType interp_val =0;
                                 if(interp->IsInsideBuffer(pt_trans))
                                     interp_val=interp->Evaluate(pt_trans);
                                 native_native_synth_dwis[vol]->SetPixel(ind3,interp_val);
                             }
                         }
                     }
                     TORTOISE::DisableOMPThread();
                 } //for vol


                 // AT this point, we have the synthesized data computed in EVERYTHING corrected space
                 // transformed back to everything UNCORRECTED space.
                 // so,  we can now check for outliers by statistical analysis.

                 this->native_weight_img = ReplaceOutliers(native_native_synth_dwis, native_native_raw_dwis,shells,bvals);
                 (*stream)<<"Replacing outliers...Done!"<<std::endl;
                 //Clean memory
                 //native_native_synth_dwis.clear(); native_native_synth_dwis.resize(Nvols);
                 s2v_replaced_synth_dwis.clear();  s2v_replaced_synth_dwis.resize(Nvols);


                 // Now that we have found the outliers in the completely uncorrected space,
                 // we can transform everything to the corrected space
                 // and recompute everything
                 if(slice_to_volume)
                 {
                     (*stream)<<"Forward transforming slices..."<<std::endl;
                     #pragma omp parallel for
                     for(int vol=0;vol<Nvols;vol++)
                     {
                         //s2v_replaced_raw_dwis[vol]= ForwardTransformImage(native_native_replaced_raw_dwis[vol], s2v_transformations[vol]);
                         s2v_replaced_raw_dwis[vol]= ForwardTransformImage(native_native_raw_dwis[vol], s2v_transformations[vol]);
                     }

                 }
                 else
                 {
                     //if no s2v correction, no need to transform native_native space data
                     // with s2v transformations.
                     //s2v_replaced_raw_dwis=native_native_replaced_raw_dwis;
                     s2v_replaced_raw_dwis=native_native_raw_dwis;
                 }
             }
             else
             {                 
                 //if no repol, no need to deal with replaced data
                 // everything is the raw version
                 if(slice_to_volume)
                 {
                     (*stream)<<"Forward transforming slices..."<<std::endl;
                     #pragma omp parallel for
                     for(int vol=0;vol<Nvols;vol++)
                     {
                         s2v_replaced_raw_dwis[vol]= ForwardTransformImage(native_native_raw_dwis[vol], s2v_transformations[vol]);
                     }
                 }
                 else
                 {
                     s2v_replaced_raw_dwis=native_native_raw_dwis;
                 }
             }


             //We have corrected space synth data now
             // We can use that to improve inter-volume motion and eddy currents distortions.
             SynthMotionEddyCorrectAllDWIs(eddy_s2v_replaced_synth_dwis,s2v_replaced_raw_dwis);


         } //for epoch
     } //if iterative

     //And FINALLY we are done.
}




void DIFFPREP::WriteOutputFiles()
{    
    bool slice_to_volume=  RegistrationSettings::get().getValue<bool>(std::string("s2v"));
    bool outlier_replacement= RegistrationSettings::get().getValue<bool>(std::string("repol"));
    float THR = RegistrationSettings::get().getValue<float>("outlier_prob");

    std::string nname= fs::path(this->nii_name).filename().string();
    fs::path proc_path2=fs::path(this->nii_name).parent_path();
    std::string basename = nname.substr(0, nname.find(".nii"));
    fs::path trans_text_path = proc_path2 / (basename + std::string("_moteddy_transformations.txt"));



    std::ofstream trans_text_file(trans_text_path.string().c_str());
    for( int vol=0; vol<Nvols;vol++)
    {
        if(vol==this->b0_vol_id)
        {
            OkanQuadraticTransformType::Pointer dummy= OkanQuadraticTransformType::New();
            dummy->SetPhase(this->PE_string);
            dummy->SetIdentity();
            trans_text_file<<dummy->GetParameters()<<std::endl;
        }
        else
        {
            trans_text_file<<dwi_transforms[vol]->GetParameters()<<std::endl;
        }
    }
    trans_text_file.close();


    if(this->eddy_s2v_replaced_synth_dwis.size())
    {
        ImageType3D::SizeType sz= this->eddy_s2v_replaced_synth_dwis[0]->GetLargestPossibleRegion().GetSize();

        #pragma omp parallel for
        for(int vol=0;vol<Nvols;vol++)
        {
            TORTOISE::EnableOMPThread();
            int NITK= TORTOISE::GetAvailableITKThreadFor();

            DisplacementFieldType::Pointer distortion_field=  ConvertEddyTransformToField(dwi_transforms[vol],this->eddy_s2v_replaced_synth_dwis[0]);
            typedef itk::InvertDisplacementFieldImageFilter<DisplacementFieldType> InverterType;
            InverterType::Pointer inverter = InverterType::New();
            inverter->SetInput( distortion_field );
            inverter->SetMaximumNumberOfIterations( 50 );
            inverter->SetMeanErrorToleranceThreshold( 0.0004 );
            inverter->SetMaxErrorToleranceThreshold( 0.04 );
            inverter->SetNumberOfWorkUnits(NITK);
            inverter->Update();
            DisplacementFieldType::Pointer distortion_field_inv = inverter->GetOutput();
            DisplacementFieldTransformType::Pointer disp_trans= DisplacementFieldTransformType::New();
            disp_trans->SetDisplacementField(distortion_field_inv);

            typedef itk::BSplineInterpolateImageFunction<ImageType3D,double> InterpolatorType;
            InterpolatorType::Pointer interp = InterpolatorType::New();
            interp->SetInputImage(this->eddy_s2v_replaced_synth_dwis[vol]);
            interp->SetSplineOrder(3);

            CompositeTransformType::Pointer all_trans= CompositeTransformType::New();
            all_trans->AddTransform(disp_trans);

            for(int k=0;k<sz[2];k++)
            {
                ImageType3D::IndexType ind3;
                ind3[2]=k;

                if(slice_to_volume && s2v_transformations.size()>0)
                {
                    ImageType3D::IndexType sl_st;
                    sl_st[2]=k;
                    sl_st[1]=0;
                    sl_st[0]=0;
                    float w= this->native_weight_img[vol]->GetPixel(sl_st);
                    if(w<THR)
                    {
                        s2v_transformations[vol][k]->SetIdentity();
                    }
                }

                for(int j=0;j<sz[1];j++)
                {
                    ind3[1]=j;
                    for(int i=0;i<sz[0];i++)
                    {
                        ind3[0]=i;

                        ImageType3D::PointType pt;
                        this->eddy_s2v_replaced_synth_dwis[0]->TransformIndexToPhysicalPoint(ind3,pt);
                        ImageType3D::PointType pt_trans=all_trans->TransformPoint(pt);
                        if(slice_to_volume && s2v_transformations.size()>0)
                            pt_trans=s2v_transformations[vol][k]->TransformPoint(pt);

                        ImageType3D::PixelType interp_val =0;
                        if(interp->IsInsideBuffer(pt_trans))
                            interp_val=interp->Evaluate(pt_trans);
                        native_native_synth_dwis[vol]->SetPixel(ind3,interp_val);
                    }
                }
            }

            TORTOISE::DisableOMPThread();
        } //for vol
    }

    if(slice_to_volume && s2v_transformations.size()>0)
    {
        fs::path s2v_txt_path = proc_path2 / (basename + std::string("_s2v_transformations.txt"));
        std::ofstream s2v_text_file(s2v_txt_path.string().c_str());
        ImageType3D::SizeType sz= this->b0_mask_img->GetLargestPossibleRegion().GetSize();

        for( int vol=0; vol<Nvols;vol++)
        {
            for(int k=0;k<sz[2];k++)
            {
                s2v_text_file<< s2v_transformations[vol][k] ->GetParameters()<<std::endl;
            }
        }
        s2v_text_file.close();
    }



    (*stream)<<"Writing Motion eddy corrected temp NIFTI file"<<std::endl;
    vnl_matrix<double> rot_Bmatrix;
    std::vector<ImageType3DBool::Pointer> final_inclusion_imgs;
    std::vector<ImageType3D::Pointer> final_data= TransformRepolData(this->nii_name,rot_Bmatrix,final_inclusion_imgs);

    fs::path mot_eddy_nii_path = proc_path2 / (basename + std::string("_moteddy.nii"));
    fs::path mot_eddy_bmtxt_path = proc_path2 / (basename + std::string("_moteddy.bmtxt"));
    fs::path mot_eddy_inc_path = proc_path2 / (basename + std::string("_moteddy_inc.nii"));    
    fs::path native_weight_path = proc_path2 / (basename + std::string("_native_weight.nii"));    
    fs::path native_snyth_nm= proc_path2 / (basename + std::string("_native_native_synth.nii"));

    for(int v=0;v<Nvols;v++)
    {
        write_3D_image_to_4D_file<ImageType3D::PixelType>(final_data[v],mot_eddy_nii_path.string(),v,Nvols);
        if(final_inclusion_imgs.size())
        {
            write_3D_image_to_4D_file<ImageType3DBool::PixelType>(final_inclusion_imgs[v],mot_eddy_inc_path.string(), v,Nvols);
            write_3D_image_to_4D_file<ImageType3D::PixelType>(this->native_weight_img[v],native_weight_path.string(), v,Nvols);            
            write_3D_image_to_4D_file<float>(native_native_synth_dwis[v],native_snyth_nm.string(),v,Nvols);
        }
    }
    std::ofstream outbmtxtfile(mot_eddy_bmtxt_path.string());
    outbmtxtfile<<rot_Bmatrix;
    outbmtxtfile.close();

}


std::vector<ImageType3D::Pointer> DIFFPREP::TransformRepolData(std::string nii_filename, vnl_matrix<double> &rot_Bmatrix, std::vector<ImageType3DBool::Pointer> &final_inclusion_imgs)
{
    std::string correction_mode= RegistrationSettings::get().getValue<std::string>(std::string("correction_mode"));
    float THR = RegistrationSettings::get().getValue<float>("outlier_prob");

    std::vector<ImageType3D::Pointer> raw_data,final_data;
    raw_data.resize(Nvols);
    final_data.resize(Nvols);
    for(int vol=0;vol<Nvols;vol++)
        raw_data[vol]=read_3D_volume_from_4D(nii_filename,vol);


    (*stream)<<"Transforming volumes ..."<<std::endl;

    std::vector<ImageType3DBool::Pointer> native_inclusion_img;
    ImageType3D::SizeType sz =raw_data[0]->GetLargestPossibleRegion().GetSize();


    //First convert the weight images to binary masks
    if(this->native_weight_img.size())
    {
        native_inclusion_img.resize(this->native_weight_img.size());
        final_inclusion_imgs.resize(native_inclusion_img.size());

        ImageType3D::Pointer ref_img = raw_data[0];
        ref_img= ChangeImageHeaderToDP<ImageType3D>(ref_img);

        #pragma omp parallel for
        for(int vol=0;vol<Nvols;vol++)
        {
            TORTOISE::EnableOMPThread();
            int NITK=TORTOISE::GetAvailableITKThreadFor();

            native_inclusion_img[vol] = ImageType3DBool::New();
            native_inclusion_img[vol]->SetRegions( this->native_weight_img[vol]->GetLargestPossibleRegion());
            native_inclusion_img[vol]->Allocate();
            native_inclusion_img[vol]->SetSpacing(this->native_weight_img[vol]->GetSpacing());
            native_inclusion_img[vol]->SetDirection(this->native_weight_img[vol]->GetDirection());
            native_inclusion_img[vol]->SetOrigin(this->native_weight_img[vol]->GetOrigin());
            native_inclusion_img[vol]->FillBuffer(1);

            vnl_vector<float> is_slice_outlier(sz[2],0);

            itk::ImageRegionIteratorWithIndex<ImageType3DBool> it(native_inclusion_img[vol],native_inclusion_img[vol]->GetLargestPossibleRegion());
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                ImageType3D::IndexType ind3= it.GetIndex();
                float prob=this->native_weight_img[vol]->GetPixel(ind3);
                if(prob<THR)
                {
                    it.Set(0);
                    is_slice_outlier[ind3[2]]=1;
                }
            }
            int outlier_slice_count =0;
            for(int k=0;k<sz[2];k++)
                if(is_slice_outlier[k])
                    outlier_slice_count++;

            // if more than 50% slices outliers, this volume is unreliable
            // completely replace its values with predicted signals.
            float outlier_frac = RegistrationSettings::get().getValue<float>("outlier_frac");
            if(1.*outlier_slice_count   > outlier_frac*(sz[2]-2*SLICE_NOTCONSIDER))
            {
                native_inclusion_img[vol]->FillBuffer(0);
                native_weight_img[vol]->FillBuffer(0.);
            }

            if(correction_mode=="off" ||  (1.*outlier_slice_count   > outlier_frac*(sz[2]-2*SLICE_NOTCONSIDER)))
            {
                final_inclusion_imgs[vol]=native_inclusion_img[vol];
            }
            else
            {
                typedef itk::NearestNeighborInterpolateImageFunction<ImageType3DBool,double> NNInterpolatorType;
                using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3DBool, ImageType3DBool> ;

                ImageType3DBool::Pointer inc_vol2= native_inclusion_img[vol];
                ImageType3DBool::Pointer inc_vol=ChangeImageHeaderToDP<ImageType3DBool>(inc_vol2);


                NNInterpolatorType::Pointer nnint= NNInterpolatorType::New();
                nnint->SetInputImage(inc_vol);
                ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
                resampleFilter2->SetOutputParametersFromImage(ref_img);
                resampleFilter2->SetInput(inc_vol);
                resampleFilter2->SetInterpolator(nnint);
                resampleFilter2->SetTransform(this->dwi_transforms[vol]);
                resampleFilter2->SetDefaultPixelValue(1);
                resampleFilter2->SetNumberOfWorkUnits(NITK);
                resampleFilter2->Update();
                final_inclusion_imgs[vol]= resampleFilter2->GetOutput();
                final_inclusion_imgs[vol]->SetDirection(inc_vol2->GetDirection());
                final_inclusion_imgs[vol]->SetOrigin(inc_vol2->GetOrigin());
            }

            TORTOISE::DisableOMPThread();
        } //for vol

        std::string nname= fs::path(this->nii_name).filename().string();
        fs::path proc_path2=fs::path(this->nii_name).parent_path();
        std::string basename = nname.substr(0, nname.find(".nii"));
        fs::path inc_img_path = proc_path2 / (basename + std::string("_native_inclusion.nii"));
        for(int v=0; v<Nvols;v++)
            write_3D_image_to_4D_file<ImageType3DBool::PixelType>(native_inclusion_img[v],inc_img_path.string(), v,Nvols);
    } //if repol


    if(this->s2v_transformations.size()==0)
    {
        if(correction_mode!="off")
        {
            #pragma omp parallel for
            for(int vol=0;vol<Nvols;vol++)
            {
                TORTOISE::EnableOMPThread();
                int NITK=TORTOISE::GetAvailableITKThreadFor();

                ImageType3D::Pointer orig_dwi= raw_data[vol];
                // Replace the outliers in the raw data with estimated values
                // This is only for the temp intermediate file for debugging/reporting
                itk::ImageRegionIteratorWithIndex<ImageType3D> it2(orig_dwi,orig_dwi->GetLargestPossibleRegion());
                for(it2.GoToBegin();!it2.IsAtEnd();++it2)
                {
                    ImageType3D::IndexType ind3= it2.GetIndex();
                    if(native_inclusion_img.size() && native_inclusion_img[vol]->GetPixel(ind3)==0)
                    {
                        if( (this->b0_mask_img && this->b0_mask_img->GetPixel(ind3)) || !this->b0_mask_img)
                            it2.Set(this->native_native_synth_dwis[vol]->GetPixel(ind3));
                    }
                }
                ImageType3D::Pointer orig_dwiDP= ChangeImageHeaderToDP<ImageType3D>(orig_dwi);

                using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
                ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
                resampleFilter->SetOutputParametersFromImage(orig_dwiDP);
                resampleFilter->SetInput(orig_dwiDP);
                resampleFilter->SetTransform(this->dwi_transforms[vol]);
                resampleFilter->SetNumberOfWorkUnits(NITK);
                resampleFilter->Update();
                final_data[vol]= resampleFilter->GetOutput();
                final_data[vol]->SetDirection(orig_dwi->GetDirection());
                final_data[vol]->SetOrigin(orig_dwi->GetOrigin());

                TORTOISE::DisableOMPThread();
            } //for vol
        }
        else
            final_data=raw_data;
    }
    else
    {
        #pragma omp parallel for
        for(int vol=0;vol<Nvols;vol++)
        {
            TORTOISE::EnableOMPThread();
            int NITK=TORTOISE::GetAvailableITKThreadFor();

            ImageType3D::Pointer orig_dwi= raw_data[vol];
            ImageType3D::Pointer orig_dwiDP= ChangeImageHeaderToDP<ImageType3D>(orig_dwi);

            std::vector<float> values;
            using MeasurementVectorType = itk::Vector<float, 3>;
            using SampleType = itk::Statistics::ListSample<MeasurementVectorType>;
            SampleType::Pointer sample = SampleType::New();
            sample->SetMeasurementVectorSize(3);


            ImageType3D::Pointer final_img = ImageType3D::New();
            final_img->SetRegions(raw_data[0]->GetLargestPossibleRegion());
            final_img->Allocate();
            final_img->SetSpacing(orig_dwi->GetSpacing());
            final_img->SetDirection(orig_dwi->GetDirection());
            final_img->SetOrigin(orig_dwi->GetOrigin());
            final_img->FillBuffer(0.);

            ImageType3D::SizeType sz= orig_dwi->GetLargestPossibleRegion().GetSize();
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
                        if(  (native_inclusion_img.size()!=0 && native_inclusion_img[vol]->GetPixel(ind3)) || (native_inclusion_img.size()==0))
                        {
                            ImageType3D::PointType pt,pt_trans;
                            orig_dwi->TransformIndexToPhysicalPoint(ind3,pt);
                            pt_trans=s2v_transformations[vol][k]->TransformPoint(pt);

                            itk::ContinuousIndex<double,3> ind3_t;
                            final_img->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_t);
                            MeasurementVectorType tt;
                            tt[0]=ind3_t[0];
                            tt[1]=ind3_t[1];
                            tt[2]=ind3_t[2];

                            sample->PushBack(tt);
                            values.push_back(orig_dwi->GetPixel(ind3));
                        }
                    }
                }
            } //for k

            if(values.size()>0)
            {
                using TreeGeneratorType = itk::Statistics::KdTreeGenerator<SampleType>;
                TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
                treeGenerator->SetSample(sample);
                treeGenerator->SetBucketSize(16);
                treeGenerator->Update();

                using TreeType = TreeGeneratorType::KdTreeType;
                TreeType::Pointer tree = treeGenerator->GetOutput();

                itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_img,final_img->GetLargestPossibleRegion());
                it.GoToBegin();
                while(!it.IsAtEnd())
                {
                    ImageType3D::IndexType ind3= it.GetIndex();
                    if(  (final_inclusion_imgs.size()!=0 && final_inclusion_imgs[vol]->GetPixel(ind3)) || (final_inclusion_imgs.size()==0))
                    {
                        ImageType3D::PointType pt;
                        orig_dwiDP->TransformIndexToPhysicalPoint(ind3,pt);
                        ImageType3D::PointType pt_trans=this->dwi_transforms[vol]->TransformPoint(pt);

                        itk::ContinuousIndex<double,3> ind3_t;
                        orig_dwiDP->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_t);

                        if(ind3_t[0]<0 || ind3_t[0]>sz[0]-1 || ind3_t[1]<0 || ind3_t[1]>sz[1]-1 || ind3_t[2]<0 || ind3_t[2]>sz[2]-1 )
                        {
                            it.Set(0);
                            ++it;
                            continue;
                        }

                        MeasurementVectorType queryPoint;
                        queryPoint[0]=ind3_t[0];
                        queryPoint[1]=ind3_t[1];
                        queryPoint[2]=ind3_t[2];

                        unsigned int                           numberOfNeighbors = 16;
                        TreeType::InstanceIdentifierVectorType neighbors;
                        std::vector<double> dists;
                        tree->Search(queryPoint, numberOfNeighbors, neighbors,dists);

                        std::vector<double>::iterator mini = std::min_element(dists.begin(), dists.end());
                        double mn= *mini;
                        int mn_id=std::distance(dists.begin(), mini);

                        if(mn<0.1)
                        {
                            float val= values[neighbors[mn_id]];
                            it.Set(val);
                        }
                        else
                        {
                            double sm_weight=0;
                            double sm_val=0;
                            for(int n=0;n<numberOfNeighbors;n++)
                            {
                                int neighbor= neighbors[n];
                                float dist = dists[n];

                                double dist2= 1./pow(dist,3);
                                if(dist2>1E-50)
                                {
                                    sm_val+= values[neighbor] *dist2;
                                    sm_weight+= dist2;
                                }
                            }
                            if(sm_weight==0)
                                it.Set(0);
                            else
                            {
                                it.Set(sm_val/sm_weight);
                            }
                        }
                    }
                    ++it;
                } //while voxel iterator
            }
            final_data[vol]=final_img;

            TORTOISE::DisableOMPThread();
        } //for vol
    } //else s2v


    if(final_inclusion_imgs.size())
    {
        #pragma omp parallel for
        for(int vol=0;vol<Nvols;vol++)
        {
            itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_data[vol],final_data[vol]->GetLargestPossibleRegion());
            it.GoToBegin();
            while(!it.IsAtEnd())
            {
                ImageType3D::IndexType ind3= it.GetIndex();
                if(   final_inclusion_imgs[vol]->GetPixel(ind3)==0 )
                {
                    it.Set(this->eddy_s2v_replaced_synth_dwis[vol]->GetPixel(ind3));
                }
                ++it;
            }
        }
    }


    vnl_matrix_fixed<double,3,3> id_dir;
    id_dir.set_identity();

    rot_Bmatrix= RotateBMatrix(this->Bmatrix,this->dwi_transforms,id_dir);
    return final_data;
}






void DIFFPREP::DPCreateMask()
{
    ImageType3D::Pointer b0_img = read_3D_volume_from_4D(this->nii_name, this->b0_vol_id);
    bool is_human_brain = RegistrationSettings::get().getValue<bool>(std::string("is_human_brain"));

    ImageType3D::Pointer noise_img=nullptr;
    if(!is_human_brain)
        noise_img= readImageD<ImageType3D>(this->nii_name.substr(0,this->nii_name.rfind(".nii"))+std::string("_noise.nii"));

    this->b0_mask_img= create_mask(b0_img,noise_img);
}


void DIFFPREP::PadAndWriteImage()
{
    (*stream)<<"Padding images"<<std::endl;
    ImageType4D::Pointer data= readImageD<ImageType4D>(this->nii_name);
    ImageType4D::Pointer data_padded= PadImage<ImageType4D>(data,2,2,2,2,2,2,true);
    writeImageD<ImageType4D>(data_padded, this->nii_name);
}



void DIFFPREP::SetBoId()
{
    vnl_vector<double> bvals = Bmatrix.get_column(0) + Bmatrix.get_column(3)+ Bmatrix.get_column(5);
    float b0_val= bvals.min_value();
    std::vector<int> b0_ids;
    for(int v=0;v<bvals.size();v++)
    {
        if(fabs(bvals[v]-b0_val)<10)
            b0_ids.push_back(v);
    }


    int b0_choice= RegistrationSettings::get().getValue<int>(std::string("b0_id"));
    if(b0_choice ==-1)
    {
        if(b0_ids.size()<4)
        {
            (*stream)<<"Less than 3 b=0 volumes in the dataset. can not automatically select best one. Using the first one instead."<<std::endl;
            this->b0_vol_id= b0_ids[0];
        }
        else
        {
            (*stream)<<"Selecting the best b=0 image."<<std::endl;
            ImageType4D::Pointer data= readImageD<ImageType4D>(this->nii_name);
            ImageType3D::Pointer dummy;
            this->b0_vol_id= select_best_b0(data,bvals,dummy);
            (*stream)<<"Selected b=0 volume id: "<<this->b0_vol_id<<std::endl;
        }
    }
    else
    {
        if(b0_choice>=Nvols)
            b0_choice= b0_ids[0];

        this->b0_vol_id= b0_choice;
        if( fabs(bvals[this->b0_vol_id]-b0_val)>10 )
        {
            (*stream)<<"THE VOLUME AT \"b0_id\"=" <<this->b0_vol_id<<" DOES NOT SEEM TO BE A b=0 VOLUME. ARE YOU SURE ABOUT THIS? PLEASE CHECK YOUR RESULTS.."<<std::endl;
        }
        else
        {
            (*stream)<<"Using \"b0_id\"=" <<this->b0_vol_id<<" as reference volume."<<std::endl;
        }
    }

    my_json["B0VolId"]= this->b0_vol_id;
    std::string json_name = this->nii_name.substr(0,this->nii_name.rfind(".nii"))+".json";
    std::ofstream out_json(json_name);
    out_json << std::setw(4) << this->my_json << std::endl;
    out_json.close();
}

template ImageType3D::Pointer DIFFPREP::QuadratictransformImage<ImageType3D>(ImageType3D::Pointer, DIFFPREP::CompositeTransformType::Pointer,std::string,float);
template ImageType3DBool::Pointer DIFFPREP::QuadratictransformImage<ImageType3DBool>(ImageType3DBool::Pointer, DIFFPREP::CompositeTransformType::Pointer,std::string,float);

template ImageType3D::Pointer DIFFPREP::ChangeImageHeaderToDP<ImageType3D>(ImageType3D::Pointer img);
template ImageType4D::Pointer DIFFPREP::ChangeImageHeaderToDP<ImageType4D>(ImageType4D::Pointer img);
template ImageType3DBool::Pointer DIFFPREP::ChangeImageHeaderToDP<ImageType3DBool>(ImageType3DBool::Pointer img);

#endif
