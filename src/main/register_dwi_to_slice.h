#ifndef _RegisterDWIToSlice_H
#define _RegisterDWIToSlice_H


#include "defines.h"
#include "itkDIFFPREPGradientDescentOptimizerv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4Okan.h"
#include "itkOkanImageRegistrationMethodv4.h"
#include "TORTOISE.h"
#include "itkInvertDisplacementFieldImageFilterOkan.h"
#include "itkBSplineInterpolateImageFunction.h"

using OkanQuadraticTransformType=itk::OkanQuadraticTransform<CoordType,3,3>;

void VolumeToSliceRegistration(ImageType3D::Pointer slice_img, ImageType3D::Pointer dwi_img , vnl_matrix<int> slspec,std::vector<float> lim_arr,std::vector<OkanQuadraticTransformType::Pointer> &s2v_transformations, bool do_eddy,std::string phase, ImageType3D::Pointer mask_img=nullptr,int vol=0)
{     
    int Nexc= slspec.rows();
    int MB= slspec.cols();
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    ImageType3D::SizeType sz =slice_img->GetLargestPossibleRegion().GetSize();
    s2v_transformations.resize(sz[2]);
    for(int k=0;k<sz[2];k++)
    {
        OkanQuadraticTransformType::Pointer  initialTransform = OkanQuadraticTransformType::New();
        initialTransform->SetPhase(phase);
        initialTransform->SetIdentity();
        s2v_transformations[k]=initialTransform;
    }


    OkanQuadraticTransformType::ParametersType flags, grd_scales;
    grd_scales.SetSize(OkanQuadraticTransformType::NQUADPARAMS);
    flags.SetSize(OkanQuadraticTransformType::NQUADPARAMS);
    flags.Fill(0);
    flags[0]=flags[1]=flags[2]=flags[3]=flags[4]=flags[5]=1;
    if(do_eddy)
    {
        flags[6]=flags[7]=flags[8]=flags[9]=flags[10]=flags[11]=1;
        flags[12]=flags[13]=1;
    }


    ImageType3D::SpacingType res = slice_img->GetSpacing();
    grd_scales[0]= res[0]*1.25;
    grd_scales[1]= res[1]*1.25;
    grd_scales[2]= res[2]*1.25;
    grd_scales[3]=0.04;
    grd_scales[4]=0.04;
    grd_scales[5]=0.04;
    grd_scales[6]= res[2]*1.5 /   ( sz[0]/2.*res[0]    )*2;
    grd_scales[7]= res[2]*1.5 /   ( sz[1]/2.*res[1]    )*2.;
    grd_scales[8]= res[2]*1.5 /   ( sz[2]/2.*res[2]    )*2.;
    grd_scales[9]=  0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    );
    grd_scales[10]= 0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[11]= 0.5*res[2]*10. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[12]= res[2]*5. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
    grd_scales[13]= res[2]*8. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    )/2.;
    grd_scales[14]=  5.*res[2]*4 /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[15]=  5.*res[2]*1 /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
    grd_scales[16]=  5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    );
    grd_scales[17]=  5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[18]=  5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
    grd_scales[19]=  5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
    grd_scales[20]=  5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
    grd_scales[21]= res[0]*1.25;
    grd_scales[22]= res[1]*1.25;
    grd_scales[23]= res[2]*1.25;
    grd_scales=grd_scales/1.5;


    int fact=1;

    int phase_id=6;
    if(phase=="vertical")
        phase_id=7;
    else if(phase=="slice")
        phase_id=8;

    if(MB==1)
    {
        grd_scales=grd_scales/2.;
        flags[6]=flags[7]=flags[8]=flags[9]=flags[10]=flags[11]=0;
        flags[12]=flags[13]=0;


        for(int k=0;k<sz[2]-fact*2;k++)
        {
            ImageType3D::Pointer  temp_slice_img_itk=ImageType3D::New();
            ImageType3D::SizeType szn= sz;

            szn[2]=fact+1;
            ImageType3D::IndexType start;start.Fill(0);
            ImageType3D::RegionType reg(start,szn);
            temp_slice_img_itk->SetRegions(reg);
            temp_slice_img_itk->Allocate();
            temp_slice_img_itk->FillBuffer(0);
            temp_slice_img_itk->SetDirection(slice_img->GetDirection());

            ImageType3D::SpacingType spc= slice_img->GetSpacing();
            spc[2]=2*spc[2];
            temp_slice_img_itk->SetSpacing(spc);
            ImageType3D::IndexType orig_ind;
            orig_ind[0]=0;
            orig_ind[1]=0;
            orig_ind[2]=k;
            ImageType3D::PointType orig;
            slice_img->TransformIndexToPhysicalPoint(orig_ind,orig);
            temp_slice_img_itk->SetOrigin(orig);

            int nvoxels=0;
            for(int kk=0;kk<fact+1;kk++)
            {
                int sl= k+ kk*2;
                ImageType3D::IndexType ind3,ind3_v2;
                ind3_v2[2]=kk;
                ind3[2]=sl;
                for(int j=0;j<sz[1];j++)
                {
                    ind3[1]=j;
                    ind3_v2[1]=j;
                    for(int i=0;i<sz[0];i++)
                    {
                        ind3[0]=i;
                        ind3_v2[0]=i;

                        temp_slice_img_itk->SetPixel(ind3_v2,slice_img->GetPixel(ind3));
                        if(mask_img->GetPixel(ind3))
                            nvoxels++;
                    }
                }
            }

            OkanQuadraticTransformType::Pointer  initialTransform = OkanQuadraticTransformType::New();
            initialTransform->SetPhase(phase);
            initialTransform->SetIdentity();
            initialTransform->SetParametersForOptimizationFlags(flags);

            if(nvoxels>0.1*sz[0]*sz[1])
            {
                typedef itk::MattesMutualInformationImageToImageMetricv4Okan<ImageType3D,ImageType3D> MetricType;
                MetricType::Pointer         metric        = MetricType::New();
                metric->SetNumberOfHistogramBins(30);
                metric->SetUseMovingImageGradientFilter(false);
                metric->SetUseFixedImageGradientFilter(false);
                metric->SetFixedMin(lim_arr[0]);
                metric->SetFixedMax(lim_arr[1]);
                metric->SetMovingMin(lim_arr[2]);
                metric->SetMovingMax(lim_arr[3]);
                metric->SetMaximumNumberOfWorkUnits(NITK);


                using OptimizerType=itk::DIFFPREPGradientDescentOptimizerv4<double>;
                OptimizerType::Pointer      optimizer     = OptimizerType::New();
                optimizer->SetOptimizationFlags(flags);
                optimizer->SetGradScales(grd_scales);
                optimizer->SetNumberHalves(3);
                optimizer->SetBrkEps(0.0005);


                typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, OkanQuadraticTransformType,ImageType3D >           RegistrationType;
                RegistrationType::Pointer   registration  = RegistrationType::New();
                registration->SetFixedImage(temp_slice_img_itk);
                registration->SetMovingImage(dwi_img);
                registration->SetMetricSamplingPercentage(1.);
                registration->SetOptimizer(optimizer);
                registration->SetInitialTransform(initialTransform);
                registration->InPlaceOn();

                RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
                shrinkFactorsPerLevel.SetSize( 1 );
                shrinkFactorsPerLevel[0] = 1;
                RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
                smoothingSigmasPerLevel.SetSize( 1 );
                smoothingSigmasPerLevel[0] = 0.;


                registration->SetNumberOfLevels( 1 );
                registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
                registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
                registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
                registration->SetNumberOfWorkUnits(NITK);
                registration->SetMetric(        metric        );

                try
                {
                      registration->Update();


                      for(int kk=0;kk<fact+1;kk++)
                      {
                          int sl= k+ kk*2;
                          if(s2v_transformations[sl]->GetParameters()[phase_id]==1)
                              s2v_transformations[sl]->SetParameters(initialTransform->GetParameters());
                          else
                          {
                              auto op = s2v_transformations[sl]->GetParameters();
                              for(int pp=0;pp<op.size();pp++ )
                                  op[pp]+= initialTransform->GetParameters()[pp];
                              s2v_transformations[sl]->SetParameters(op);
                          }
                      }

                }
                catch( itk::ExceptionObject & err )
                {
                      std::cerr << "ExceptionObject caught !" << std::endl;
                      std::cerr<<"Vol: " << vol << " Slice: "<< k<<std::endl;
                      std::cerr<<"Probably insufficient data on slice. Reverting to identity transform"<<std::endl;

                 //     s2v_transformations[k]->SetIdentity();
                }
            } // if nvoxels
            else
            {
              //  s2v_transformations[k]->SetIdentity();
            }
        } //for k


        for(int k=0;k<sz[2];k++)
        {
            auto op = s2v_transformations[k]->GetParameters();
            for(int pp=0;pp<op.size();pp++ )
                op[pp]/= op[phase_id];
            s2v_transformations[k]->SetParameters(op);
        }

    }
    else
    {
        sz[2]=MB;
#ifndef USECUDA
//  #pragma omp parallel for
#endif
        for(int e=0;e<Nexc;e++)
        {
            ImageType3D::Pointer  temp_slice_img_itk=ImageType3D::New();

            ImageType3D::IndexType start;start.Fill(0);
            ImageType3D::RegionType reg(start,sz);
            temp_slice_img_itk->SetRegions(reg);
            temp_slice_img_itk->Allocate();
            temp_slice_img_itk->FillBuffer(0);
            temp_slice_img_itk->SetDirection(slice_img->GetDirection());

            ImageType3D::SpacingType spc= slice_img->GetSpacing();
            spc[2]=(slspec(e,1)-slspec(e,0))*spc[2];
            temp_slice_img_itk->SetSpacing(spc);

            ImageType3D::IndexType orig_ind;
            orig_ind[0]=0;
            orig_ind[1]=0;
            orig_ind[2]=slspec(e,0);

            ImageType3D::PointType orig;
            slice_img->TransformIndexToPhysicalPoint(orig_ind,orig);
            temp_slice_img_itk->SetOrigin(orig);

            for(int kk=0;kk<MB;kk++)
            {
                int k=slspec(e,kk);
                ImageType3D::IndexType ind3,ind3_v2;
                ind3_v2[2]=kk;
                ind3[2]=k;
                for(int j=0;j<sz[1];j++)
                {
                    ind3[1]=j;
                    ind3_v2[1]=j;
                    for(int i=0;i<sz[0];i++)
                    {
                        ind3[0]=i;
                        ind3_v2[0]=i;
                        temp_slice_img_itk->SetPixel(ind3_v2,slice_img->GetPixel(ind3));
                    }
                }
            }

            OkanQuadraticTransformType::Pointer  initialTransform = OkanQuadraticTransformType::New();
            initialTransform->SetPhase(phase);
            initialTransform->SetIdentity();
            initialTransform->SetParametersForOptimizationFlags(flags);

            typedef itk::MattesMutualInformationImageToImageMetricv4Okan<ImageType3D,ImageType3D> MetricType;
            MetricType::Pointer         metric        = MetricType::New();
            metric->SetNumberOfHistogramBins(30);
            metric->SetUseMovingImageGradientFilter(false);
            metric->SetUseFixedImageGradientFilter(false);
            metric->SetFixedMin(lim_arr[0]);
            metric->SetFixedMax(lim_arr[1]);
            metric->SetMovingMin(lim_arr[2]);
            metric->SetMovingMax(lim_arr[3]);
            metric->SetMaximumNumberOfWorkUnits(NITK);


            using OptimizerType=itk::DIFFPREPGradientDescentOptimizerv4<double>;
            OptimizerType::Pointer      optimizer     = OptimizerType::New();
            optimizer->SetOptimizationFlags(flags);
            optimizer->SetGradScales(grd_scales);
            optimizer->SetNumberHalves(5);
            optimizer->SetBrkEps(0.0005);


            typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, OkanQuadraticTransformType,ImageType3D >           RegistrationType;
            RegistrationType::Pointer   registration  = RegistrationType::New();
            registration->SetFixedImage(temp_slice_img_itk);
            registration->SetMovingImage(dwi_img);
            registration->SetMetricSamplingPercentage(1.);
            registration->SetOptimizer(optimizer);
            registration->SetInitialTransform(initialTransform);
            registration->InPlaceOn();

            RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
            shrinkFactorsPerLevel.SetSize( 1 );
            shrinkFactorsPerLevel[0] = 1;
            RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
            smoothingSigmasPerLevel.SetSize( 1 );
            smoothingSigmasPerLevel[0] = 0.;


            registration->SetNumberOfLevels( 1 );
            registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
            registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
            registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
            registration->SetNumberOfWorkUnits(NITK);
            registration->SetMetric(        metric        );

            try
            {
                  registration->Update();
                  for(int kk=0;kk<MB;kk++)
                      s2v_transformations[ slspec(e,kk)]=initialTransform;
            }
            catch( itk::ExceptionObject & err )
            {
                  std::cerr << "ExceptionObject caught !" << std::endl;
                  std::cerr<<"Vol: " << vol << " Slice: "<< e<<std::endl;
                  std::cerr<<"Probably insufficient data on slice. Reverting to identity transform"<<std::endl;

                  for(int kk=0;kk<MB;kk++)
                  {
                      OkanQuadraticTransformType::Pointer tTransform = OkanQuadraticTransformType::New();
                      tTransform->SetPhase(phase);
                      tTransform->SetIdentity();
                      s2v_transformations[ slspec(e,kk)]=tTransform;
                  }
            }
        } //for nexc

    } //if mb==1




}

ImageType3D::Pointer ForwardTransformImage(ImageType3D::Pointer img, std::vector<OkanQuadraticTransformType::Pointer> s2v_trans)
{
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    ImageType3D::Pointer final_img = ImageType3D::New();
    final_img->SetRegions(img->GetLargestPossibleRegion());
    final_img->Allocate();
    final_img->SetSpacing(img->GetSpacing());
    final_img->SetDirection(img->GetDirection());
    final_img->SetOrigin(img->GetOrigin());
    final_img->FillBuffer(0.);


    std::vector<float> values;

    using MeasurementVectorType = itk::Vector<float, 3>;
    using SampleType = itk::Statistics::ListSample<MeasurementVectorType>;
    SampleType::Pointer sample = SampleType::New();
    sample->SetMeasurementVectorSize(3);


    ImageType3D::SizeType sz= img->GetLargestPossibleRegion().GetSize();


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

                ImageType3D::PointType pt,pt_trans;

                img->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=s2v_trans[k]->TransformPoint(pt);

                itk::ContinuousIndex<double,3> ind3_t;
                final_img->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_t);
                MeasurementVectorType tt;
                tt[0]=ind3_t[0];
                tt[1]=ind3_t[1];
                tt[2]=ind3_t[2];

                sample->PushBack(tt);
                values.push_back(img->GetPixel(ind3));
            }
        }
    }

    using TreeGeneratorType = itk::Statistics::KdTreeGenerator<SampleType>;
    TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
    treeGenerator->SetSample(sample);
    treeGenerator->SetBucketSize(8);
    treeGenerator->Update();

    using TreeType = TreeGeneratorType::KdTreeType;
    TreeType::Pointer tree = treeGenerator->GetOutput();


    itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_img, final_img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3 = it.GetIndex();
        MeasurementVectorType queryPoint;
        queryPoint[0]=ind3[0];
        queryPoint[1]=ind3[1];
        queryPoint[2]=ind3[2];

        unsigned int                           numberOfNeighbors = 8;
        TreeType::InstanceIdentifierVectorType neighbors;
        tree->Search(queryPoint, numberOfNeighbors, neighbors);

        double sm_weight=0;
        double sm_val=0;

        for (unsigned long neighbor : neighbors)
        {
            MeasurementVectorType  aa= tree->GetMeasurementVector(neighbor) ;
            float dx=aa[0]-ind3[0];
            float dy=aa[1]-ind3[1];
            float dz=aa[2]-ind3[2];

            float dist= sqrt(dx*dx+dy*dy+dz*dz);
            float dist2= pow(dist,4.);

            if(dist<0.1)
            {
                sm_val= values[neighbor];
                sm_weight=1;
                break;
            }
            else
            {
                sm_val+= values[neighbor] / dist2;
                sm_weight+= 1./dist2;
            }
        }

        if(sm_weight==0 || std::isnan(sm_weight) ||  !std::isfinite(sm_weight))
            it.Set(0);
        else
        {
            it.Set(sm_val/sm_weight);
        }


        ++it;
    }

    return final_img;



}




ImageType3D::Pointer ForwardTransformImage2(ImageType3D::Pointer img, std::vector<OkanQuadraticTransformType::Pointer> s2v_trans)
{
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    ImageType3D::Pointer final_img = ImageType3D::New();
    final_img->SetRegions(img->GetLargestPossibleRegion());
    final_img->Allocate();
    final_img->SetSpacing(img->GetSpacing());
    final_img->SetDirection(img->GetDirection());
    final_img->SetOrigin(img->GetOrigin());
    final_img->FillBuffer(0.);


    std::vector<float> values;

    using MeasurementVectorType = itk::Vector<float, 3>;
    using SampleType = itk::Statistics::ListSample<MeasurementVectorType>;
    SampleType::Pointer sample = SampleType::New();
    sample->SetMeasurementVectorSize(3);


    ImageType3D::SizeType sz= img->GetLargestPossibleRegion().GetSize();


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

                ImageType3D::PointType pt,pt_trans;

                img->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=s2v_trans[k]->TransformPoint(pt);

                itk::ContinuousIndex<double,3> ind3_t;
                final_img->TransformPhysicalPointToContinuousIndex(pt_trans,ind3_t);
                MeasurementVectorType tt;
                tt[0]=ind3_t[0];
                tt[1]=ind3_t[1];
                tt[2]=ind3_t[2];

                sample->PushBack(tt);
                values.push_back(img->GetPixel(ind3));
            }
        }
    }

    using TreeGeneratorType = itk::Statistics::KdTreeGenerator<SampleType>;
    TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
    treeGenerator->SetSample(sample);
    treeGenerator->SetBucketSize(8);
    treeGenerator->Update();

    using TreeType = TreeGeneratorType::KdTreeType;
    TreeType::Pointer tree = treeGenerator->GetOutput();



    DisplacementFieldType::Pointer forward_s2v_field = DisplacementFieldType::New();
    forward_s2v_field->SetRegions(img->GetLargestPossibleRegion());
    forward_s2v_field->Allocate();
    forward_s2v_field->SetSpacing(img->GetSpacing());
    forward_s2v_field->SetSpacing(img->GetSpacing());
    forward_s2v_field->SetOrigin(img->GetOrigin());
    forward_s2v_field->SetDirection(img->GetDirection());
    DisplacementFieldType::PixelType zero; zero.Fill(0);
    forward_s2v_field->FillBuffer(zero);


    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it2(forward_s2v_field,forward_s2v_field->GetLargestPossibleRegion());
    for(it2.GoToBegin();!it2.IsAtEnd();++it2)
    {
        ImageType3D::IndexType ind3=it2.GetIndex();

        DisplacementFieldType::PointType pt,pt_trans;
        forward_s2v_field->TransformIndexToPhysicalPoint(ind3,pt);
        pt_trans = s2v_trans[ind3[2]]->TransformPoint(pt);

        DisplacementFieldType::PixelType vec;
        vec[0]= pt_trans[0] - pt[0];
        vec[1]= pt_trans[1] - pt[1];
        vec[2]= pt_trans[2] - pt[2];
        it2.Set(vec);
    }


    typedef itk::InvertDisplacementFieldImageFilterOkan<DisplacementFieldType> InverterType;
    InverterType::Pointer inverter = InverterType::New();
    inverter->SetInput( forward_s2v_field );
    inverter->SetMaximumNumberOfIterations( 50 );
    inverter->SetMeanErrorToleranceThreshold( 0.0004 );
    inverter->SetMaxErrorToleranceThreshold( 0.04 );
    //inverter->SetNumberOfWorkUnits(NITK);
    inverter->SetNumberOfWorkUnits(1);
    inverter->Update();
    DisplacementFieldType::Pointer backward_s2v_field =inverter->GetOutput();

    using DisplacementFieldTransformType= itk::DisplacementFieldTransform<double,3>;
    DisplacementFieldTransformType::Pointer backward_s2v_field_trans= DisplacementFieldTransformType::New();
    backward_s2v_field_trans->SetDisplacementField(backward_s2v_field);


    using BSInterpolatorType = itk::BSplineInterpolateImageFunction<ImageType3D, double>;
    BSInterpolatorType::Pointer BSinterpolator = BSInterpolatorType::New();
    BSinterpolator->SetSplineOrder(3);
    BSinterpolator->SetInputImage(img);


    itk::ImageRegionIteratorWithIndex<ImageType3D> it(final_img, final_img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        bool forward_interp=false;

        ImageType3D::IndexType ind3 = it.GetIndex();
        DisplacementFieldType::PointType pt,pt_trans;
        final_img->TransformIndexToPhysicalPoint(ind3,pt);

        ImageType3D::PointType pt_trans_backward= backward_s2v_field_trans->TransformPoint(pt);
        itk::ContinuousIndex<double,3> ind3_ts2v;
        img->TransformPhysicalPointToContinuousIndex(pt_trans_backward,ind3_ts2v);


        int SL= (int)std::round(ind3_ts2v[2]);
        if(SL<0 || SL > img->GetLargestPossibleRegion().GetSize()[2]-1)
            forward_interp=true;
        else
        {
            ImageType3D::PointType pt_trans_s2v_inv= s2v_trans[SL]->TransformPoint(pt_trans_backward);

            DisplacementFieldType::PixelType diff;
            diff[0]= pt_trans_s2v_inv[0] - pt[0];
            diff[1]= pt_trans_s2v_inv[1] - pt[1];
            diff[2]= pt_trans_s2v_inv[2] - pt[2];
            double diff_mag = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);

            if(diff_mag > 0.1*img->GetSpacing()[2])
                forward_interp=true;
        }

        if(!forward_interp )
        {
            if(BSinterpolator->IsInsideBuffer(pt_trans_backward))
            {
                ImageType3D::PixelType val = BSinterpolator->Evaluate(pt_trans_backward);
                if(val<0)
                    val=0;
                final_img->SetPixel(ind3,val);
            }
        }
        else
        {





            ImageType3D::IndexType ind3 = it.GetIndex();
            MeasurementVectorType queryPoint;
            queryPoint[0]=ind3[0];
            queryPoint[1]=ind3[1];
            queryPoint[2]=ind3[2];

            unsigned int                           numberOfNeighbors = 8;
            TreeType::InstanceIdentifierVectorType neighbors;
            tree->Search(queryPoint, numberOfNeighbors, neighbors);

            double sm_weight=0;
            double sm_val=0;

            for (unsigned long neighbor : neighbors)
            {
                MeasurementVectorType  aa= tree->GetMeasurementVector(neighbor) ;
                float dx=aa[0]-ind3[0];
                float dy=aa[1]-ind3[1];
                float dz=aa[2]-ind3[2];

                float dist= sqrt(dx*dx+dy*dy+dz*dz);
                float dist2= pow(dist,8.);

                if(dist<0.1)
                {
                    sm_val= values[neighbor];
                    sm_weight=1;
                    break;
                }
                else
                {
                    sm_val+= values[neighbor] / dist2;
                    sm_weight+= 1./dist2;
                }
            }

            if(sm_weight==0 || std::isnan(sm_weight) ||  !std::isfinite(sm_weight))
                it.Set(0);
            else
            {
                it.Set(sm_val/sm_weight);
            }
        }


        ++it;
    }

    return final_img;



}





#endif
