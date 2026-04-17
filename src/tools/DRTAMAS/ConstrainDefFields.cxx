#include <iostream>
#include <fstream>
using namespace std;

#include "itkInvertDisplacementFieldImageFilter.h"
#include "itkWindowConvergenceMonitoringFunction.h"
#include "itkCompositeTransform.h"
#include "itkGaussianOperator.h"
#include "itkVectorNeighborhoodOperatorImageFilter.h"
#include "itkDisplacementFieldTransform.h"

#include "itkAddImageFilter.h"
#include <chrono>

//#include "defines.h"
//#include "DRTAMAS_utilities_cp.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "itkImageFileReader.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

typedef itk::MatrixOffsetTransformBase<float, 3,      3> AffineTransformType;
using JacobianImageType = itk::Image< vnl_matrix_fixed<float,3,3> , 3>;
using DisplacementFieldType = itk::DisplacementFieldTransform<float,3>::DisplacementFieldType;
using ImageType3D = itk::Image<float,3>;



vnl_matrix_fixed<float,3,3> ComputeJacobian(DisplacementFieldType::Pointer field,DisplacementFieldType::IndexType &ind3 )
{
    vnl_matrix_fixed<float,3,3> A;
    A.set_identity();

    ImageType3D::SizeType sz= field->GetLargestPossibleRegion().GetSize();

    if(ind3[0]<1 || ind3[0]>sz[0]-2 || ind3[1]<1 || ind3[1]>sz[1]-2 || ind3[2]<1 || ind3[2]>sz[2]-2 )
        return A;


    ImageType3D::SpacingType spc = field->GetSpacing();
    ImageType3D::DirectionType dir = field->GetDirection();

    vnl_matrix_fixed<float,3,3> SD;
    SD(0,0)=dir(0,0)/spc[0]; SD(0,1)=dir(1,0)/spc[0]; SD(0,2)=dir(2,0)/spc[0];
    SD(1,0)=dir(0,1)/spc[1]; SD(1,1)=dir(1,1)/spc[1]; SD(1,2)=dir(2,1)/spc[1];
    SD(2,0)=dir(0,2)/spc[2]; SD(2,1)=dir(1,2)/spc[2]; SD(2,2)=dir(2,2)/spc[2];

    ImageType3D::IndexType indt=ind3;
    for(int d=0;d<3;d++)
    {
        indt[d]++;
        DisplacementFieldType::PixelType vecp = field->GetPixel(indt);
        indt[d]-=2;
        DisplacementFieldType::PixelType vecm = field->GetPixel(indt);
        indt[d]++;

        auto diff = 0.5f*(vecp-vecm);

        A(0,d)=diff[0];
        A(1,d)=diff[1];
        A(2,d)=diff[2];
    }

    A= A*SD;
    A(0,0)+=1;
    A(1,1)+=1;
    A(2,2)+=1;

    return A;
}


template <typename ImageType>
void FixBoundaries(typename ImageType::Pointer field)
{
    typename ImageType::SizeType sz= field->GetLargestPossibleRegion().GetSize();


    int K=2;

    {//i=0
        ImageType3D::IndexType ind3, ind3n;
        ind3n[0]=K;
        for(int i=0;i<K;i++)
        {
            ind3[0]=i;
            for(int k=0;k<sz[2];k++)
            {
                ind3[2]=k;
                ind3n[2]=k;

                for(int j=0;j<sz[1];j++)
                {
                    ind3[1]=j;
                    ind3n[1]=j;
                    field->SetPixel(ind3, field->GetPixel(ind3n));
                }
            }
        }
    }
    {//i=sz0
        ImageType3D::IndexType ind3, ind3n;
        ind3n[0]=sz[0]-1-K;
        for(int i=sz[0]-1;i>sz[0]-1-K;i--)
        {
            ind3[0]=i;
            for(int k=0;k<sz[2];k++)
            {
                ind3[2]=k;
                ind3n[2]=k;

                for(int j=0;j<sz[1];j++)
                {
                    ind3[1]=j;
                    ind3n[1]=j;
                    field->SetPixel(ind3, field->GetPixel(ind3n));
                }
            }
        }
    }


    {//j=0
        ImageType3D::IndexType ind3, ind3n;
        ind3n[1]=K;
        for(int j=0;j<K;j++)
        {
            ind3[1]=j;
            for(int k=0;k<sz[2];k++)
            {
                ind3[2]=k;
                ind3n[2]=k;

                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    ind3n[0]=i;
                    field->SetPixel(ind3, field->GetPixel(ind3n));
                }
            }
        }
    }
    {//j=sz0
        ImageType3D::IndexType ind3, ind3n;
        ind3n[1]=sz[1]-1-K;
        for(int j=sz[1]-1;j>sz[1]-1-K;j--)
        {
            ind3[1]=j;
            for(int k=0;k<sz[2];k++)
            {
                ind3[2]=k;
                ind3n[2]=k;

                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    ind3n[0]=i;
                    field->SetPixel(ind3, field->GetPixel(ind3n));
                }
            }
        }
    }


    {//k=0
        ImageType3D::IndexType ind3, ind3n;
        ind3n[2]=K;
        for(int k=0;k<K;k++)
        {
            ind3[2]=k;
            for(int j=0;j<sz[1];j++)
            {
                ind3[1]=j;
                ind3n[1]=j;

                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    ind3n[0]=i;
                    field->SetPixel(ind3, field->GetPixel(ind3n));
                }
            }
        }
    }
    {//j=sz0
        ImageType3D::IndexType ind3, ind3n;
        ind3n[2]=sz[2]-1-K;
        for(int k=sz[2]-1;k>sz[2]-1-K;k--)
        {
            ind3[2]=k;
            for(int j=0;j<sz[1];j++)
            {
                ind3[1]=j;
                ind3n[1]=j;

                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    ind3n[0]=i;
                    field->SetPixel(ind3, field->GetPixel(ind3n));
                }
            }
        }
    }
}


DisplacementFieldType::Pointer  GaussianSmoothDisplacementField(const DisplacementFieldType::Pointer field, const float variance)
{
    using DuplicatorType = itk::ImageDuplicator<DisplacementFieldType>;
    auto duplicator = DuplicatorType::New();
    duplicator->SetInputImage(field);
    duplicator->Update();

    DisplacementFieldType::Pointer smoothField = duplicator->GetOutput();

    if (variance <= 0.0)
    {
        return smoothField;
    }

    using GaussianSmoothingOperatorType = itk::GaussianOperator<float, 3>;
    GaussianSmoothingOperatorType gaussianSmoothingOperator;

    using GaussianSmoothingSmootherType =   itk::VectorNeighborhoodOperatorImageFilter<DisplacementFieldType, DisplacementFieldType>;
    auto smoother = GaussianSmoothingSmootherType::New();

    for (int d = 0; d < 3; ++d)
    {
        // smooth along this dimension
        gaussianSmoothingOperator.SetDirection(d);
        gaussianSmoothingOperator.SetVariance(variance);
        gaussianSmoothingOperator.SetMaximumError(0.001);
        gaussianSmoothingOperator.SetMaximumKernelWidth(smoothField->GetRequestedRegion().GetSize()[d]);
        gaussianSmoothingOperator.CreateDirectional();

        // todo: make sure we only smooth within the buffered region
        smoother->SetOperator(gaussianSmoothingOperator);
        smoother->SetInput(smoothField);
        smoother->Update();

        smoothField = smoother->GetOutput();
        smoothField->Update();
        smoothField->DisconnectPipeline();
    }

    const DisplacementFieldType::PixelType zeroVector{};

    // make sure boundary does not move
    float weight1 = 1.0;
    if (variance < 0.5)
    {
        weight1 = 1.0 - 1.0 * (variance / 0.5);
    }
    float weight2 = 1.0 - weight1;

    DisplacementFieldType::RegionType region = field->GetLargestPossibleRegion();
    DisplacementFieldType::SizeType   size = region.GetSize();
    DisplacementFieldType::IndexType  startIndex = region.GetIndex();

    itk::ImageRegionConstIteratorWithIndex<DisplacementFieldType> ItF(field, field->GetLargestPossibleRegion());
    itk::ImageRegionIteratorWithIndex<DisplacementFieldType>      ItS(smoothField, smoothField->GetLargestPossibleRegion());
    for (ItF.GoToBegin(), ItS.GoToBegin(); !ItF.IsAtEnd(); ++ItF, ++ItS)
    {
        DisplacementFieldType::IndexType index = ItF.GetIndex();
        bool                                      isOnBoundary = false;
        for (unsigned int d = 0; d < 3; ++d)
        {
            if (index[d] == startIndex[d] || index[d] == (size[d]) - startIndex[d] - 1)
            {
                isOnBoundary = true;
                break;
            }
        }
        if (isOnBoundary)
        {
            ItS.Set(zeroVector);
        }
        else
        {
            ItS.Set(ItS.Get() * weight1 + ItF.Get() * weight2);
        }
    }

    return smoothField;
}

ImageType3D::Pointer  ExtractJacobianComponent(JacobianImageType::Pointer JAC_img,int row,int col)
{
    ImageType3D::Pointer img=ImageType3D::New();
    img->SetRegions(JAC_img->GetLargestPossibleRegion());
    img->Allocate();
    img->SetDirection(JAC_img->GetDirection());
    img->SetOrigin(JAC_img->GetOrigin());
    img->SetSpacing(JAC_img->GetSpacing());


    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img,img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        JacobianImageType::PixelType A= JAC_img->GetPixel(ind3);

        it.Set(A(row,col));
    }

    return img;
}




DisplacementFieldType::Pointer  AverageFields(std::vector<DisplacementFieldType::Pointer> fields)
{
    int N= fields.size();

    DisplacementFieldType::Pointer avg_field=DisplacementFieldType::New();
    avg_field->SetRegions(fields[0]->GetLargestPossibleRegion());
    avg_field->Allocate();
    avg_field->SetDirection(fields[0]->GetDirection());
    avg_field->SetOrigin(fields[0]->GetOrigin());
    avg_field->SetSpacing(fields[0]->GetSpacing());
    DisplacementFieldType::PixelType zero; zero.Fill(0);
    avg_field->FillBuffer(zero);

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(fields[0],fields[0]->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        DisplacementFieldType::PixelType vec; vec.Fill(0);

        for(int v=0;v<N;v++)
        {
            vec= vec + fields[v]->GetPixel(ind3);
        }
        vec= vec/N;
        avg_field->SetPixel(ind3,vec);
    }
    return avg_field;
}

DisplacementFieldType::Pointer InvertDisplacementField( const DisplacementFieldType * field)
{
    typedef itk::InvertDisplacementFieldImageFilter<DisplacementFieldType> InverterType;

    InverterType::Pointer inverter = InverterType::New();
    inverter->SetInput( field );
    inverter->SetMaximumNumberOfIterations( 200 );
    inverter->SetMeanErrorToleranceThreshold( 0.0003 );
    inverter->SetMaxErrorToleranceThreshold( 0.03 );
    inverter->Update();

    DisplacementFieldType::Pointer inverseField = inverter->GetOutput();

    return inverseField;
}


vnl_matrix_fixed<float,3,3>  AverageAffinesReturnInv(std::vector<vnl_matrix_fixed<float,3,3> >& all_affines)
{
    int N= all_affines.size();

    Eigen::Matrix3f avg_eigen_mat = Eigen::Matrix3f::Zero();
    for(int i=0;i<N;i++)
    {
        Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> curr_eigen_mat(
            all_affines[i].data_block(),
            all_affines[i].rows(),
            all_affines[i].cols()
            );


        Eigen::Matrix3f L =curr_eigen_mat.log();
        avg_eigen_mat= avg_eigen_mat +L;
    }
    avg_eigen_mat=avg_eigen_mat/N;
    avg_eigen_mat= avg_eigen_mat.exp();

    Eigen::Matrix3f avg_eigen_mat_inv = avg_eigen_mat.inverse();

    vnl_matrix_fixed<float,3,3> mat;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat(i, j) = avg_eigen_mat_inv(i, j);
        }
    }
    return mat;
}



DisplacementFieldType::Pointer  CreateFieldOutOfJacobian(JacobianImageType::Pointer jac_field,DisplacementFieldType::Pointer init_disp_field=nullptr,ImageType3D::Pointer mask_img=nullptr)
{
    typedef itk::ImageDuplicator<DisplacementFieldType> DupType;

    DisplacementFieldType::Pointer curr_field=nullptr;
    if(init_disp_field)
    {

        DupType::Pointer dup = DupType::New();
        dup->SetInputImage(init_disp_field);
        dup->Update();
        curr_field = dup->GetOutput();
    }
    else
    {
        curr_field=DisplacementFieldType::New();
        curr_field->SetRegions(jac_field->GetLargestPossibleRegion());
        curr_field->Allocate();
        curr_field->SetDirection(jac_field->GetDirection());
        curr_field->SetSpacing(jac_field->GetSpacing());
        curr_field->SetOrigin(jac_field->GetOrigin());
        DisplacementFieldType::PixelType zer; zer.Fill(0);
        curr_field->FillBuffer(zer);
    }



    DisplacementFieldType::SpacingType spc= curr_field->GetSpacing();
    DisplacementFieldType::SizeType sz= curr_field->GetLargestPossibleRegion().GetSize();
    DisplacementFieldType::DirectionType dir=curr_field->GetDirection();

    vnl_matrix_fixed<float,3,3> SD;
    SD(0,0)=dir(0,0)/spc[0];   SD(0,1)=dir(1,0)/spc[0];   SD(0,2)=dir(2,0)/spc[0];
    SD(1,0)=dir(0,1)/spc[1];   SD(1,1)=dir(1,1)/spc[1];   SD(1,2)=dir(2,1)/spc[1];
    SD(2,0)=dir(0,2)/spc[2];   SD(2,1)=dir(1,2)/spc[2];   SD(2,2)=dir(2,2)/spc[2];


    // Monitor the convergence
    typedef itk::Function::WindowConvergenceMonitoringFunction<float> ConvergenceMonitoringType;
    ConvergenceMonitoringType::Pointer ConvergenceMonitoring=ConvergenceMonitoringType::New();
    ConvergenceMonitoring->SetWindowSize( 10);


    float alpha=0.05;

    int Mit=0;
    bool converged=false;
    float curr_convergence =1;

    JacobianImageType::Pointer jacs_img=JacobianImageType::New();
    jacs_img->SetRegions(jac_field->GetLargestPossibleRegion());
    jacs_img->Allocate();
    jacs_img->SetDirection(jac_field->GetDirection());
    jacs_img->SetOrigin(jac_field->GetOrigin());
    jacs_img->SetSpacing(jac_field->GetSpacing());
    JacobianImageType::PixelType id;
    id.set_identity();
    jacs_img->FillBuffer(id);



    DupType::Pointer dup2 = DupType::New();
    dup2->SetInputImage(curr_field);
    dup2->Update();
    DisplacementFieldType::Pointer delta_img = dup2->GetOutput();
    DisplacementFieldType::PixelType zer; zer.Fill(0);
    delta_img->FillBuffer(zer);

    const int MARGIN=1;

    while(!converged )
    //while(Mit<50)
    {
        std::vector<float> TOT_errors(sz[2],0);

        for(int d=0;d<3;d++)
        {
            #pragma omp parallel for
            for(int k=MARGIN;k<sz[2]-MARGIN;k++)
            {
                DisplacementFieldType::IndexType ind3, indt;
                ind3[2]=k;
                indt[2]=k;
                for(int j=MARGIN;j<sz[1]-MARGIN;j++)
                {
                    ind3[1]=j;
                    indt[1]=j;
                    for(int i=MARGIN;i<sz[0]-MARGIN;i++)
                    {
                        ind3[0]=i;
                        indt[0]=i;

                        indt[d]++;
                        DisplacementFieldType::PixelType vecp = curr_field->GetPixel(indt);
                        indt[d]-=2;
                        DisplacementFieldType::PixelType vecm = curr_field->GetPixel(indt);
                        indt[d]++;
                        auto dff = 0.5f*(vecp-vecm);

                        JacobianImageType::PixelType curr_comp_jac=jacs_img->GetPixel(ind3);
                        curr_comp_jac(0,d)=dff[0];
                        curr_comp_jac(1,d)=dff[1];
                        curr_comp_jac(2,d)=dff[2];
                        if(d==2)
                        {
                            curr_comp_jac= curr_comp_jac*SD;
                            curr_comp_jac(0,0)++;
                            curr_comp_jac(1,1)++;
                            curr_comp_jac(2,2)++;
                        }
                        jacs_img->SetPixel(ind3,curr_comp_jac);
                    }
                }
            }
        }

        FixBoundaries<JacobianImageType>(jacs_img);



        #pragma omp parallel for
        for(int k=MARGIN;k<sz[2]-MARGIN;k++)
        {
            DisplacementFieldType::IndexType ind3;
            ind3[2]=k;
            for(int j=MARGIN;j<sz[1]-MARGIN;j++)
            {
                ind3[1]=j;
                for(int i=MARGIN;i<sz[0]-MARGIN;i++)
                {
                    ind3[0]=i;

                    if( (mask_img && mask_img->GetPixel(ind3)) || (!mask_img))
                    {

                        JacobianImageType::PixelType real_jac= jac_field->GetPixel(ind3);
                        JacobianImageType::PixelType curr_comp_jac = jacs_img->GetPixel(ind3);

                        JacobianImageType::PixelType diff= real_jac - curr_comp_jac;
                        float error = diff.frobenius_norm();

                        TOT_errors[k]+=error*error;

                        DisplacementFieldType::PixelType grad; grad.Fill(0);
                        for(int sdim=0;sdim<3;sdim++)
                        {
                            //i-1
                            {
                                DisplacementFieldType::IndexType tind=ind3;
                                tind[sdim]= ind3[sdim]-1;

                                if(tind[sdim]>=0)
                                {
                                    JacobianImageType::PixelType RJ= jac_field->GetPixel(tind);
                                    JacobianImageType::PixelType CCJ = jacs_img->GetPixel(tind);
                                    JacobianImageType::PixelType df= RJ-CCJ;

                                    grad[0]+= -2 * df(0,sdim) * 0.5  *(SD(sdim,0)+SD(sdim,1)+SD(sdim,2)) ;// / spc[sdim];
                                    grad[1]+= -2 * df(1,sdim) * 0.5  *(SD(sdim,0)+SD(sdim,1)+SD(sdim,2));// / spc[sdim];
                                    grad[2]+= -2 * df(2,sdim) * 0.5  *(SD(sdim,0)+SD(sdim,1)+SD(sdim,2));// / spc[sdim];
                                }
                            }
                            //i+1
                            {
                                DisplacementFieldType::IndexType tind=ind3;
                                tind[sdim]= ind3[sdim]+1;

                                if(tind[sdim]<sz[sdim])
                                {
                                    JacobianImageType::PixelType RJ= jac_field->GetPixel(tind);
                                    JacobianImageType::PixelType CCJ = jacs_img->GetPixel(tind);

                                    JacobianImageType::PixelType df= RJ-CCJ;

                                    grad[0]+= 2 * df(0,sdim) * 0.5  *(SD(sdim,0)+SD(sdim,1)+SD(sdim,2)) ;// / spc[sdim];
                                    grad[1]+= 2 * df(1,sdim) * 0.5  *(SD(sdim,0)+SD(sdim,1)+SD(sdim,2));// / spc[sdim];
                                    grad[2]+= 2 * df(2,sdim) * 0.5  *(SD(sdim,0)+SD(sdim,1)+SD(sdim,2));// / spc[sdim];

                                }
                            }
                        }

                        grad= -alpha * grad;
                        delta_img->SetPixel(ind3,grad);
                    }
                } //i
            } //j
        } //k


        using AdderType =itk::AddImageFilter<DisplacementFieldType,DisplacementFieldType,DisplacementFieldType>;
        AdderType::Pointer adder= AdderType::New();
        adder->SetInput1(curr_field);
        adder->SetInput2(delta_img);
        adder->Update();
        curr_field= adder->GetOutput();

        float TOT_error=0;
        for(int k=0;k<sz[2];k++)
            TOT_error+= TOT_errors[k];

        ConvergenceMonitoring->AddEnergyValue( TOT_error );

        float prev_conv= curr_convergence;
        curr_convergence= ConvergenceMonitoring->GetConvergenceValue();


        if( (0.7*curr_convergence+0.3*prev_conv) < 1E-8)
        {
                converged = true;
        }

        std::cout<<"ITER: "<< Mit << "  error: " <<TOT_error << " conv: " << curr_convergence<<std::endl;

        ++Mit;
    } //CONVERGED

    return curr_field;
}

DisplacementFieldType::Pointer CombineFields(DisplacementFieldType::Pointer field1, DisplacementFieldType::Pointer field2)
{
    DisplacementFieldType::Pointer combined_displacement_field= DisplacementFieldType::New();
    combined_displacement_field->SetRegions(field1->GetLargestPossibleRegion());
    combined_displacement_field->Allocate();
    combined_displacement_field->SetSpacing(field1->GetSpacing());
    combined_displacement_field->SetOrigin(field1->GetOrigin());
    combined_displacement_field->SetDirection(field1->GetDirection());
    DisplacementFieldType::PixelType zer; zer.Fill(0);
    combined_displacement_field->FillBuffer(zer)    ;

    DisplacementFieldType::SizeType imsize= field1->GetLargestPossibleRegion().GetSize();

    typedef itk::DisplacementFieldTransform<float, 3>    DisplacementFieldTransformType;
    typedef itk::CompositeTransform<float, 3>                CompositeTransformType;

    CompositeTransformType::Pointer composite_trans= CompositeTransformType::New();
    DisplacementFieldTransformType::Pointer disp_trans1= DisplacementFieldTransformType::New();
    disp_trans1->SetDisplacementField(field1);
    DisplacementFieldTransformType::Pointer disp_trans2= DisplacementFieldTransformType::New();
    disp_trans2->SetDisplacementField(field2);
    composite_trans->AddTransform(disp_trans1);
    composite_trans->AddTransform(disp_trans2);


    for( int k=0; k<(int)imsize[2];k++)
    {
        DisplacementFieldType::IndexType index;
        index[2]=k;

        for(unsigned int j=0; j<imsize[1];j++)
        {
            index[1]=j;
            for(unsigned int i=0; i<imsize[0];i++)
            {
                index[0]=i;

                DisplacementFieldType::PointType pf;
                combined_displacement_field->TransformIndexToPhysicalPoint(index,pf);
                DisplacementFieldType::PointType pmt = composite_trans->TransformPoint(pf);
                DisplacementFieldType::PixelType vec = pmt-pf;
                combined_displacement_field->SetPixel(index,vec);
            }
        }
    }
    return combined_displacement_field;
}


JacobianImageType::Pointer  GimmeNewJacobian(DisplacementFieldType::Pointer field, JacobianImageType::Pointer corr_term_img)
{
    JacobianImageType::Pointer new_JAC_img=JacobianImageType::New();
    new_JAC_img->SetRegions(field->GetLargestPossibleRegion());
    new_JAC_img->Allocate();
    new_JAC_img->SetDirection(field->GetDirection());
    new_JAC_img->SetOrigin(field->GetOrigin());
    new_JAC_img->SetSpacing(field->GetSpacing());
    JacobianImageType::PixelType id;
    id.set_identity();
    new_JAC_img->FillBuffer(id);

    itk::ImageRegionIteratorWithIndex<JacobianImageType> it(new_JAC_img,new_JAC_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        JacobianImageType::PixelType mat1 = ComputeJacobian(field,ind3);
        JacobianImageType::PixelType mat2= corr_term_img->GetPixel(ind3);
        auto mat = mat1*mat2;

        it.Set(mat);
    }
    return new_JAC_img;
}


ImageType3D::Pointer  MaskDilate(ImageType3D::Pointer template_img)
{
    ImageType3D::Pointer mask_img=ImageType3D::New();
    mask_img->SetRegions(template_img->GetLargestPossibleRegion());
    mask_img->Allocate();
    mask_img->SetDirection(template_img->GetDirection());
    mask_img->SetSpacing(template_img->GetSpacing());
    mask_img->SetOrigin(template_img->GetOrigin());
    mask_img->FillBuffer(0);

    int rad=5;
    ImageType3D::SizeType sz = mask_img->GetLargestPossibleRegion().GetSize();

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(mask_img,mask_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        if(template_img->GetPixel(ind3)>1)
        {
            ImageType3D::IndexType tind;
            for(int k=ind3[2]-rad;k<=ind3[2]+rad;k++)
            {
                tind[2]=k;
                if(tind[2]<0)
                    tind[2]=0;
                if(tind[2]>sz[2]-1)
                    tind[2]=sz[2]-1;

                for(int j=ind3[1]-rad;j<=ind3[1]+rad;j++)
                {
                    tind[1]=j;
                    if(tind[1]<0)
                        tind[1]=0;
                    if(tind[1]>sz[1]-1)
                        tind[1]=sz[1]-1;
                    for(int i=ind3[0]-rad;i<=ind3[0]+rad;i++)
                    {
                        tind[0]=i;
                        if(tind[0]<0)
                            tind[0]=0;
                        if(tind[0]>sz[0]-1)
                            tind[0]=sz[0]-1;
                        mask_img->SetPixel(tind,1);
                    }
                }
            }
        }
    }
    return mask_img;
}



int main( int argc , char * argv[] )
{
    if(argc<2)
    {
        std::cout<<"Usage:   ConstrainDefFields   full_path_to_textfile_containing_list_of_deformation_fields full_path_to_tensor_template "<<std::endl;
        return EXIT_FAILURE;
    }

    ifstream inFile(argv[1]);
    if (!inFile)
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return EXIT_FAILURE;
    }

    ifstream inFile2(argv[2]);
    if (!inFile2)
    {
        cerr << "Template file " << argv[2] << " not found." << endl;
        return EXIT_FAILURE;
    }

    using ReaderType3D=itk::ImageFileReader<ImageType3D>;
    ReaderType3D::Pointer readt=ReaderType3D::New();
    readt->SetFileName(argv[2]);
    readt->Update();
    ImageType3D::Pointer template_img=readt->GetOutput();
    //ImageType3D::Pointer template_img= readImageD<ImageType3D>(argv[2]);
  //  ImageType3D::Pointer mask_img= MaskDilate(template_img);


    std::vector<DisplacementFieldType::Pointer> fields;

    std::string currdir;
    std::string nm(argv[1]);
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else
        currdir= nm.substr(0,mypos+1);

    std::vector<std::string> filenames;

    bool write_avg_jac=false;
    if(argc>3 && atoi(argv[3])==1)
        write_avg_jac=true;

    std::cout<<"Reading fields..."<<std::endl;
    string line;
    while (getline(inFile, line))
    {
        if (line.empty())
            continue;

        std::string file_name=line;
        FILE * fp= fopen(file_name.c_str(),"rb");

        if(!fp)
        {
            file_name= currdir + file_name;

            FILE * fp2= fopen(file_name.c_str(),"rb");
            if(!fp2)
            {
                std::cout<< "File " << line << " does not exist. Exiting!" << std::endl;
                return 0;
            }
            else
                fclose(fp2);
        }
        else
            fclose(fp);


        using ReaderTypeF=itk::ImageFileReader<DisplacementFieldType>;
        ReaderTypeF::Pointer readf=ReaderTypeF::New();
        readf->SetFileName(file_name);
        readf->Update();
        DisplacementFieldType::Pointer curr_field= readf->GetOutput();

        //DisplacementFieldType::Pointer curr_field= readImageD<DisplacementFieldType>(file_name);
        FixBoundaries<DisplacementFieldType>(curr_field);

        fields.push_back(curr_field);
        filenames.push_back(file_name);
    }
    inFile.close();
    std::cout<<"Done reading fields!"<<std::endl;



    DisplacementFieldType::Pointer avg_field= AverageFields(fields);
    DisplacementFieldType::Pointer avg_field_inv= InvertDisplacementField(avg_field);
    DisplacementFieldType::Pointer guess= avg_field_inv;

    JacobianImageType::Pointer avg_affine_inv_img=JacobianImageType::New();
    avg_affine_inv_img->SetRegions(fields[0]->GetLargestPossibleRegion());
    avg_affine_inv_img->Allocate();
    avg_affine_inv_img->SetDirection(fields[0]->GetDirection());
    avg_affine_inv_img->SetOrigin(fields[0]->GetOrigin());
    avg_affine_inv_img->SetSpacing(fields[0]->GetSpacing());
    JacobianImageType::PixelType id;
    id.set_identity();
    avg_affine_inv_img->FillBuffer(id);

    std::cout<<"Computing voxelwise average Jacobian image..."<<std::endl;

    DisplacementFieldType::SizeType sz= fields[0]->GetLargestPossibleRegion().GetSize();

    #pragma omp parallel for
    for(int k=1;k<sz[2]-1;k++)
    {
        DisplacementFieldType::IndexType ind3;
        ind3[2]=k;
        for(int j=1;j<sz[1]-1;j++)
        {
            ind3[1]=j;
            for(int i=1;i<sz[0]-1;i++)
            {
                ind3[0]=i;

                std::vector< vnl_matrix_fixed<float,3,3> > all_affines;
                for(int v=0;v<fields.size();v++)
                {
                    vnl_matrix_fixed<float,3,3> curr_JAC= ComputeJacobian(fields[v],ind3);
                    all_affines.push_back(curr_JAC);
                }

                vnl_matrix_fixed<float,3,3> avg_affine_inv = AverageAffinesReturnInv(all_affines);
                avg_affine_inv_img->SetPixel(ind3,avg_affine_inv);
            }
        }
    }

    if(write_avg_jac)
    {
        using JacobianPixelTypeVec = itk::Vector<float,9>;
        using JacobianImageTypeVec= itk::Image<JacobianPixelTypeVec,3>;

        JacobianPixelTypeVec zervec; zervec.Fill(0);

        JacobianImageTypeVec::Pointer avg_jac_img= JacobianImageTypeVec::New();
        avg_jac_img->SetRegions(avg_affine_inv_img->GetLargestPossibleRegion());
        avg_jac_img->Allocate();
        avg_jac_img->SetDirection(avg_affine_inv_img->GetDirection());
        avg_jac_img->SetOrigin(avg_affine_inv_img->GetOrigin());
        avg_jac_img->SetSpacing(avg_affine_inv_img->GetSpacing());
        avg_jac_img->FillBuffer(zervec);

        itk::ImageRegionIteratorWithIndex<JacobianImageTypeVec> mmiit(avg_jac_img,avg_jac_img->GetLargestPossibleRegion());
        for(mmiit.GoToBegin();!mmiit.IsAtEnd();++mmiit)
        {
            ImageType3D::IndexType ind3= mmiit.GetIndex();

            JacobianPixelTypeVec vec;
            vnl_matrix_fixed<float,3,3> JAC = avg_affine_inv_img->GetPixel(ind3);

            vec[0]=JAC(0,0);vec[1]=JAC(0,1);vec[2]=JAC(0,2);
            vec[3]=JAC(1,0);vec[4]=JAC(1,1);vec[5]=JAC(1,2);
            vec[6]=JAC(2,0);vec[7]=JAC(2,1);vec[8]=JAC(2,2);

            avg_jac_img->SetPixel(ind3,vec);
        }

        std::string iname=argv[1];
        std::string oname = iname.substr(0,iname.rfind(".txt")) + "_avg_JACMAT.nii";

        using JacWriterType = itk::ImageFileWriter<JacobianImageTypeVec>;
        JacWriterType::Pointer wr=JacWriterType::New();
        wr->SetFileName(oname);
        wr->SetInput(avg_jac_img);
        wr->Update();

    }

    auto bb= CreateFieldOutOfJacobian(avg_affine_inv_img,avg_field_inv);
    std::string otxtnm= argv[1];
    otxtnm= otxtnm.substr(0,otxtnm.rfind(".txt"))+"_avgfield.nii";
    //writeImageD<DisplacementFieldType>(bb,otxtnm);

    using WriterTypeF = itk::ImageFileWriter<DisplacementFieldType>;
    WriterTypeF::Pointer writerf= WriterTypeF::New();
    writerf->SetInput(bb);
    writerf->SetFileName(otxtnm);
    writerf->Update();


    while(fields.size()>0)
    {
        DisplacementFieldType::Pointer curr_field = fields.back();
        std::string curr_name = filenames.back();

        JacobianImageType::Pointer new_field_jacobian = GimmeNewJacobian(curr_field,avg_affine_inv_img);
        curr_field = CreateFieldOutOfJacobian(new_field_jacobian,curr_field);

        std::string oname = curr_name.substr(0,curr_name.rfind(".nii"))+ "_cnstr.nii";
        //writeImageD<DisplacementFieldType>(curr_field,oname);

        WriterTypeF::Pointer writerf= WriterTypeF::New();
        writerf->SetInput(curr_field);
        writerf->SetFileName(oname);
        writerf->Update();

        fields.pop_back();
        filenames.pop_back();
    }

    return EXIT_SUCCESS;
}
