#ifndef _DRTAMAS_UTILITIES_CP_CXX
#define _DRTAMAS_UTILITIES_CP_CXX



#include "DRTAMAS_utilities_cp.h"
#include "itkImageDuplicator.h"
#include <vnl/algo/vnl_real_eigensystem.h>
#include <vnl/vnl_real.h>
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"

#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"


#define LOGNEG -25


vnl_matrix_fixed<double,3,3> ComputeRotationFromAffine(vnl_matrix_fixed<double,3,3> A)
{
    auto AAT = A * A.transpose();

    vnl_symmetric_eigensystem<double> eig(AAT);


    eig.D(0,0)= pow(eig.D(0,0), -0.5);
    eig.D(1,1)= pow(eig.D(1,1), -0.5);
    eig.D(2,2)= pow(eig.D(2,2), -0.5);

    auto AAT_sq_inv = eig.recompose();

    return AAT_sq_inv * A;

}

DTMatrixImageType::Pointer LogTensorImage(DTMatrixImageType::Pointer dt_img)
{
    using DupType = itk::ImageDuplicator<DTMatrixImageType>;
    DupType::Pointer dup = DupType::New();
    dup->SetInputImage(dt_img);
    dup->Update();
    DTMatrixImageType::Pointer out_img = dup->GetOutput();

    DTMatrixImageType::SizeType sz= out_img->GetLargestPossibleRegion().GetSize();

    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        DTMatrixImageType::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;

                DTMatrixImageType::PixelType mat= out_img->GetPixel(ind3);

                vnl_symmetric_eigensystem<double> eig(mat);
                if(eig.D(0,0)<=0)
                    eig.D(0,0)=LOGNEG;
                else
                    eig.D(0,0)=std::log(eig.D(0,0));
                if(eig.D(1,1)<=0)
                    eig.D(1,1)=LOGNEG;
                else
                    eig.D(1,1)=std::log(eig.D(1,1));
                if(eig.D(2,2)<=0)
                    eig.D(2,2)=LOGNEG;
                else
                    eig.D(2,2)=std::log(eig.D(2,2));

                vnl_matrix_fixed<double,3,3> mat_corr= eig.recompose();
                out_img->SetPixel(ind3,mat_corr);
            }
        }
    }

    return out_img;
}


DTMatrixImageType::Pointer ExpTensorImage(DTMatrixImageType::Pointer dt_img)
{
    using DupType = itk::ImageDuplicator<DTMatrixImageType>;
    DupType::Pointer dup = DupType::New();
    dup->SetInputImage(dt_img);
    dup->Update();
    DTMatrixImageType::Pointer out_img = dup->GetOutput();

    DTMatrixImageType::SizeType sz= out_img->GetLargestPossibleRegion().GetSize();

  //  #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        DTMatrixImageType::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;

                if(ind3[0]==104 && ind3[1]==22 && ind3[2]==10)
                    int ma=0;

                DTMatrixImageType::PixelType mat= out_img->GetPixel(ind3);

                vnl_symmetric_eigensystem<double> eig(mat);
                eig.D(0,0)=std::exp(eig.D(0,0));
                eig.D(1,1)=std::exp(eig.D(1,1));
                eig.D(2,2)=std::exp(eig.D(2,2));

                vnl_matrix_fixed<double,3,3> mat_corr= eig.recompose();
                out_img->SetPixel(ind3,mat_corr);
            }
        }
    }

    return out_img;
}


InternalMatrixType  InterpolateAt(DTMatrixImageType::Pointer img, DTMatrixImageType::PointType pt)
{
    itk::ContinuousIndex<double,3> index;
    img->TransformPhysicalPointToContinuousIndex(pt,index);


    using IndexType=DTMatrixImageType::IndexType;
    using InternalComputationType =double;
    using IndexValueType= DTMatrixImageType::IndexValueType;
    using OutputType=InternalMatrixType;


    IndexType m_StartIndex;  m_StartIndex.Fill(0);
    IndexType m_EndIndex;
    m_EndIndex[0]=img->GetLargestPossibleRegion().GetSize()[0]-1;
    m_EndIndex[1]=img->GetLargestPossibleRegion().GetSize()[1]-1;
    m_EndIndex[2]=img->GetLargestPossibleRegion().GetSize()[2]-1;


    bool isinsidebuffer=true;
    InternalMatrixType mat; mat.fill(0);


    for (unsigned int j = 0; j < 3; ++j)
    {
      if (index[j] < m_StartIndex[j])
      {
        isinsidebuffer=false;
      }
      if (index[j] > m_EndIndex[j])
      {
        isinsidebuffer=false;
      }
    }

    if(!isinsidebuffer)
        return mat;


    IndexType basei;
    basei[0] = itk::Math::Floor<IndexValueType>(index[0]);
    if (basei[0] < m_StartIndex[0])
    {
      basei[0] = m_StartIndex[0];
    }
    const InternalComputationType & distance0 = index[0] - static_cast<InternalComputationType>(basei[0]);

    basei[1] = itk::Math::Floor<IndexValueType>(index[1]);
    if (basei[1] < m_StartIndex[1])
    {
      basei[1] = m_StartIndex[1];
    }
    const InternalComputationType & distance1 = index[1] - static_cast<InternalComputationType>(basei[1]);

    basei[2] = itk::Math::Floor<IndexValueType>(index[2]);
    if (basei[2] < m_StartIndex[2])
    {
      basei[2] = m_StartIndex[2];
    }
    const InternalComputationType & distance2 = index[2] - static_cast<InternalComputationType>(basei[2]);

    DTMatrixImageType::Pointer inputImagePtr = img;
    const InternalMatrixType &          val000 = inputImagePtr->GetPixel(basei);
    if (distance0 <= 0. && distance1 <= 0. && distance2 <= 0.)
    {
      return (static_cast<OutputType>(val000));
    }

    if (distance2 <= 0.)
    {
      if (distance1 <= 0.) // interpolate across "x"
      {
        ++basei[0];
        if (basei[0] > m_EndIndex[0])
        {
          return (static_cast<OutputType>(val000));
        }
        const InternalMatrixType & val100 = inputImagePtr->GetPixel(basei);

        return static_cast<OutputType>(val000 + (val100 - val000) * distance0);
      }
      else if (distance0 <= 0.) // interpolate across "y"
      {
        ++basei[1];
        if (basei[1] > m_EndIndex[1])
        {
          return (static_cast<OutputType>(val000));
        }
        const InternalMatrixType & val010 = inputImagePtr->GetPixel(basei);

        return static_cast<OutputType>(val000 + (val010 - val000) * distance1);
      }
      else // interpolate across "xy"
      {
        ++basei[0];
        if (basei[0] > m_EndIndex[0]) // interpolate across "y"
        {
          --basei[0];
          ++basei[1];
          if (basei[1] > m_EndIndex[1])
          {
            return (static_cast<OutputType>(val000));
          }
          const InternalMatrixType & val010 = inputImagePtr->GetPixel(basei);
          return static_cast<OutputType>(val000 + (val010 - val000) * distance1);
        }
        const InternalMatrixType & val100 = inputImagePtr->GetPixel(basei);
        const InternalMatrixType & valx00 = val000 + (val100 - val000) * distance0;

        ++basei[1];
        if (basei[1] > m_EndIndex[1]) // interpolate across "x"
        {
          return (static_cast<OutputType>(valx00));
        }
        const InternalMatrixType & val110 = inputImagePtr->GetPixel(basei);

        --basei[0];
        const InternalMatrixType & val010 = inputImagePtr->GetPixel(basei);
        const InternalMatrixType & valx10 = val010 + (val110 - val010) * distance0;

        return static_cast<OutputType>(valx00 + (valx10 - valx00) * distance1);
      }
    }
    else
    {
      if (distance1 <= 0.)
      {
        if (distance0 <= 0.) // interpolate across "z"
        {
          ++basei[2];
          if (basei[2] > m_EndIndex[2])
          {
            return (static_cast<OutputType>(val000));
          }
          const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

          return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
        }
        else // interpolate across "xz"
        {
          ++basei[0];
          if (basei[0] > m_EndIndex[0]) // interpolate across "z"
          {
            --basei[0];
            ++basei[2];
            if (basei[2] > m_EndIndex[2])
            {
              return (static_cast<OutputType>(val000));
            }
            const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

            return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
          }
          const InternalMatrixType & val100 = inputImagePtr->GetPixel(basei);

          const InternalMatrixType & valx00 = val000 + (val100 - val000) * distance0;

          ++basei[2];
          if (basei[2] > m_EndIndex[2]) // interpolate across "x"
          {
            return (static_cast<OutputType>(valx00));
          }
          const InternalMatrixType & val101 = inputImagePtr->GetPixel(basei);

          --basei[0];
          const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

          const InternalMatrixType & valx01 = val001 + (val101 - val001) * distance0;

          return static_cast<OutputType>(valx00 + (valx01 - valx00) * distance2);
        }
      }
      else if (distance0 <= 0.) // interpolate across "yz"
      {
        ++basei[1];
        if (basei[1] > m_EndIndex[1]) // interpolate across "z"
        {
          --basei[1];
          ++basei[2];
          if (basei[2] > m_EndIndex[2])
          {
            return (static_cast<OutputType>(val000));
          }
          const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

          return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
        }
        const InternalMatrixType & val010 = inputImagePtr->GetPixel(basei);

        const InternalMatrixType & val0x0 = val000 + (val010 - val000) * distance1;

        ++basei[2];
        if (basei[2] > m_EndIndex[2]) // interpolate across "y"
        {
          return (static_cast<OutputType>(val0x0));
        }
        const InternalMatrixType & val011 = inputImagePtr->GetPixel(basei);

        --basei[1];
        const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

        const InternalMatrixType & val0x1 = val001 + (val011 - val001) * distance1;

        return static_cast<OutputType>(val0x0 + (val0x1 - val0x0) * distance2);
      }
      else // interpolate across "xyz"
      {
        ++basei[0];
        if (basei[0] > m_EndIndex[0]) // interpolate across "yz"
        {
          --basei[0];
          ++basei[1];
          if (basei[1] > m_EndIndex[1]) // interpolate across "z"
          {
            --basei[1];
            ++basei[2];
            if (basei[2] > m_EndIndex[2])
            {
              return (static_cast<OutputType>(val000));
            }
            const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

            return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
          }
          const InternalMatrixType & val010 = inputImagePtr->GetPixel(basei);
          const InternalMatrixType & val0x0 = val000 + (val010 - val000) * distance1;

          ++basei[2];
          if (basei[2] > m_EndIndex[2]) // interpolate across "y"
          {
            return (static_cast<OutputType>(val0x0));
          }
          const InternalMatrixType & val011 = inputImagePtr->GetPixel(basei);

          --basei[1];
          const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

          const InternalMatrixType & val0x1 = val001 + (val011 - val001) * distance1;

          return static_cast<OutputType>(val0x0 + (val0x1 - val0x0) * distance2);
        }
        const InternalMatrixType & val100 = inputImagePtr->GetPixel(basei);

        const InternalMatrixType & valx00 = val000 + (val100 - val000) * distance0;

        ++basei[1];
        if (basei[1] > m_EndIndex[1]) // interpolate across "xz"
        {
          --basei[1];
          ++basei[2];
          if (basei[2] > m_EndIndex[2]) // interpolate across "x"
          {
            return (static_cast<OutputType>(valx00));
          }
          const InternalMatrixType & val101 = inputImagePtr->GetPixel(basei);

          --basei[0];
          const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

          const InternalMatrixType & valx01 = val001 + (val101 - val001) * distance0;

          return static_cast<OutputType>(valx00 + (valx01 - valx00) * distance2);
        }
        const InternalMatrixType & val110 = inputImagePtr->GetPixel(basei);

        --basei[0];
        const InternalMatrixType & val010 = inputImagePtr->GetPixel(basei);

        const InternalMatrixType & valx10 = val010 + (val110 - val010) * distance0;

        const InternalMatrixType & valxx0 = valx00 + (valx10 - valx00) * distance1;

        ++basei[2];
        if (basei[2] > m_EndIndex[2]) // interpolate across "xy"
        {
          return (static_cast<OutputType>(valxx0));
        }
        const InternalMatrixType & val011 = inputImagePtr->GetPixel(basei);

        ++basei[0];
        const InternalMatrixType & val111 = inputImagePtr->GetPixel(basei);

        --basei[1];
        const InternalMatrixType & val101 = inputImagePtr->GetPixel(basei);

        --basei[0];
        const InternalMatrixType & val001 = inputImagePtr->GetPixel(basei);

        const InternalMatrixType & valx01 = val001 + (val101 - val001) * distance0;
        const InternalMatrixType & valx11 = val011 + (val111 - val011) * distance0;
        const InternalMatrixType & valxx1 = valx01 + (valx11 - valx01) * distance1;

        return (static_cast<OutputType>(valxx0 + (valxx1 - valxx0) * distance2));
      }
    }
}



DTMatrixImageType::Pointer ReadAndOrientTensor(std::string fname)
{

    std::vector<ImageType3D::Pointer> tensorv;

    for(int v=0;v<6;v++)
    {
        ImageType3D::Pointer dt_comp= read_3D_volume_from_4D(fname,v);
        tensorv.push_back(dt_comp);
    }

    DTMatrixImageType::Pointer tensor_img = DTMatrixImageType::New();
    tensor_img->SetRegions(tensorv[0]->GetLargestPossibleRegion());
    tensor_img->Allocate();
    tensor_img->SetDirection(tensorv[0]->GetDirection());
    tensor_img->SetOrigin(tensorv[0]->GetOrigin());
    tensor_img->SetSpacing(tensorv[0]->GetSpacing());

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(tensorv[0],tensorv[0]->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();

        InternalMatrixType mat;
        mat(0,0)= tensorv[0]->GetPixel(ind3);
        mat(1,1)= tensorv[1]->GetPixel(ind3);
        mat(2,2)= tensorv[2]->GetPixel(ind3);
        mat(0,1)= tensorv[3]->GetPixel(ind3);
        mat(1,0)= tensorv[3]->GetPixel(ind3);
        mat(0,2)= tensorv[4]->GetPixel(ind3);
        mat(2,0)= tensorv[4]->GetPixel(ind3);
        mat(2,1)= tensorv[5]->GetPixel(ind3);
        mat(1,2)= tensorv[5]->GetPixel(ind3);

        auto mat2= tensorv[0]->GetDirection().GetVnlMatrix() * mat * tensorv[0]->GetDirection().GetTranspose();

        tensor_img->SetPixel(ind3,mat2);

    }

    return tensor_img;

}

DTMatrixImageType::Pointer TransformAndWriteAffineImage(DTMatrixImageType::Pointer moving_tensor,DRTAMAS::AffineTransformType::Pointer my_affine_trans, DTMatrixImageType::Pointer fixed_tensor, std::string output_nii_name)
{
    using DupType=itk::ImageDuplicator<DTMatrixImageType>;
    DupType::Pointer dup = DupType::New();
    dup->SetInputImage(fixed_tensor);
    dup->Update();
    auto moving_tensor_aff=dup->GetOutput();

    DTMatrixImageType::PixelType small_tens; small_tens.fill(0);small_tens.fill_diagonal(-1E10);
    moving_tensor_aff->FillBuffer(small_tens);


    DTMatrixImageType::Pointer log_moving_tensor = LogTensorImage(moving_tensor);

    auto A = my_affine_trans->GetMatrix().GetVnlMatrix();    
    auto R =  ComputeRotationFromAffine(A);


    DTMatrixImageType::SizeType sz= moving_tensor_aff->GetLargestPossibleRegion().GetSize();
    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        DTMatrixImageType::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;

                ImageType3D::PointType pt,pt_trans;
                fixed_tensor->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans= my_affine_trans->TransformPoint(pt);

                InternalMatrixType mat_trans = InterpolateAt(log_moving_tensor, pt_trans);
                mat_trans = R.transpose() *  mat_trans * R;

                moving_tensor_aff->SetPixel(ind3,mat_trans);
            }
        }
    }

    auto moving_tensor_aff2=ExpTensorImage(moving_tensor_aff);

    std::string moving_aff_nii_name = output_nii_name;
    std::string moving_aff_trans_name = output_nii_name.substr(0,output_nii_name.rfind(".nii")) + ".txt";

    OrientAndWriteTensor(moving_tensor_aff2,moving_aff_nii_name);


    itk::TransformFileWriter::Pointer trwriter = itk::TransformFileWriter::New();
    trwriter->SetInput(my_affine_trans);
    trwriter->SetFileName(moving_aff_trans_name);
    trwriter->Update();

    return moving_tensor_aff2;
}


void TransformAndWriteDiffeoImage(DTMatrixImageType::Pointer moving_tensor,DisplacementFieldType::Pointer disp_field,DTMatrixImageType::Pointer fixed_tensor, std::string output_nii_name)
{
    using DupType=itk::ImageDuplicator<DTMatrixImageType>;
    DupType::Pointer dup = DupType::New();
    dup->SetInputImage(fixed_tensor);
    dup->Update();
    DTMatrixImageType::Pointer moving_tensor_diffeo=dup->GetOutput();


    DTMatrixImageType::Pointer log_moving_tensor = LogTensorImage(moving_tensor);

    using DisplacementFieldTransformType = DRTAMAS::DisplacementFieldTransformType;
    DisplacementFieldTransformType::Pointer disp_trans= DisplacementFieldTransformType::New();
    disp_trans->SetDisplacementField(disp_field);


    DTMatrixImageType::SizeType sz= moving_tensor_diffeo->GetLargestPossibleRegion().GetSize();
    #pragma omp parallel for
    for(int k=0;k<sz[2];k++)
    {
        DTMatrixImageType::IndexType ind3;
        ind3[2]=k;
        for(int j=0;j<sz[1];j++)
        {
            ind3[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;

                ImageType3D::PointType pt,pt_trans;
                fixed_tensor->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans= disp_trans->TransformPoint(pt);

                auto J = ComputeJacobian(disp_field,ind3) ;
                auto R =  ComputeRotationFromAffine(J);

                InternalMatrixType mat_trans = InterpolateAt(log_moving_tensor, pt_trans);
                mat_trans = R.transpose() *  mat_trans * R;

                moving_tensor_diffeo->SetPixel(ind3,mat_trans);
            }
        }
    }

    moving_tensor_diffeo=ExpTensorImage(moving_tensor_diffeo);


    std::vector<ImageType3D::Pointer> dt_diffeo_imgs;
    dt_diffeo_imgs.resize(6);
    for(int v=0;v<6;v++)
    {
        ImageType3D::Pointer img = ImageType3D::New();
        img->SetRegions(fixed_tensor->GetLargestPossibleRegion());
        img->Allocate();
        img->SetSpacing(fixed_tensor->GetSpacing());
        img->SetDirection(fixed_tensor->GetDirection());
        img->SetOrigin(fixed_tensor->GetOrigin());
        dt_diffeo_imgs[v]=img;
    }


    itk::ImageRegionIteratorWithIndex<DTMatrixImageType> it(moving_tensor_diffeo,moving_tensor_diffeo->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3 = it.GetIndex();
        InternalMatrixType mat=it.Get();

        mat = moving_tensor_diffeo->GetDirection().GetTranspose() * mat * moving_tensor_diffeo->GetDirection().GetVnlMatrix();
        dt_diffeo_imgs[0]->SetPixel(ind3, mat(0,0));
        dt_diffeo_imgs[1]->SetPixel(ind3, mat(1,1));
        dt_diffeo_imgs[2]->SetPixel(ind3, mat(2,2));
        dt_diffeo_imgs[3]->SetPixel(ind3, mat(0,1));
        dt_diffeo_imgs[4]->SetPixel(ind3, mat(0,2));
        dt_diffeo_imgs[5]->SetPixel(ind3, mat(1,2));
    }


    std::string moving_diffeo_nii_name = output_nii_name;

    for(int v=0;v<6;v++)
    {
        write_3D_image_to_4D_file<float>(dt_diffeo_imgs[v],moving_diffeo_nii_name,v,6);
    }
}


vnl_matrix_fixed<double,3,3> ComputeJacobian(DisplacementFieldType::Pointer field,DisplacementFieldType::IndexType ind3 )
{
    vnl_matrix_fixed<double,3,3> A;
    A.set_identity();

    ImageType3D::SizeType sz= field->GetLargestPossibleRegion().GetSize();

    if(ind3[0]<1 || ind3[0]>sz[0]-2 || ind3[1]<1 || ind3[1]>sz[1]-2 || ind3[2]<1 || ind3[2]>sz[2]-2 )
        return A;


    ImageType3D::SpacingType spc = field->GetSpacing();
    ImageType3D::DirectionType dir = field->GetDirection();

    vnl_matrix_fixed<double,3,3> SD;
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

        auto diff = 0.5*(vecp-vecm);

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

void OrientAndWriteTensor(DTMatrixImageType::Pointer tens, std::string nm)
{
    std::vector<ImageType3D::Pointer> dt_aff_imgs;
    dt_aff_imgs.resize(6);
    for(int v=0;v<6;v++)
    {
        ImageType3D::Pointer img = ImageType3D::New();
        img->SetRegions(tens->GetLargestPossibleRegion());
        img->Allocate();
        img->SetSpacing(tens->GetSpacing());
        img->SetDirection(tens->GetDirection());
        img->SetOrigin(tens->GetOrigin());
        dt_aff_imgs[v]=img;
    }


    itk::ImageRegionIteratorWithIndex<DTMatrixImageType> it(tens,tens->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3 = it.GetIndex();
        InternalMatrixType mat=it.Get();

        mat = tens->GetDirection().GetTranspose() * mat * tens->GetDirection().GetVnlMatrix();
        dt_aff_imgs[0]->SetPixel(ind3, mat(0,0));
        dt_aff_imgs[1]->SetPixel(ind3, mat(1,1));
        dt_aff_imgs[2]->SetPixel(ind3, mat(2,2));
        dt_aff_imgs[3]->SetPixel(ind3, mat(0,1));
        dt_aff_imgs[4]->SetPixel(ind3, mat(0,2));
        dt_aff_imgs[5]->SetPixel(ind3, mat(1,2));
    }


    for(int v=0;v<6;v++)
    {
        write_3D_image_to_4D_file<float>(dt_aff_imgs[v],nm,v,6);
    }

}



#endif

