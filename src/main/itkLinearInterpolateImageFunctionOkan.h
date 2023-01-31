/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkLinearInterpolateImageFunctionOkan_h
#define itkLinearInterpolateImageFunctionOkan_h

#include "itkInterpolateImageFunction.h"
#include "itkVariableLengthVector.h"

namespace itk
{
/**
 *\class LinearInterpolateImageFunctionOkan
 * \brief Linearly interpolate an image at specified positions.
 *
 * LinearInterpolateImageFunctionOkan linearly interpolates image intensity at
 * a non-integer pixel position. This class is templated
 * over the input image type and the coordinate representation type
 * (e.g. float or double).
 *
 * This function works for N-dimensional images.
 *
 * This function works for images with scalar and vector pixel
 * types, and for images of type VectorImage.
 *
 * \sa VectorLinearInterpolateImageFunctionOkan
 *
 * \ingroup ImageFunctions ImageInterpolators
 * \ingroup ITKImageFunction
 *
 * \sphinx
 * \sphinxexample{Core/ImageFunction/LinearlyInterpolatePositionInImage,Linearly Interpolate Position In Image}
 * \endsphinx
 */
template <typename TInputImage, typename TCoordRep = double>
class ITK_TEMPLATE_EXPORT LinearInterpolateImageFunctionOkan : public InterpolateImageFunction<TInputImage, TCoordRep>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(LinearInterpolateImageFunctionOkan);

  /** Standard class type aliases. */
  using Self = LinearInterpolateImageFunctionOkan;
  using Superclass = InterpolateImageFunction<TInputImage, TCoordRep>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(LinearInterpolateImageFunctionOkan, InterpolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType type alias support */
  using typename Superclass::OutputType;

  /** InputImageType type alias support */
  using typename Superclass::InputImageType;

  /** InputPixelType type alias support */
  using typename Superclass::InputPixelType;

  /** RealType type alias support */
  using typename Superclass::RealType;

  /** Dimension underlying input image. */
  static constexpr unsigned int ImageDimension = Superclass::ImageDimension;

  /** Index type alias support */
  using typename Superclass::IndexType;

  /** Size type alias support */
  using typename Superclass::SizeType;

  /** ContinuousIndex type alias support */
  using typename Superclass::ContinuousIndexType;
  using InternalComputationType = typename ContinuousIndexType::ValueType;

  /** Evaluate the function at a ContinuousIndex position
   *
   * Returns the linearly interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method. */
  OutputType
  EvaluateAtContinuousIndex(const ContinuousIndexType & index) const override
  {
    return this->EvaluateOptimized(Dispatch<ImageDimension>(), index);
  }

  SizeType
  GetRadius() const override
  {
    return SizeType::Filled(1);
  }
  
    void SetMaskImage(ImageType3D::Pointer img){mask_img=img;}
  ImageType3D::Pointer GetMaskImage(){return mask_img;}

protected:
  LinearInterpolateImageFunctionOkan() = default;
  ~LinearInterpolateImageFunctionOkan() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  struct DispatchBase
  {};
  template <unsigned int>
  struct Dispatch : public DispatchBase
  {};
  
  
  ImageType3D::Pointer  mask_img{nullptr};

  inline OutputType
  EvaluateOptimized(const Dispatch<0> &, const ContinuousIndexType &) const
  {
    return 0;
  }

public:
  OutputType
  EvaluateOkan(const typename  Superclass::PointType & point) const
  {
      const ContinuousIndexType index = this->GetInputImage()->template TransformPhysicalPointToContinuousIndex<TCoordRep>(point);
      if(!mask_img)
          return (this->EvaluateAtContinuousIndex(index));

      const TInputImage * const inputImagePtr = this->GetInputImage();

      IndexType basei;
      basei[0] = Math::Floor<IndexValueType>(index[0]);
      if (basei[0] < this->m_StartIndex[0])
      {
        basei[0] = this->m_StartIndex[0];
      }
      basei[1] = Math::Floor<IndexValueType>(index[1]);
      if (basei[1] < this->m_StartIndex[1])
      {
        basei[1] = this->m_StartIndex[1];
      }
      basei[2] = Math::Floor<IndexValueType>(index[2]);
      if (basei[2] < this->m_StartIndex[2])
      {
        basei[2] = this->m_StartIndex[2];
      }

      double xd = index[0] - static_cast<InternalComputationType>(basei[0]);
      double yd = index[1] - static_cast<InternalComputationType>(basei[1]);
      double zd = index[2] - static_cast<InternalComputationType>(basei[2]);


      float w[8], v[8];
      bool m[8];

      /*float w000 = (1-xd)*(1-yd)*(1-zd);
      float w100 = (xd)*(1-yd)*(1-zd);
      float w010 = (1-xd)*(yd)*(1-zd);
      float w110 = (xd)*(yd)*(1-zd);
      float w001 = (1-xd)*(1-yd)*(zd);
      float w101 = (xd)*(1-yd)*(zd);
      float w011 = (1-xd)*(yd)*(zd);
      float w111 = (xd)*(yd)*(zd);
      */
      w[0] = (1-xd)*(1-yd)*(1-zd);
      w[1] = (xd)*(1-yd)*(1-zd);
      w[2] = (1-xd)*(yd)*(1-zd);
      w[3] = (xd)*(yd)*(1-zd);
      w[4] = (1-xd)*(1-yd)*(zd);
      w[5] = (xd)*(1-yd)*(zd);
      w[6] = (1-xd)*(yd)*(zd);
      w[7] = (xd)*(yd)*(zd);

      int cnt=0;
      IndexType cind;
      for(int kk=0;kk<2;kk++)
      {

          cind[2]=basei[2]+kk;

          if( kk==1 && (cind[2] >  this->m_EndIndex[2]))
          {
              v[4]=v[0];
              v[5]=v[1];
              v[6]=v[2];
              v[7]=v[3];
              v[4]=m[0];
              m[5]=m[1];
              m[6]=m[2];
              m[7]=m[3];
          }
          else
          {
              for(int jj=0;jj<2;jj++)
              {
                  cind[1]=basei[1]+jj;
                  if( jj==1 && (cind[1] >  this->m_EndIndex[1]))
                  {
                      v[2]=v[0];
                      v[3]=v[1];
                      m[2]=m[0];
                      m[3]=m[1];
                  }
                  else
                  {
                      for(int ii=0;ii<2;ii++)
                      {
                          cind[0]=basei[0]+ii;

                          if( ii==1 && (cind[0] >  this->m_EndIndex[0]))
                          {
                              v[1]=v[0];
                              m[1]=m[0];
                          }
                          else
                          {
                              v[cnt]= inputImagePtr->GetPixel(cind);
                              m[cnt]= (this->mask_img->GetPixel(cind) >0.1);
                          }

                          cnt++;
                      }

                  }
              }
          }
      }

      double tot_w=0;
      float val=0;
      for(int nn=0;nn<8;nn++)
      {
          val+= v[nn]*m[nn]*w[nn];
          tot_w+=m[nn]*w[nn];

      }
      if(tot_w==0)
          return v[0];
      else
          return val/tot_w;


  }
private:

  inline OutputType
  EvaluateOptimized(const Dispatch<1> &, const ContinuousIndexType & index) const
  {
    IndexType basei;
    basei[0] = Math::Floor<IndexValueType>(index[0]);
    if (basei[0] < this->m_StartIndex[0])
    {
      basei[0] = this->m_StartIndex[0];
    }

    const InternalComputationType & distance = index[0] - static_cast<InternalComputationType>(basei[0]);

    const TInputImage * const inputImagePtr = this->GetInputImage();
    const RealType &          val0 = inputImagePtr->GetPixel(basei);
    if (distance <= 0.)
    {
      return (static_cast<OutputType>(val0));
    }

    ++basei[0];
    if (basei[0] > this->m_EndIndex[0])
    {
      return (static_cast<OutputType>(val0));
    }
    const RealType & val1 = inputImagePtr->GetPixel(basei);

    return (static_cast<OutputType>(val0 + (val1 - val0) * distance));
  }

  inline OutputType
  EvaluateOptimized(const Dispatch<2> &, const ContinuousIndexType & index) const
  {
    IndexType basei;

    basei[0] = Math::Floor<IndexValueType>(index[0]);
    if (basei[0] < this->m_StartIndex[0])
    {
      basei[0] = this->m_StartIndex[0];
    }
    const InternalComputationType & distance0 = index[0] - static_cast<InternalComputationType>(basei[0]);

    basei[1] = Math::Floor<IndexValueType>(index[1]);
    if (basei[1] < this->m_StartIndex[1])
    {
      basei[1] = this->m_StartIndex[1];
    }
    const InternalComputationType & distance1 = index[1] - static_cast<InternalComputationType>(basei[1]);

    const TInputImage * const inputImagePtr = this->GetInputImage();
    const RealType &          val00 = inputImagePtr->GetPixel(basei);
    if (distance0 <= 0. && distance1 <= 0.)
    {
      return (static_cast<OutputType>(val00));
    }
    else if (distance1 <= 0.) // if they have the same "y"
    {
      ++basei[0]; // then interpolate across "x"
      if (basei[0] > this->m_EndIndex[0])
      {
        return (static_cast<OutputType>(val00));
      }
      const RealType & val10 = inputImagePtr->GetPixel(basei);
      return (static_cast<OutputType>(val00 + (val10 - val00) * distance0));
    }
    else if (distance0 <= 0.) // if they have the same "x"
    {
      ++basei[1]; // then interpolate across "y"
      if (basei[1] > this->m_EndIndex[1])
      {
        return (static_cast<OutputType>(val00));
      }
      const RealType & val01 = inputImagePtr->GetPixel(basei);
      return (static_cast<OutputType>(val00 + (val01 - val00) * distance1));
    }
    // fall-through case:
    // interpolate across "xy"
    ++basei[0];
    if (basei[0] > this->m_EndIndex[0]) // interpolate across "y"
    {
      --basei[0];
      ++basei[1];
      if (basei[1] > this->m_EndIndex[1])
      {
        return (static_cast<OutputType>(val00));
      }
      const RealType & val01 = inputImagePtr->GetPixel(basei);
      return (static_cast<OutputType>(val00 + (val01 - val00) * distance1));
    }
    const RealType & val10 = inputImagePtr->GetPixel(basei);

    const RealType & valx0 = val00 + (val10 - val00) * distance0;

    ++basei[1];
    if (basei[1] > this->m_EndIndex[1]) // interpolate across "x"
    {
      return (static_cast<OutputType>(valx0));
    }
    const RealType & val11 = inputImagePtr->GetPixel(basei);
    --basei[0];
    const RealType & val01 = inputImagePtr->GetPixel(basei);

    const RealType & valx1 = val01 + (val11 - val01) * distance0;

    return (static_cast<OutputType>(valx0 + (valx1 - valx0) * distance1));
  }

  inline OutputType
  EvaluateOptimized(const Dispatch<3> &, const ContinuousIndexType & index) const
  {
    IndexType basei;
    basei[0] = Math::Floor<IndexValueType>(index[0]);
    if (basei[0] < this->m_StartIndex[0])
    {
      basei[0] = this->m_StartIndex[0];
    }
    const InternalComputationType & distance0 = index[0] - static_cast<InternalComputationType>(basei[0]);

    basei[1] = Math::Floor<IndexValueType>(index[1]);
    if (basei[1] < this->m_StartIndex[1])
    {
      basei[1] = this->m_StartIndex[1];
    }
    const InternalComputationType & distance1 = index[1] - static_cast<InternalComputationType>(basei[1]);

    basei[2] = Math::Floor<IndexValueType>(index[2]);
    if (basei[2] < this->m_StartIndex[2])
    {
      basei[2] = this->m_StartIndex[2];
    }
    const InternalComputationType & distance2 = index[2] - static_cast<InternalComputationType>(basei[2]);

    const TInputImage * const inputImagePtr = this->GetInputImage();
    const RealType &          val000 = inputImagePtr->GetPixel(basei);
    if (distance0 <= 0. && distance1 <= 0. && distance2 <= 0.)
    {
      return (static_cast<OutputType>(val000));
    }

    if (distance2 <= 0.)
    {
      if (distance1 <= 0.) // interpolate across "x"
      {
        ++basei[0];
        if (basei[0] > this->m_EndIndex[0])
        {
          return (static_cast<OutputType>(val000));
        }
        const RealType & val100 = inputImagePtr->GetPixel(basei);

        return static_cast<OutputType>(val000 + (val100 - val000) * distance0);
      }
      else if (distance0 <= 0.) // interpolate across "y"
      {
        ++basei[1];
        if (basei[1] > this->m_EndIndex[1])
        {
          return (static_cast<OutputType>(val000));
        }
        const RealType & val010 = inputImagePtr->GetPixel(basei);

        return static_cast<OutputType>(val000 + (val010 - val000) * distance1);
      }
      else // interpolate across "xy"
      {
        ++basei[0];
        if (basei[0] > this->m_EndIndex[0]) // interpolate across "y"
        {
          --basei[0];
          ++basei[1];
          if (basei[1] > this->m_EndIndex[1])
          {
            return (static_cast<OutputType>(val000));
          }
          const RealType & val010 = inputImagePtr->GetPixel(basei);
          return static_cast<OutputType>(val000 + (val010 - val000) * distance1);
        }
        const RealType & val100 = inputImagePtr->GetPixel(basei);
        const RealType & valx00 = val000 + (val100 - val000) * distance0;

        ++basei[1];
        if (basei[1] > this->m_EndIndex[1]) // interpolate across "x"
        {
          return (static_cast<OutputType>(valx00));
        }
        const RealType & val110 = inputImagePtr->GetPixel(basei);

        --basei[0];
        const RealType & val010 = inputImagePtr->GetPixel(basei);
        const RealType & valx10 = val010 + (val110 - val010) * distance0;

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
          if (basei[2] > this->m_EndIndex[2])
          {
            return (static_cast<OutputType>(val000));
          }
          const RealType & val001 = inputImagePtr->GetPixel(basei);

          return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
        }
        else // interpolate across "xz"
        {
          ++basei[0];
          if (basei[0] > this->m_EndIndex[0]) // interpolate across "z"
          {
            --basei[0];
            ++basei[2];
            if (basei[2] > this->m_EndIndex[2])
            {
              return (static_cast<OutputType>(val000));
            }
            const RealType & val001 = inputImagePtr->GetPixel(basei);

            return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
          }
          const RealType & val100 = inputImagePtr->GetPixel(basei);

          const RealType & valx00 = val000 + (val100 - val000) * distance0;

          ++basei[2];
          if (basei[2] > this->m_EndIndex[2]) // interpolate across "x"
          {
            return (static_cast<OutputType>(valx00));
          }
          const RealType & val101 = inputImagePtr->GetPixel(basei);

          --basei[0];
          const RealType & val001 = inputImagePtr->GetPixel(basei);

          const RealType & valx01 = val001 + (val101 - val001) * distance0;

          return static_cast<OutputType>(valx00 + (valx01 - valx00) * distance2);
        }
      }
      else if (distance0 <= 0.) // interpolate across "yz"
      {
        ++basei[1];
        if (basei[1] > this->m_EndIndex[1]) // interpolate across "z"
        {
          --basei[1];
          ++basei[2];
          if (basei[2] > this->m_EndIndex[2])
          {
            return (static_cast<OutputType>(val000));
          }
          const RealType & val001 = inputImagePtr->GetPixel(basei);

          return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
        }
        const RealType & val010 = inputImagePtr->GetPixel(basei);

        const RealType & val0x0 = val000 + (val010 - val000) * distance1;

        ++basei[2];
        if (basei[2] > this->m_EndIndex[2]) // interpolate across "y"
        {
          return (static_cast<OutputType>(val0x0));
        }
        const RealType & val011 = inputImagePtr->GetPixel(basei);

        --basei[1];
        const RealType & val001 = inputImagePtr->GetPixel(basei);

        const RealType & val0x1 = val001 + (val011 - val001) * distance1;

        return static_cast<OutputType>(val0x0 + (val0x1 - val0x0) * distance2);
      }
      else // interpolate across "xyz"
      {
        ++basei[0];
        if (basei[0] > this->m_EndIndex[0]) // interpolate across "yz"
        {
          --basei[0];
          ++basei[1];
          if (basei[1] > this->m_EndIndex[1]) // interpolate across "z"
          {
            --basei[1];
            ++basei[2];
            if (basei[2] > this->m_EndIndex[2])
            {
              return (static_cast<OutputType>(val000));
            }
            const RealType & val001 = inputImagePtr->GetPixel(basei);

            return static_cast<OutputType>(val000 + (val001 - val000) * distance2);
          }
          const RealType & val010 = inputImagePtr->GetPixel(basei);
          const RealType & val0x0 = val000 + (val010 - val000) * distance1;

          ++basei[2];
          if (basei[2] > this->m_EndIndex[2]) // interpolate across "y"
          {
            return (static_cast<OutputType>(val0x0));
          }
          const RealType & val011 = inputImagePtr->GetPixel(basei);

          --basei[1];
          const RealType & val001 = inputImagePtr->GetPixel(basei);

          const RealType & val0x1 = val001 + (val011 - val001) * distance1;

          return static_cast<OutputType>(val0x0 + (val0x1 - val0x0) * distance2);
        }
        const RealType & val100 = inputImagePtr->GetPixel(basei);

        const RealType & valx00 = val000 + (val100 - val000) * distance0;

        ++basei[1];
        if (basei[1] > this->m_EndIndex[1]) // interpolate across "xz"
        {
          --basei[1];
          ++basei[2];
          if (basei[2] > this->m_EndIndex[2]) // interpolate across "x"
          {
            return (static_cast<OutputType>(valx00));
          }
          const RealType & val101 = inputImagePtr->GetPixel(basei);

          --basei[0];
          const RealType & val001 = inputImagePtr->GetPixel(basei);

          const RealType & valx01 = val001 + (val101 - val001) * distance0;

          return static_cast<OutputType>(valx00 + (valx01 - valx00) * distance2);
        }
        const RealType & val110 = inputImagePtr->GetPixel(basei);

        --basei[0];
        const RealType & val010 = inputImagePtr->GetPixel(basei);

        const RealType & valx10 = val010 + (val110 - val010) * distance0;

        const RealType & valxx0 = valx00 + (valx10 - valx00) * distance1;

        ++basei[2];
        if (basei[2] > this->m_EndIndex[2]) // interpolate across "xy"
        {
          return (static_cast<OutputType>(valxx0));
        }
        const RealType & val011 = inputImagePtr->GetPixel(basei);

        ++basei[0];
        const RealType & val111 = inputImagePtr->GetPixel(basei);

        --basei[1];
        const RealType & val101 = inputImagePtr->GetPixel(basei);

        --basei[0];
        const RealType & val001 = inputImagePtr->GetPixel(basei);

        const RealType & valx01 = val001 + (val101 - val001) * distance0;
        const RealType & valx11 = val011 + (val111 - val011) * distance0;
        const RealType & valxx1 = valx01 + (valx11 - valx01) * distance1;

        return (static_cast<OutputType>(valxx0 + (valxx1 - valxx0) * distance2));
      }
    }
  }

  inline OutputType
  EvaluateOptimized(const DispatchBase &, const ContinuousIndexType & index) const
  {
    return this->EvaluateUnoptimized(index);
  }

  /** Evaluate interpolator at image index position. */
  virtual inline OutputType
  EvaluateUnoptimized(const ContinuousIndexType & index) const;

  /** \brief A method to generically set all components to zero
   */
  template <typename RealTypeScalarRealType>
  void
  MakeZeroInitializer(const TInputImage * const                      inputImagePtr,
                      VariableLengthVector<RealTypeScalarRealType> & tempZeros) const
  {
    // Variable length vector version to get the size of the pixel correct.
    constexpr typename TInputImage::IndexType idx = { { 0 } };
    const typename TInputImage::PixelType &   tempPixel = inputImagePtr->GetPixel(idx);
    const unsigned int                        sizeOfVarLengthVector = tempPixel.GetSize();
    tempZeros.SetSize(sizeOfVarLengthVector);
    tempZeros.Fill(NumericTraits<RealTypeScalarRealType>::ZeroValue());
  }

  template <typename RealTypeScalarRealType>
  void
  MakeZeroInitializer(const TInputImage * const itkNotUsed(inputImagePtr), RealTypeScalarRealType & tempZeros) const
  {
    // All other cases
    tempZeros = NumericTraits<RealTypeScalarRealType>::ZeroValue();
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkLinearInterpolateImageFunctionOkan.hxx"
#endif

#endif

