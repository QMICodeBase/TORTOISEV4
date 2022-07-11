/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkMattesMutualInformationImageToImageMetricv4Okan_hxx
#define itkMattesMutualInformationImageToImageMetricv4Okan_hxx

#include "itkMattesMutualInformationImageToImageMetricv4Okan.h"
#include "itkCompensatedSummation.h"
#include <mutex>

namespace itk
{

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::MattesMutualInformationImageToImageMetricv4Okan() :
  m_MovingImageNormalizedMin(0.0),
  m_FixedImageNormalizedMin(0.0),
  m_FixedImageTrueMin(0.0),
  m_FixedImageTrueMax(0.0),
  m_MovingImageTrueMin(0.0),
  m_MovingImageTrueMax(0.0),
  m_FixedImageBinSize(0.0),
  m_MovingImageBinSize(0.0),
  m_CubicBSplineKernel(nullptr),
  m_CubicBSplineDerivativeKernel(nullptr),
  m_PRatioArray(0),
  // Initialize memory
  m_MovingImageMarginalPDF(0),
  m_ThreaderFixedImageMarginalPDF(0),
  // For multi-threading the metric
  m_ThreaderJointPDF(0),
  m_JointPDFDerivatives(nullptr),
  m_JointPDFSum(0.0)
{
  // We have our own GetValueAndDerivativeThreader's that we want
  // ImageToImageMetricv4 to use.
  this->m_DenseGetValueAndDerivativeThreader  = MattesMutualInformationDenseGetValueAndDerivativeThreaderType::New();
  this->m_SparseGetValueAndDerivativeThreader = MattesMutualInformationSparseGetValueAndDerivativeThreaderType::New();
  this->m_CubicBSplineKernel = CubicBSplineFunctionType::New();
  this->m_CubicBSplineDerivativeKernel = CubicBSplineDerivativeFunctionType::New();
}

/**
 * Initialize
 */
template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::Initialize()
{
  /* Superclass initialization */
  this->Superclass::Initialize();

  /* Expects moving image gradient source */
  if( this->GetGradientSourceIncludesFixed() || !this->GetGradientSourceIncludesMoving() )
    {
    itkExceptionMacro("Expected gradient source to be only Moving. Instead gradient source is: "
                      " Fixed: " << this->GetGradientSourceIncludesFixed()
                      << " Moving: " << this->GetGradientSourceIncludesMoving() );
    }

  {
  /**
   * Compute the minimum and maximum within the specified
   * analysis region (specified by a mask or sampled point set)
   * for creating the size of the 2D joint histogram.
   * Areas outside the defined analysis region should be ignored
   * in computing the range of intensity values.
   */

      if(this->m_FixedImageTrueMax==0)
      {

          this->m_FixedImageTrueMin = std::numeric_limits<typename TFixedImage::PixelType>::max();
          this->m_FixedImageTrueMax = std::numeric_limits<typename TFixedImage::PixelType>::min();

         // We need to make robust measures only over the requested mask region
         itk::ImageRegionConstIteratorWithIndex<TFixedImage> fi(this->m_FixedImage, this->m_FixedImage->GetBufferedRegion() );
         while( !fi.IsAtEnd() )
           {
           bool usePoint = true;
           if( usePoint )
             {
             const typename TFixedImage::PixelType currValue = fi.Get();
             this->m_FixedImageTrueMin = (m_FixedImageTrueMin < currValue) ? this->m_FixedImageTrueMin : currValue;
             this->m_FixedImageTrueMax = (m_FixedImageTrueMax > currValue) ? this->m_FixedImageTrueMax : currValue;
             }
           ++fi;
           }
      }

      if(this->m_MovingImageTrueMax==0)
      {
          this->m_MovingImageTrueMin = std::numeric_limits<typename TMovingImage::PixelType>::max();
          this->m_MovingImageTrueMax = std::numeric_limits<typename TMovingImage::PixelType>::min();

         {
         itk::ImageRegionConstIteratorWithIndex<TMovingImage> mi(this->m_MovingImage,this->m_MovingImage->GetBufferedRegion() );
         while( !mi.IsAtEnd() )
           {
           bool usePoint = true;
           if( usePoint )
             {
             const typename TMovingImage::PixelType currValue = mi.Get();
             this->m_MovingImageTrueMin = (m_MovingImageTrueMin < currValue) ? this->m_MovingImageTrueMin : currValue;
             this->m_MovingImageTrueMax = (m_MovingImageTrueMax > currValue) ? this->m_MovingImageTrueMax : currValue;
             }
           ++mi;
           }
         }
      }
  itkDebugMacro(" FixedImageMin: " << this->m_FixedImageTrueMin << " FixedImageMax: " << this->m_FixedImageTrueMax << std::endl);
  itkDebugMacro(" MovingImageMin: " << this->m_MovingImageTrueMin << " MovingImageMax: " << this->m_MovingImageTrueMax << std::endl);
  }

  /**
   * Compute binsize for the histograms.
   *
   * The binsize for the image intensities needs to be adjusted so that
   * we can avoid dealing with boundary conditions using the cubic
   * spline as the Parzen window.  We do this by increasing the size
   * of the bins so that the joint histogram becomes "padded" at the
   * borders. Because we are changing the binsize,
   * we also need to shift the minimum by the padded amount in order to
   * avoid minimum values filling in our padded region.
   *
   * Note that there can still be non-zero bin values in the padded region,
   * it's just that these bins will never be a central bin for the Parzen
   * window.
   *
   */
  constexpr int padding = 2;  // this will pad by 2 bins

  this->m_FixedImageBinSize = ( this->m_FixedImageTrueMax - this->m_FixedImageTrueMin )
    / static_cast<PDFValueType>( this->m_NumberOfHistogramBins - 2 * padding );
  this->m_FixedImageNormalizedMin = this->m_FixedImageTrueMin / this->m_FixedImageBinSize - static_cast<PDFValueType>( padding );

  this->m_MovingImageBinSize = ( this->m_MovingImageTrueMax - this->m_MovingImageTrueMin )
    / static_cast<PDFValueType>( this->m_NumberOfHistogramBins - 2 * padding );
  this->m_MovingImageNormalizedMin = this->m_MovingImageTrueMin / this->m_MovingImageBinSize - static_cast<PDFValueType>( padding );

  itkDebugMacro("FixedImageNormalizedMin: " << this->m_FixedImageNormalizedMin);
  itkDebugMacro("MovingImageNormalizedMin: " << this->m_MovingImageNormalizedMin);
  itkDebugMacro("FixedImageBinSize: " << this->m_FixedImageBinSize);
  itkDebugMacro("MovingImageBinSize; " << this->m_MovingImageBinSize);

  /* Porting note: the rest of the initialization that was performed
   * in MattesMutualImageToImageMetric::Initialize
   * is now performed in the threader BeforeThreadedExecution method */
  }

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::FinalizeThread( const ThreadIdType threadId )
{
  if( this->GetComputeDerivative() && ( !this->HasLocalSupport() ) )
    {
    this->m_ThreaderDerivativeManager[threadId].BlockAndReduce();
    }
}


template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::ComputeResults() const
{
  if( this->m_JointPDFSum < itk::NumericTraits< PDFValueType >::epsilon() )
    {
    itkExceptionMacro("Joint PDF summed to zero");
    }

  std::fill(this->m_MovingImageMarginalPDF.begin(), this->m_MovingImageMarginalPDF.end(), 0.0F);

  // Collect some results
  PDFValueType       totalMassOfPDF = 0.0;
  for( unsigned int i = 0; i < this->m_NumberOfHistogramBins; ++i )
    {
    totalMassOfPDF += this->m_ThreaderFixedImageMarginalPDF[0][i];
    }

  const PDFValueType normalizationFactor = 1.0 / this->m_JointPDFSum;
  JointPDFValueType *pdfPtr = this->m_ThreaderJointPDF[0]->GetBufferPointer();
  for( unsigned int i = 0; i < this->m_NumberOfHistogramBins; ++i )
    {
    PDFValueType * movingMarginalPtr = &(m_MovingImageMarginalPDF[0]);
    for( unsigned int j = 0; j < this->m_NumberOfHistogramBins; j++ )
      {
      *( pdfPtr ) *= normalizationFactor;
      *( movingMarginalPtr++ ) += *( pdfPtr++ );
      }
    }

  if( this->GetNumberOfValidPoints() == 0 )
    {
    itkExceptionMacro("All samples map outside moving image buffer. "
                      "The images do not sufficiently "
                      "overlap. They need to be initialized to have more overlap before this "
                      "metric will work. For instance, you can align the image centers by translation."
                      << std::endl);
    }

  // Normalize the fixed image marginal PDF
  if( totalMassOfPDF == 0.0 )
    {
    itkExceptionMacro("Fixed image marginal PDF summed to zero");
    }
  for( unsigned int bin = 0; bin < this->m_NumberOfHistogramBins; ++bin )
    {
    this->m_ThreaderFixedImageMarginalPDF[0][bin] /= totalMassOfPDF;
    }

  /**
   * Compute the metric by double summation over histogram.
   */

  // Setup pointer to point to the first bin
  JointPDFValueType *jointPDFPtr = this->m_ThreaderJointPDF[0]->GetBufferPointer();

  // Initialize sum to zero
  PDFValueType sumjh = 0.0;
  PDFValueType sumf = 0.0;
  PDFValueType summ= 0.0;


  const PDFValueType nFactor = 1.0 / ( this->m_MovingImageBinSize * this->GetNumberOfValidPoints() );
  static const PDFValueType closeToZero = std::numeric_limits<PDFValueType>::epsilon();

  for( unsigned int fixedIndex = 0; fixedIndex < this->m_NumberOfHistogramBins; ++fixedIndex )
  {
      const PDFValueType fixedImagePDFValue = this->m_ThreaderFixedImageMarginalPDF[0][fixedIndex];
      if( (fixedImagePDFValue > closeToZero) )
      {
        sumf+=    fixedImagePDFValue * std::log(fixedImagePDFValue);
      }
  }

  for( unsigned int movingIndex = 0; movingIndex < this->m_NumberOfHistogramBins; ++movingIndex)
  {
      const PDFValueType movingImagePDFValue = this->m_MovingImageMarginalPDF[movingIndex];
      if( (movingImagePDFValue > closeToZero) )
      {
        summ+=    movingImagePDFValue * std::log(movingImagePDFValue);
      }
  }

  for( unsigned int fixedIndex = 0; fixedIndex < this->m_NumberOfHistogramBins; ++fixedIndex )
  {
      for( unsigned int movingIndex = 0; movingIndex < this->m_NumberOfHistogramBins; ++movingIndex, jointPDFPtr++ )
      {
          const PDFValueType jointPDFValue = *( jointPDFPtr );
          if(  (jointPDFValue > closeToZero ) )
          {
              sumjh+= jointPDFValue *std::log(jointPDFValue);
          }
      }
  }

  double sum= (sumf+summ)/sumjh;


  // in ITKv4, metrics always minimize
  this->m_Value = static_cast<MeasureType>( -1.0 * sum );
}


template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::GetValueCommonAfterThreadedExecution()
{
  const ThreadIdType localNumberOfWorkUnitsUsed = this->GetNumberOfWorkUnitsUsed();

  const SizeValueType numberOfVoxels = this->m_NumberOfHistogramBins* this->m_NumberOfHistogramBins;
  JointPDFValueType * const pdfPtrStart = this->m_ThreaderJointPDF[0]->GetBufferPointer();

  for( unsigned int t = 1; t < localNumberOfWorkUnitsUsed; ++t )
    {
    JointPDFValueType *             pdfPtr = pdfPtrStart;
    JointPDFValueType const *       tPdfPtr = this->m_ThreaderJointPDF[t]->GetBufferPointer();
    JointPDFValueType const * const tPdfPtrEnd = tPdfPtr + numberOfVoxels;
    while( tPdfPtr < tPdfPtrEnd )
      {
      *( pdfPtr++ ) += *( tPdfPtr++ );
      }
    for( SizeValueType i = 0; i < this->m_NumberOfHistogramBins; ++i )
      {
      this->m_ThreaderFixedImageMarginalPDF[0][i] += this->m_ThreaderFixedImageMarginalPDF[t][i];
      }
    }

  // Sum of this threads domain into the this->m_JointPDFSum that covers that part of the domain.
  JointPDFValueType const * pdfPtr = pdfPtrStart;
  CompensatedSummation< PDFValueType > jointPDFSum;
  for( SizeValueType i = 0; i < numberOfVoxels; ++i )
    {
    jointPDFSum += *( pdfPtr++ );
    }
  this->m_JointPDFSum = jointPDFSum.GetSum();
}


template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
OffsetValueType
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::ComputeSingleFixedImageParzenWindowIndex( const FixedImagePixelType & value ) const
{
  // Note. The previous version of this metric pre-computed these values
  // during metric Initializaiton. But with the Metricv4 design, it's
  // more difficult to do so and retrieve as needed in an efficient way.

  // Determine parzen window arguments (see eqn 6 of Mattes paper [2]).
  const PDFValueType windowTerm = static_cast<PDFValueType>( value ) / this->m_FixedImageBinSize - this->m_FixedImageNormalizedMin;
  auto pindex = static_cast<OffsetValueType>( windowTerm );

  // Make sure the extreme values are in valid bins
  if( pindex < 2 )
    {
    pindex = 2;
    }
  else
    {
    const OffsetValueType nindex = static_cast<OffsetValueType>( this->m_NumberOfHistogramBins ) - 3;
    if( pindex > nindex )
      {
      pindex = nindex;
      }
    }

  return pindex;
}

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::DerivativeBufferManager
::Initialize( size_t maxBufferLength, const size_t cachedNumberOfLocalParameters,
              std::mutex * parentDerivativeLockPtr,
              typename JointPDFDerivativesType::Pointer parentJointPDFDerivatives)
{
  m_CurrentFillSize = 0;
  m_MemoryBlockSize = cachedNumberOfLocalParameters * maxBufferLength;
  m_BufferPDFValuesContainer.resize(maxBufferLength, nullptr);
  m_BufferOffsetContainer.resize(maxBufferLength, 0);
  m_CachedNumberOfLocalParameters = cachedNumberOfLocalParameters;
  m_MaxBufferSize = maxBufferLength;
  m_ParentJointPDFDerivativesLockPtr = parentDerivativeLockPtr;
  m_ParentJointPDFDerivatives = parentJointPDFDerivatives;
  // Allocate and initialize to zero (note the () at the end of the new
  // operator)
  // the memory as a single block
  m_MemoryBlock.resize(m_MemoryBlockSize, 0.0);
  for( size_t index = 0; index < maxBufferLength; ++index )
    {
    this->m_BufferPDFValuesContainer[index] = &(this->m_MemoryBlock[0]) + index * m_CachedNumberOfLocalParameters;
    }
}

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::DerivativeBufferManager
::DoubleBufferSize()
{
  m_MaxBufferSize = m_MaxBufferSize * 2;
  m_MemoryBlockSize = m_MemoryBlockSize * 2;
  m_BufferPDFValuesContainer.resize(m_MaxBufferSize, nullptr);
  m_BufferOffsetContainer.resize(m_MaxBufferSize, 0);
  m_MemoryBlock.resize(m_MemoryBlockSize, 0.0);
  for( size_t index = 0; index < m_MaxBufferSize; ++index )
    {
    this->m_BufferPDFValuesContainer[index] = &(this->m_MemoryBlock[0]) + index * m_CachedNumberOfLocalParameters;
    }
}

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::DerivativeBufferManager
::CheckAndReduceIfNecessary()
{
  if( m_CurrentFillSize ==  m_MaxBufferSize )
    {
    //Attempt to acquire the lock once
    std::unique_lock< std::mutex > FirstTryLockHolder(
      *this->m_ParentJointPDFDerivativesLockPtr, std::try_to_lock );
    if(FirstTryLockHolder.owns_lock())
      {
      ReduceBuffer();
      }
    else if (m_MaxBufferSize<5000)
      {
      DoubleBufferSize();
      //Attempt to acquire the lock a second time
      std::unique_lock< std::mutex > SecondTryLockHolder(
        *this->m_ParentJointPDFDerivativesLockPtr, std::try_to_lock);
      if(SecondTryLockHolder.owns_lock())
        {
        ReduceBuffer();
        }
      }
    else
      {
      // when CPU speed is higher than memory bandwith
      // the buffer could grow endlessly, so we limit it
      BlockAndReduce();
      }
    }
}

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::DerivativeBufferManager
::BlockAndReduce()
{
  if( m_CurrentFillSize > 0 )
    {
    std::lock_guard< std::mutex > LockHolder(*this->m_ParentJointPDFDerivativesLockPtr);
    ReduceBuffer();
    }
}

template <typename TFixedImage, typename TMovingImage, typename TVirtualImage, typename TInternalComputationValueType, typename TMetricTraits>
void
MattesMutualInformationImageToImageMetricv4Okan<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
::DerivativeBufferManager
::ReduceBuffer()
{
  auto BufferOffsetContainerIter(this->m_BufferOffsetContainer.begin() );
  auto  BufferPDFValuesContainerIter(
    this->m_BufferPDFValuesContainer.begin() );

  // NOTE: Only need to write out portion of buffer filled.
  size_t bufferIndex = 0;

  while( bufferIndex < m_CurrentFillSize )
    {
    const OffsetValueType         ThisIndexOffset = *BufferOffsetContainerIter;
    JointPDFDerivativesValueType *derivPtr = this->m_ParentJointPDFDerivatives->GetBufferPointer()
      + ThisIndexOffset;

    PDFValueType *             derivativeContribution = *BufferPDFValuesContainerIter;
    const PDFValueType * const endContribution = derivativeContribution + m_CachedNumberOfLocalParameters;
    while( derivativeContribution < endContribution )
      {
      *( derivPtr ) += *( derivativeContribution );
      // NOTE: Preliminary inconclusive tests indicates that setting to zero
      // while it's local in cache is faster than bulk memset after the loop
      // for small data sets
      *( derivativeContribution ) = 0.0; // Reset to zero after getting
                                         // value
      ++derivativeContribution;
      ++derivPtr;
      }

    ++BufferOffsetContainerIter;
    ++BufferPDFValuesContainerIter;
    ++bufferIndex;
    }
  m_CurrentFillSize = 0; // Reset fill size back to zero.
}

} // end namespace itk

#endif

