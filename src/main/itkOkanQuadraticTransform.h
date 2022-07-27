
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
#ifndef itkOkanQuadraticTransform_h
#define itkOkanQuadraticTransform_h


#include "itkMacro.h"
#include "itkMatrix.h"
#include "itkTransform.h"

#include <iostream>

namespace itk
{

/* MatrixOrthogonalityTolerance is a utility to
 * allow setting the tolerance limits used for
 * checking if a matrix meet the orthogonality
 * constraints of being a rigid rotation matrix.
 * The tolerance needs to be different for
 * matrices of type float vs. double.
 */


template <typename TParametersValueType = double, unsigned int VInputDimension = 3, unsigned int VOutputDimension = 3>
class ITK_TEMPLATE_EXPORT OkanQuadraticTransform
  : public Transform<TParametersValueType, VInputDimension, VOutputDimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(OkanQuadraticTransform);

  using Self = OkanQuadraticTransform;
  using Superclass = Transform<TParametersValueType, VInputDimension, VOutputDimension>;

  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkTypeMacro(OkanQuadraticTransform, Transform);

  itkNewMacro(Self);

  static constexpr unsigned int InputSpaceDimension = VInputDimension;
  static constexpr unsigned int OutputSpaceDimension = VOutputDimension;
  static constexpr unsigned int ParametersDimension = VOutputDimension * (VInputDimension + 1);

  static constexpr unsigned int NQUADPARAMS=24;

  using typename Superclass::FixedParametersType;
  using typename Superclass::FixedParametersValueType;
  using typename Superclass::ParametersType;
  using typename Superclass::ParametersValueType;

  using typename Superclass::JacobianType;
  using typename Superclass::JacobianPositionType;
  using typename Superclass::InverseJacobianPositionType;

  using typename Superclass::TransformCategoryEnum;

  using typename Superclass::ScalarType;

  using InputVectorType = Vector<TParametersValueType, Self::InputSpaceDimension>;
  using OutputVectorType = Vector<TParametersValueType, Self::OutputSpaceDimension>;
  using OutputVectorValueType = typename OutputVectorType::ValueType;

  using InputCovariantVectorType = CovariantVector<TParametersValueType, Self::InputSpaceDimension>;
  using OutputCovariantVectorType = CovariantVector<TParametersValueType, Self::OutputSpaceDimension>;

  using typename Superclass::InputVectorPixelType;
  using typename Superclass::OutputVectorPixelType;

  using typename Superclass::InputDiffusionTensor3DType;
  using typename Superclass::OutputDiffusionTensor3DType;

  using typename Superclass::InputSymmetricSecondRankTensorType;
  using typename Superclass::OutputSymmetricSecondRankTensorType;

  using InputTensorEigenVectorType = CovariantVector<TParametersValueType, InputDiffusionTensor3DType::Dimension>;

  using InputVnlVectorType = vnl_vector_fixed<TParametersValueType, Self::InputSpaceDimension>;
  using OutputVnlVectorType = vnl_vector_fixed<TParametersValueType, Self::OutputSpaceDimension>;

  using InputPointType = Point<TParametersValueType, Self::InputSpaceDimension>;
  using InputPointValueType = typename InputPointType::ValueType;
  using OutputPointType = Point<TParametersValueType, Self::OutputSpaceDimension>;
  using OutputPointValueType = typename OutputPointType::ValueType;

  using MatrixType = Matrix<TParametersValueType, Self::OutputSpaceDimension, Self::InputSpaceDimension>;
  using MatrixValueType = typename MatrixType::ValueType;

  using InverseMatrixType = Matrix<TParametersValueType, Self::InputSpaceDimension, Self::OutputSpaceDimension>;

  using CenterType = InputPointType;

 // using OffsetType = OutputVectorType;
 // using OffsetValueType = typename OffsetType::ValueType;

  using TranslationType = OutputVectorType;

  using TranslationValueType = typename TranslationType::ValueType;

  using InverseTransformBaseType = typename Superclass::InverseTransformBaseType;
  using InverseTransformBasePointer = typename InverseTransformBaseType::Pointer;

  using InverseTransformType = OkanQuadraticTransform<TParametersValueType, VOutputDimension, VInputDimension>;
  friend class OkanQuadraticTransform<TParametersValueType, VOutputDimension, VInputDimension>;

  virtual void
  SetIdentity();

  TransformCategoryEnum
  GetTransformCategory() const override
  {
    return Self::TransformCategoryEnum::Linear;
  }

  virtual void
  SetMatrix(const MatrixType & matrix)
  {
    m_Matrix = matrix;    
    this->ComputeMatrixParameters();
    m_MatrixMTime.Modified();
    this->Modified();
    return;
  }
  virtual const MatrixType &
  GetMatrix() const
  {
    return m_Matrix;
  }



  void
  SetTranslation(const OutputVectorType & translation)
  {
      this->m_Parameters[0]= translation[0];
      this->m_Parameters[1]= translation[1];
      this->m_Parameters[2]= translation[2];

      m_Translation = translation;      
      this->Modified();
      return;
  }
  const OutputVectorType &
  GetTranslation() const
  {
    return m_Translation;
  }

  void
  SetParameters(const ParametersType & parameters) override;

  const ParametersType &
  GetParameters() const override;

  void SetParametersForOptimizationFlags(const ParametersType & flags)
  {
       this->m_ParametersForOptimizationFlags   =flags;
       if(this->m_ParametersForOptimizationFlags[14]!=0  || this->m_ParametersForOptimizationFlags[15]!=0 || this->m_ParametersForOptimizationFlags[16]!=0 || this->m_ParametersForOptimizationFlags[17]!=0 || this->m_ParametersForOptimizationFlags[18]!=0 || this->m_ParametersForOptimizationFlags[19]!=0 || this->m_ParametersForOptimizationFlags[20]!=0)
           do_cubic=true;
  }

  ParametersType GetParametersForOptimizationFlags()
  {
       return this->m_ParametersForOptimizationFlags;
  }



  virtual void SetFixedParameters(const ParametersType &fp){this->m_FixedParameters = fp;};

  /** Get the Fixed Parameters. */
//  virtual const ParametersType & GetFixedParameters(void) const;
  virtual const ParametersType & GetFixedParameters(void) const{return this->m_FixedParameters;};

  void
  Compose(const Self * other, bool pre = false);

  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  using Superclass::TransformVector;

  OutputVectorType
  TransformVector(const InputVectorType & vect) const override;

  OutputVnlVectorType
  TransformVector(const InputVnlVectorType & vect) const override;

  OutputVectorPixelType
  TransformVector(const InputVectorPixelType & vect) const override;

  using Superclass::TransformCovariantVector;

  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType & vec) const override;

  OutputVectorPixelType
  TransformCovariantVector(const InputVectorPixelType & vect) const override;

  using Superclass::TransformDiffusionTensor3D;

  OutputDiffusionTensor3DType
  TransformDiffusionTensor3D(const InputDiffusionTensor3DType & tensor) const override;

  OutputVectorPixelType
  TransformDiffusionTensor3D(const InputVectorPixelType & tensor) const override;

  using Superclass::TransformSymmetricSecondRankTensor;
  OutputSymmetricSecondRankTensorType
  TransformSymmetricSecondRankTensor(const InputSymmetricSecondRankTensorType & inputTensor) const override;

  OutputVectorPixelType
  TransformSymmetricSecondRankTensor(const InputVectorPixelType & inputTensor) const override;


  void
  ComputeJacobianWithRespectToParameters(const InputPointType & p, JacobianType & jacobian) const override;


  //void
  //ComputeJacobianWithRespectToPosition(const InputPointType & x, JacobianPositionType & jac) const override;
  //using Superclass::ComputeJacobianWithRespectToPosition;
  virtual void ComputeJacobianWithRespectToPosition(const InputPointType  & x, JacobianType & jac) const;


  //void   ComputeInverseJacobianWithRespectToPosition(const InputPointType &        x,
  //                                            InverseJacobianPositionType & jac) const override;
  //using Superclass::ComputeInverseJacobianWithRespectToPosition;

  virtual void ComputeInverseJacobianWithRespectToPosition(const InputPointType  & x, JacobianType & jac) const;

  bool
  GetInverse(InverseTransformType * inverse) const;
  InverseTransformBasePointer
  GetInverseTransform() const override;

  bool
  IsLinear() const override
  {
    return false;
  }
  void SetPhase(std::string np)
  {
      if(np=="horizontal")
          this->phase=0;
      if(np=="vertical")
          this->phase=1;
      if(np=="slice")
          this->phase=2;
  }

  void SetPhase(short p)
  {
      this->phase=p;
  }

  short GetPhase(){return this->phase;};


protected:
  const InverseMatrixType &
  GetInverseMatrix() const;

protected:
  OkanQuadraticTransform(const ParametersType params);
  OkanQuadraticTransform(unsigned int paramDims);
  OkanQuadraticTransform();
  ~OkanQuadraticTransform() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  const InverseMatrixType &
  GetVarInverseMatrix() const
  {
    return m_InverseMatrix;
  }
  void
  SetVarInverseMatrix(const InverseMatrixType & matrix) const
  {
    m_InverseMatrix = matrix;
    m_InverseMatrixMTime.Modified();
  }
  bool
  InverseMatrixIsOld() const
  {
    if (m_MatrixMTime != m_InverseMatrixMTime)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  virtual void
  ComputeMatrixParameters();

  virtual void
  ComputeMatrix();

  void
  SetVarMatrix(const MatrixType & matrix)
  {
    m_Matrix = matrix;
    m_MatrixMTime.Modified();
  }

  virtual void
  ComputeTranslation();

  void
  SetVarTranslation(const OutputVectorType & translation)
  {
    m_Translation = translation;
  }


  void
  SetVarCenter(const InputPointType & center)
  {
    m_Center = center;
  }

  itkGetConstMacro(Singular, bool);

private:
  MatrixType                m_Matrix{ MatrixType::GetIdentity() };               // Matrix of the transformation
  OutputVectorType          m_Offset{};                                          // Offset of the transformation
  mutable InverseMatrixType m_InverseMatrix{ InverseMatrixType::GetIdentity() }; // Inverse of the matrix
  mutable bool              m_Singular{ false };                                 // Is m_Inverse singular?

  InputPointType   m_Center{};
  OutputVectorType m_Translation{};

   ParametersType m_ParametersForOptimizationFlags;
   short phase;
   bool do_cubic;


  TimeStamp         m_MatrixMTime;
  mutable TimeStamp m_InverseMatrixMTime;
}; // class OkanQuadraticTransform
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkOkanQuadraticTransform.hxx"
#endif

#endif /* itkOkanQuadraticTransform_h */
