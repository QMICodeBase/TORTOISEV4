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
#ifndef __itkOkanQuadraticTransform_h
#define __itkOkanQuadraticTransform_h

#include "itkMatrix.h"
#include "itkTransform.h"
#include "itkMacro.h"

#include <iostream> 
#include <fstream> 
//using namespace std;
#include <stdio.h>

 #define NQUADPARAMS 24

namespace itk
{
// \class itkOkanQuadraticTransform


template <  class TScalarType = double,         // Data type for scalars
            unsigned int NInputDimensions = 3,  // Number of dimensions in the input space
            unsigned int NOutputDimensions = 3>
// Number of dimensions in the output space
class OkanQuadraticTransform :   public Transform<TScalarType, NInputDimensions, NOutputDimensions>
{
public:
  /** Standard typedefs   */
  typedef OkanQuadraticTransform Self;
  typedef Transform<TScalarType, NInputDimensions,NOutputDimensions>        Superclass;

  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Run-time type information (and related methods).   */
  itkTypeMacro(OkanQuadraticTransform, Transform);

  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro(Self);



  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);
  itkStaticConstMacro( ParametersDimension, unsigned int, NQUADPARAMS);
  
  
  

  /** Parameters Type   */
  typedef typename Superclass::ParametersType      ParametersType;
  typedef typename Superclass::ParametersValueType ParametersValueType;

  /** Jacobian Type   */
  typedef typename Superclass::JacobianType JacobianType;

  /** Transform category type. */
  typedef typename Superclass::TransformCategoryType TransformCategoryType;

  /** Standard scalar type for this class */
  typedef typename Superclass::ScalarType ScalarType;

  /** Standard vector type for this class   */
  typedef Vector<TScalarType, itkGetStaticConstMacro(InputSpaceDimension)>  InputVectorType;
  typedef Vector<TScalarType, itkGetStaticConstMacro(OutputSpaceDimension)> OutputVectorType;
  typedef typename OutputVectorType::ValueType OutputVectorValueType;

  /** Standard covariant vector type for this class   */
  typedef CovariantVector<TScalarType, itkGetStaticConstMacro(InputSpaceDimension)>   InputCovariantVectorType;
  typedef CovariantVector<TScalarType, itkGetStaticConstMacro(OutputSpaceDimension)>  OutputCovariantVectorType;

  typedef typename Superclass::InputVectorPixelType  InputVectorPixelType;
  typedef typename Superclass::OutputVectorPixelType OutputVectorPixelType;

  /** Standard diffusion tensor type for this class */
  typedef typename Superclass::InputDiffusionTensor3DType   InputDiffusionTensor3DType;
  typedef typename Superclass::OutputDiffusionTensor3DType  OutputDiffusionTensor3DType;

  /** Standard tensor type for this class */
  typedef typename Superclass::InputSymmetricSecondRankTensorType   InputSymmetricSecondRankTensorType;
  typedef typename Superclass::OutputSymmetricSecondRankTensorType   OutputSymmetricSecondRankTensorType;

  typedef CovariantVector<TScalarType, InputDiffusionTensor3DType::Dimension>   InputTensorEigenVectorType;

  /** Standard vnl_vector type for this class   */
  typedef vnl_vector_fixed<TScalarType, itkGetStaticConstMacro(InputSpaceDimension)>  InputVnlVectorType;
  typedef vnl_vector_fixed<TScalarType, itkGetStaticConstMacro(OutputSpaceDimension)>   OutputVnlVectorType;

  /** Standard coordinate point type for this class   */
  typedef Point<TScalarType, itkGetStaticConstMacro(InputSpaceDimension)>  InputPointType;
  typedef typename InputPointType::ValueType InputPointValueType;
  typedef Point<TScalarType,itkGetStaticConstMacro(OutputSpaceDimension)>  OutputPointType;
  typedef typename OutputPointType::ValueType OutputPointValueType;

  /** Standard matrix type for this class   */
  typedef Matrix<TScalarType, itkGetStaticConstMacro(OutputSpaceDimension), itkGetStaticConstMacro(InputSpaceDimension)>  MatrixType;
  typedef typename MatrixType::ValueType MatrixValueType;

  /** Standard inverse matrix type for this class   */
  typedef Matrix<TScalarType, itkGetStaticConstMacro(InputSpaceDimension), itkGetStaticConstMacro(OutputSpaceDimension)>   InverseMatrixType;

  //typedef InputPointType CenterType;

  //typedef OutputVectorType               OffsetType;
  //typedef typename OffsetType::ValueType OffsetValueType;

  typedef OutputVectorType TranslationType;

  typedef typename TranslationType::ValueType TranslationValueType;

  /** Base inverse transform type. This type should not be changed to the
   * concrete inverse transform type or inheritance would be lost. */
  typedef typename Superclass::InverseTransformBaseType InverseTransformBaseType;
  typedef typename InverseTransformBaseType::Pointer    InverseTransformBasePointer;

  /** Set the transformation to an Identity
   *
   * This sets the matrix to identity and the Offset to null. */
  virtual void SetIdentity(void);

  /** Indicates the category transform.
   *  e.g. an affine transform, or a local one, e.g. a deformation field.
   */
  virtual TransformCategoryType GetTransformCategory() const
  {
    return Self::UnknownTransformCategory;    
  }

  /** Set matrix of an OkanQuadraticTransform
   *
   * This method sets the matrix of an OkanQuadraticTransform to a
   * value specified by the user.
   *
   * This updates the Offset wrt to current translation
   * and center.  See the warning regarding offset-versus-translation
   * in the documentation for SetCenter.
   *
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset */
  
  
  


  virtual const MatrixType & GetMatrix() const
  {
      return m_Matrix;
  }


  /** Set translation of an OkanQuadraticTransform
   *
   * This method sets the translation of an OkanQuadraticTransform.
   * This updates Offset to reflect current translation.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset */
  void SetTranslation(const OutputVectorType & translation)
  {
      this->m_Parameters[0]= translation[0];
      this->m_Parameters[1]= translation[1];
      this->m_Parameters[2]= translation[2];
    m_Translation = translation; 
    this->Modified(); 
    return;
  }

  /** Get translation component of the OkanQuadraticTransform
   *
   * This method returns the translation used after rotation
   * about the center point.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset */
  const OutputVectorType & GetTranslation(void) const
  {
    return m_Translation;
  }

  /** Set the transformation from a container of parameters.
   * The first (NOutputDimension x NInputDimension) parameters define the
   * matrix and the last NOutputDimension parameters the translation.
   * Offset is updated based on current center. */
  void SetParameters(const ParametersType & parameters);


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



  /** Get the Transformation Parameters. */
  const ParametersType & GetParameters(void) const;

  /** Set the fixed parameters and update internal transformation. */
  virtual void SetFixedParameters(const ParametersType &fp){this->m_FixedParameters = fp;};

  /** Get the Fixed Parameters. */
//  virtual const ParametersType & GetFixedParameters(void) const;
  virtual const ParametersType & GetFixedParameters(void) const{return this->m_FixedParameters;};

  /** Compose with another OkanQuadraticTransform
   *
   * This method composes self with another OkanQuadraticTransform of the
   * same dimension, modifying self to be the composition of self
   * and other.  If the argument pre is true, then other is
   * precomposed with self; that is, the resulting transformation
   * consists of first applying other to the source, followed by
   * self.  If pre is false or omitted, then other is post-composed
   * with self; that is the resulting transformation consists of
   * first applying self to the source, followed by other.
   * This updates the Translation based on current center. */
  void Compose(const Self *other, bool pre = 0);

  /** Transform by an affine transformation
   *
   * This method applies the affine transform given by self to a
   * given point or vector, returning the transformed point or
   * vector.  The TransformPoint method transforms its argument as
   * an affine point, whereas the TransformVector method transforms
   * its argument as a vector. */

  OutputPointType       TransformPoint(const InputPointType & point) const;

  using Superclass::TransformVector;

  OutputVectorType      TransformVector(const InputVectorType & vector) const;

  OutputVnlVectorType   TransformVector(const InputVnlVectorType & vector) const;

  OutputVectorPixelType TransformVector(const InputVectorPixelType & vector) const;

  using Superclass::TransformCovariantVector;

  OutputCovariantVectorType TransformCovariantVector(const InputCovariantVectorType & vector) const;

  OutputVectorPixelType TransformCovariantVector(const InputVectorPixelType & vector) const;

  using Superclass::TransformDiffusionTensor3D;

  OutputDiffusionTensor3DType TransformDiffusionTensor3D(const InputDiffusionTensor3DType & tensor) const;

  OutputVectorPixelType TransformDiffusionTensor3D(const InputVectorPixelType & tensor ) const;

  using Superclass::TransformSymmetricSecondRankTensor;
  OutputSymmetricSecondRankTensorType TransformSymmetricSecondRankTensor( const InputSymmetricSecondRankTensorType & tensor ) const;

  OutputVectorPixelType TransformSymmetricSecondRankTensor( const InputVectorPixelType & tensor ) const;

  /** Compute the Jacobian of the transformation
   *
   * This method computes the Jacobian matrix of the transformation.
   * given point or vector, returning the transformed point or
   * vector. The rank of the Jacobian will also indicate if the transform
   * is invertible at this point.
   * Get local Jacobian for the given point
   * \c j will sized properly as needed.
   */
  virtual void ComputeJacobianWithRespectToParameters(const InputPointType  & x, JacobianType & j) const;

  /** Get the jacobian with respect to position. This simply returns
   * the current Matrix. jac will be resized as needed, but it's
   * more efficient if it's already properly sized. */
  virtual void ComputeJacobianWithRespectToPosition(const InputPointType  & x, JacobianType & jac) const;

  /** Get the jacobian with respect to position. This simply returns
   * the inverse of the current Matrix. jac will be resized as needed, but it's
   * more efficient if it's already properly sized. */
  virtual void ComputeInverseJacobianWithRespectToPosition(const InputPointType  & x, JacobianType & jac) const;

  /** Create inverse of an affine transformation
   *
   * This populates the parameters an affine transform such that
   * the transform is the inverse of self. If self is not invertible,
   * an exception is thrown.
   * Note that by default the inverese transform is centered at
   * the origin. If you need to compute the inverse centered at a point, p,
   *
   * \code
   * transform2->SetCenter( p );
   * transform1->GetInverse( transform2 );
   * \endcode
   *
   * transform2 will now contain the inverse of transform1 and will
   * with its center set to p. Flipping the two statements will produce an
   * incorrect transform.
   *
   */
  bool GetInverse(Self *inverse) const;

  /** Return an inverse of this transform. */
  virtual InverseTransformBasePointer GetInverseTransform() const;

  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  virtual bool IsLinear() const
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

 
  

#if !defined(ITK_LEGACY_REMOVE)

public:
#else

protected:
#endif
  /** \deprecated Use GetInverse for public API instead.
   * Method will eventually be made a protected member function */
  const InverseMatrixType & GetInverseMatrix(void) const;

  
protected:
  /** Construct an OkanQuadraticTransform object
   *
   * This method constructs a new OkanQuadraticTransform object and
   * initializes the matrix and offset parts of the transformation
   * to values specified by the caller.  If the arguments are
   * omitted, then the OkanQuadraticTransform is initialized to an identity
   * transformation in the appropriate number of dimensions. */
  OkanQuadraticTransform(const ParametersType params);
  OkanQuadraticTransform(unsigned int paramDims);
  OkanQuadraticTransform();

  /** Destroy an OkanQuadraticTransform object */
  virtual ~OkanQuadraticTransform();

  /** Print contents of an OkanQuadraticTransform */
  void PrintSelf(std::ostream & s, Indent indent) const;

  const InverseMatrixType & GetVarInverseMatrix(void) const
  {
    return m_InverseMatrix;
  }
  void SetVarInverseMatrix(const InverseMatrixType & matrix) const
  {
    m_InverseMatrix = matrix; m_InverseMatrixMTime.Modified();
  }
  
  bool InverseMatrixIsOld(void) const
  {
    if( m_MatrixMTime != m_InverseMatrixMTime )
      {
      return true;
      }
    else
      {
      return false;
      }
  }

  virtual void ComputeMatrixParameters(void);

  virtual void ComputeMatrix(void);

  void SetVarMatrix(const MatrixType & matrix)
  {
    m_Matrix = matrix; 
    m_MatrixMTime.Modified();
  }

  virtual void ComputeTranslation(void);

  void SetVarTranslation(const OutputVectorType & translation)
  {
    m_Translation = translation;
  }

  

  

private:

  OkanQuadraticTransform(const Self & other);
  const Self & operator=(const Self &);

  MatrixType                m_Matrix;           // Matrix of the transformation
  mutable InverseMatrixType m_InverseMatrix;    // Inverse of the matrix
  mutable bool              m_Singular;         // Is m_Inverse singular?
 

  OutputVectorType m_Translation;
  
  
 // ParametersType m_Parameters;
  ParametersType m_ParametersForOptimizationFlags;
  short phase;
  bool do_cubic;
  

  /** To avoid recomputation of the inverse if not needed */
  TimeStamp         m_MatrixMTime;
  mutable TimeStamp m_InverseMatrixMTime;
}; // class OkanQuadraticTransform



}  // namespace itk



#ifndef ITK_MANUAL_INSTANTIATION
#include "itkOkanQuadraticTransform.hxx"
#endif

#endif /* __itkOkanQuadraticTransform_h */
 
