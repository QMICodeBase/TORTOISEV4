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
#ifndef __itkOkanQuadraticTransform_hxx
#define __itkOkanQuadraticTransform_hxx

#include "itkNumericTraits.h"
#include "itkOkanQuadraticTransform.h"
#include "vnl/algo/vnl_matrix_inverse.h"
#include "itkMath.h"
#include "itkCrossHelper.h"


namespace itk
{
// Constructor with default arguments
template <class TScalarType, unsigned int NInputDimensions,unsigned int NOutputDimensions>
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::OkanQuadraticTransform() :  Superclass(ParametersDimension)
{
  m_Matrix.SetIdentity();
  m_MatrixMTime.Modified();
  m_Translation.Fill(0);
  m_Singular = false;
  m_InverseMatrix.SetIdentity();
  m_InverseMatrixMTime = m_MatrixMTime;
  this->m_FixedParameters.SetSize(NQUADPARAMS);
  this->m_FixedParameters.Fill(0.0);
  this->m_Parameters.SetSize(NQUADPARAMS);
  this->m_ParametersForOptimizationFlags.SetSize(NQUADPARAMS);
  this->do_cubic=0;

  this->phase=1;

  SetIdentity();
}

// Constructor with default arguments
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::OkanQuadraticTransform(unsigned int paramDims) :
  Superclass(paramDims)
{
  m_Matrix.SetIdentity();
  m_MatrixMTime.Modified();
  m_Translation.Fill(0);
  m_Singular = false;
  m_InverseMatrix.SetIdentity();
  m_InverseMatrixMTime = m_MatrixMTime;
  this->m_Parameters.SetSize(paramDims);
  this->m_ParametersForOptimizationFlags.SetSize(paramDims);
  this->do_cubic=0;

  this->phase=1;

  SetIdentity();
}

// Constructor with explicit arguments
template <class TScalarType, unsigned int NInputDimensions,
          unsigned int NOutputDimensions>
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::OkanQuadraticTransform(const ParametersType params)
{
    int sz= params.GetSize();
    this->m_Parameters.SetSize(sz);
    this->m_Parameters=params;
    this->m_ParametersForOptimizationFlags.SetSize(sz);
    this->do_cubic=0;

    if(params[8]>= params[7] && params[8]>= params[6])
        this->phase=2;
    if(params[6]>= params[7] && params[6]>= params[8])
        this->phase=0;
    if(params[7]>= params[6] && params[7]>= params[8])
        this->phase=1;



    this->ComputeMatrix();
    this->ComputeTranslation();

    m_MatrixMTime.Modified();
}



// Print self
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);


  os << indent << "Params: " << std::endl;


  for( int i = 0; i < this->m_Parameters.GetSize(); i++ )
  {
      os << this->m_Parameters[i] << " ";
  }
  os << std::endl;

}

// Constructor with explicit arguments
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SetIdentity(void)
{
  m_Matrix.SetIdentity();
  m_MatrixMTime.Modified();
  m_Translation.Fill(NumericTraits<OutputVectorValueType>::Zero);
  m_Singular = false;
  m_InverseMatrix.SetIdentity();
  m_InverseMatrixMTime = m_MatrixMTime;
  this->m_Parameters.Fill(0);
  if(phase==0 || phase==1 || phase==2)
      this->m_Parameters[6+phase]=1;
  this->Modified();
}



// Compose with another affine transformation
template <class TScalarType, unsigned int NInputDimensions,  unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::Compose(const Self *other, bool pre)
{
    std::cout<<"COMPOSE NOT IMLEMENTED YET~!!"<<std::endl;
  return;
}



// Transform a point
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>::OutputPointType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformPoint(const InputPointType & point) const
{      
    OutputPointType p=point;
    p[0]-= this->m_Parameters[21];
    p[1]-= this->m_Parameters[22];
    p[2]-= this->m_Parameters[23];

    p = m_Matrix * p + m_Translation;

    double new_phase_coord= this->m_Parameters[6]*p[0]      +this->m_Parameters[7] *p[1]      + this->m_Parameters[8] *p[2]+
                        this->m_Parameters[9]*p[0]*p[1] +this->m_Parameters[10]*p[0]*p[2] + this->m_Parameters[11]*p[1]*p[2]+
                        this->m_Parameters[12]*(p[0]*p[0]-p[1]*p[1]) +this->m_Parameters[13]*(2*p[2]*p[2]-p[0]*p[0]-p[1]*p[1]);


    if(this->do_cubic)
    {       
        double total_change = this->m_Parameters[14]*p[0]*p[1]*p[2] +
                              this->m_Parameters[15]*p[2]*(p[0]*p[0]-p[1]*p[1]) +
                              this->m_Parameters[16]*p[0]*(4*p[2]*p[2]-p[0]*p[0]-p[1]*p[1]) +
                              this->m_Parameters[17]*p[1]*(4*p[2]*p[2]-p[0]*p[0]-p[1]*p[1]) +
                              this->m_Parameters[18]*p[0]*(p[0]*p[0]-3*p[1]*p[1]) +
                              this->m_Parameters[19]*p[1]*(3*p[0]*p[0]-p[1]*p[1]) +
                              this->m_Parameters[20]*p[2]*(2*p[2]*p[2]-3*p[0]*p[0]-3*p[1]*p[1]) ;



        //p[this->phase]+=total_change;
        
          p[this->phase]=new_phase_coord+total_change;
    }
    else
        p[this->phase]=new_phase_coord;

    p[0]+=this->m_Parameters[21];
    p[1]+=this->m_Parameters[22];
    p[2]+=this->m_Parameters[23];

    return p;
}



// Transform a vector
template <class TScalarType, unsigned int NInputDimensions,unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType, NInputDimensions,NOutputDimensions>::OutputVectorType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformVector(const InputVectorType & vect) const
{
    InputPointType p,q;
    p[0]=vect[0];
    p[1]=vect[1];
    p[2]=vect[2];
    q.Fill(0);
    
    OutputPointType a= this->TransformPoint(p);
    OutputPointType b= this->TransformPoint(q);
    return a-b;    
}

// Transform a vnl_vector_fixed
template <class TScalarType, unsigned int NInputDimensions,unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputVnlVectorType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformVector(const InputVnlVectorType & vect) const
{
    InputPointType p,q;
    p[0]=vect[0];
    p[1]=vect[1];
    p[2]=vect[2];
    q.Fill(0);
    
    OutputPointType a= this->TransformPoint(p);
    OutputPointType b= this->TransformPoint(q);
    OutputVectorType ot=a-b;
    
    const unsigned int vectorDim = 3;
    vnl_vector<TScalarType> vnl_vect( vectorDim );
    vnl_vect[0]=ot[0];
    vnl_vect[1]=ot[1];
    vnl_vect[2]=ot[2];
    return vnl_vect;
    

}

// Transform a variable length vector
template <class TScalarType, unsigned int NInputDimensions,unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputVectorPixelType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformVector(const InputVectorPixelType & vect) const
{
    InputPointType p,q;
    p[0]=vect[0];
    p[1]=vect[1];
    p[2]=vect[2];
    q.Fill(0);
    
    OutputPointType a= this->TransformPoint(p);
    OutputPointType b= this->TransformPoint(q);
    OutputVectorType ot=a-b;
      

    OutputVectorPixelType   outVect;
    outVect[0]=ot[0];
    outVect[1]=ot[1];
    outVect[2]=ot[2];
    
    
  return outVect;
}



// Transform a CovariantVector
template <class TScalarType, unsigned int NInputDimensions,unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputCovariantVectorType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformCovariantVector(const InputCovariantVectorType & vec) const
{
    std::cout<<"TransformCovariantVector NOT IMPLEMENTED!!"<<std::endl;
    return vec;            
}

// Transform a variable length vector
template <class TScalarType, unsigned int NInputDimensions,unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputVectorPixelType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformCovariantVector(const InputVectorPixelType & vect) const
{
    std::cout<<"TransformCovariantVector NOT IMPLEMENTED!!"<<std::endl;
    return vect;            

}

// Transform a Tensor
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputDiffusionTensor3DType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformDiffusionTensor3D(const InputDiffusionTensor3DType & tensor) const
{
    std::cout<<"TransformDiffusionTensor3D NOT IMPLEMENTED!!"<<std::endl;
    return tensor;            
}

// Transform a Tensor
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputVectorPixelType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformDiffusionTensor3D(const InputVectorPixelType & tensor) const
{
    std::cout<<"TransformDiffusionTensor3D NOT IMPLEMENTED!!"<<std::endl;
    return tensor;  
}

/**
 * Transform tensor
 */
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputSymmetricSecondRankTensorType
OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>
::TransformSymmetricSecondRankTensor( const InputSymmetricSecondRankTensorType& inputTensor ) const
{
    std::cout<<"TransformSymmetricSecondRankTensor NOT IMPLEMENTED!!"<<std::endl;
    return inputTensor;  
}

/**
 * Transform tensor
 */
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::OutputVectorPixelType
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformSymmetricSecondRankTensor( const InputVectorPixelType& inputTensor ) const
{

    std::cout<<"TransformSymmetricSecondRankTensor NOT IMPLEMENTED!!"<<std::endl;
    return inputTensor;  
}

// Recompute the inverse matrix (internal)
template <class TScalarType, unsigned int NInputDimensions,unsigned int NOutputDimensions>
const typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::InverseMatrixType 
& OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetInverseMatrix(void) const
  {
  // If the transform has been modified we recompute the inverse
  if( m_InverseMatrixMTime != m_MatrixMTime )
    {
    m_Singular = false;
    try
      {
      m_InverseMatrix  = m_Matrix.GetInverse();
      }
    catch( ... )
      {
      m_Singular = true;
      }
    m_InverseMatrixMTime = m_MatrixMTime;
    }

  return m_InverseMatrix;
  }


// Return an inverse of this transform
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
typename OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>::InverseTransformBasePointer
OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetInverseTransform() const
{
  return nullptr;
}



// Get parameters
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
const typename OkanQuadraticTransform<TScalarType,NInputDimensions,NOutputDimensions>::ParametersType
& OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetParameters(void) const
{
  return this->m_Parameters;
}

// Set parameters
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SetParameters(const ParametersType & parameters)
{
  if( parameters.Size() < NQUADPARAMS )
    {
    itkExceptionMacro
      (<< "Error setting parameters: parameters array size ("
       << parameters.Size() << ") is less than expected (24) "
       << " (NInputDimensions * NOutputDimensions + NOutputDimensions) "
       << " (" << NInputDimensions << " * " << NOutputDimensions
       << " + " << NOutputDimensions
       << " = " << NInputDimensions * NOutputDimensions + NOutputDimensions << ")"
      );
    }

  // Save parameters. Needed for proper operation of TransformUpdateParameters.
  if( &parameters != &(this->m_Parameters) )
    {
    this->m_Parameters = parameters;
    }

    this->ComputeMatrix();


  OutputVectorType newTranslation;
  newTranslation[0] = parameters[0];
  newTranslation[1] = parameters[1];
  newTranslation[2] = parameters[2];

    this->SetVarTranslation(newTranslation);



  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();
}


// Compute the Matrix
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeMatrix(void)
{
    double m_AngleX= this->m_Parameters[3];
    double m_AngleY= this->m_Parameters[4];
    double m_AngleZ= this->m_Parameters[5];

    const ScalarType cx = std::cos(m_AngleX);
    const ScalarType sx = std::sin(m_AngleX);
    const ScalarType cy = std::cos(m_AngleY);
    const ScalarType sy = std::sin(m_AngleY);
    const ScalarType cz = std::cos(m_AngleZ);
    const ScalarType sz = std::sin(m_AngleZ);
    const ScalarType one = NumericTraits<ScalarType>::OneValue();
    const ScalarType zero = NumericTraits<ScalarType>::ZeroValue();

    Matrix<TScalarType, 3, 3> RotationX;
      RotationX[0][0] = one;  RotationX[0][1] = zero; RotationX[0][2] = zero;
      RotationX[1][0] = zero; RotationX[1][1] = cx;   RotationX[1][2] = -sx;
      RotationX[2][0] = zero; RotationX[2][1] = sx;   RotationX[2][2] = cx;

      Matrix<TScalarType, 3, 3> RotationY;
      RotationY[0][0] = cy;   RotationY[0][1] = zero; RotationY[0][2] = sy;
      RotationY[1][0] = zero; RotationY[1][1] = one;  RotationY[1][2] = zero;
      RotationY[2][0] = -sy;  RotationY[2][1] = zero; RotationY[2][2] = cy;

      Matrix<TScalarType, 3, 3> RotationZ;
      RotationZ[0][0] = cz;   RotationZ[0][1] = -sz;  RotationZ[0][2] = zero;
      RotationZ[1][0] = sz;   RotationZ[1][1] = cz;   RotationZ[1][2] = zero;
      RotationZ[2][0] = zero; RotationZ[2][1] = zero; RotationZ[2][2] = one;

      this->SetVarMatrix(RotationZ * RotationY * RotationX);
}

// Compute the Jacobian in one position
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeJacobianWithRespectToParameters(const InputPointType & p, JacobianType & jacobian) const
{
  // This will not reallocate memory if the dimensions are equal
  // to the matrix's current dimensions.


    jacobian.SetSize( NOutputDimensions, this->GetNumberOfLocalParameters() );
    jacobian.Fill(0.0);

    JacobianType  delx1_delphi;
    delx1_delphi.SetSize( NOutputDimensions, this->GetNumberOfLocalParameters() );
    delx1_delphi.Fill(0);

    JacobianType  delx2_delphi;
    delx2_delphi.SetSize( NOutputDimensions, this->GetNumberOfLocalParameters() );
    delx2_delphi.Fill(0);


    double m_AngleX= this->m_Parameters[3];
    double m_AngleY= this->m_Parameters[4];
    double m_AngleZ= this->m_Parameters[5];

    const double cx = std::cos(m_AngleX);
     const double sx = std::sin(m_AngleX);
     const double cy = std::cos(m_AngleY);
     const double sy = std::sin(m_AngleY);
     const double cz = std::cos(m_AngleZ);
     const double sz = std::sin(m_AngleZ);



     OutputPointType ps[4];
     ps[0]= p;
     ps[1]= m_Matrix * p + m_Translation;
     ps[2]=ps[1];
     ps[2][this->phase]=     this->m_Parameters[6]*ps[1][0]      + this->m_Parameters[7] *ps[1][1]      + this->m_Parameters[8] *ps[1][2]+
                             this->m_Parameters[9]*ps[1][0]*ps[1][1] + this->m_Parameters[10]*ps[1][0]*ps[1][2] + this->m_Parameters[11]*ps[1][1]*ps[1][2]+
                             this->m_Parameters[12]*(ps[1][0]*ps[1][0]-ps[1][1]*ps[1][1]) + this->m_Parameters[13]*(2*ps[1][2]*ps[1][2]-ps[1][0]*ps[1][0]-ps[1][1]*ps[1][1]);



     ps[3]=ps[2];
     if(this->do_cubic)
         ps[3][this->phase]+=    this->m_Parameters[14]*ps[2][0]*ps[2][1]*ps[2][2] +
             this->m_Parameters[15]*(3*ps[2][0]*ps[2][0]*ps[2][1] -ps[2][1]*ps[2][1]*ps[2][1]) +  this->m_Parameters[16]*(3*ps[2][2]*ps[2][2]*ps[2][1] -ps[2][1]*ps[2][1]*ps[2][1]) +
             this->m_Parameters[17]*(3*ps[2][0]*ps[2][0]*ps[2][2] -ps[2][2]*ps[2][2]*ps[2][2]) +  this->m_Parameters[18]*(3*ps[2][1]*ps[2][1]*ps[2][2] -ps[2][2]*ps[2][2]*ps[2][2]) +
             this->m_Parameters[19]*(3*ps[2][1]*ps[2][1]*ps[2][0] -ps[2][0]*ps[2][0]*ps[2][0]) +  this->m_Parameters[20]*(3*ps[2][2]*ps[2][2]*ps[2][0] -ps[2][0]*ps[2][0]*ps[2][0]) ;



     double px = p[0];
     double py = p[1];
     double pz = p[2];



     delx1_delphi[0][3] = ( cz * sy * cx + sz * sx ) * py + ( -cz * sy * sx + sz * cx ) * pz;
     delx1_delphi[1][3] = ( sz * sy * cx - cz * sx ) * py + ( -sz * sy * sx - cz * cx ) * pz;
     delx1_delphi[2][3] = ( cy * cx ) * py + ( -cy * sx ) * pz;

     delx1_delphi[0][4] = ( -cz * sy ) * px + ( cz * cy * sx ) * py + ( cz * cy * cx ) * pz;
     delx1_delphi[1][4] = ( -sz * sy ) * px + ( sz * cy * sx ) * py + ( sz * cy * cx ) * pz;
     delx1_delphi[2][4] = ( -cy ) * px + ( -sy * sx ) * py + ( -sy * cx ) * pz;

     delx1_delphi[0][5] = ( -sz * cy ) * px + ( -sz * sy * sx - cz * cx ) * py      + ( -sz * sy * cx + cz * sx ) * pz;
     delx1_delphi[1][5] = ( cz * cy ) * px + ( cz * sy * sx - sz * cx ) * py + ( cz * sy * cx + sz * sx ) * pz;
     delx1_delphi[2][5] = 0;

     delx1_delphi[0][0]=1;
     delx1_delphi[1][1]=1;
     delx1_delphi[2][2]=1;

     for(int i=0;i<NOutputDimensions;i++)
             for(int j=0;j<this->GetNumberOfLocalParameters();j++)
                 delx2_delphi[i][j]= delx1_delphi[i][j];

     for(int i=0;i<6;i++)
         delx2_delphi[this->phase][i]  = this->m_Parameters[6]* delx1_delphi[0][i] +  this->m_Parameters[7]* delx1_delphi[1][i] +this-> m_Parameters[8]* delx1_delphi[2][i]   +
           this->m_Parameters[9]*(delx1_delphi[0][i]*ps[1][1]+ps[1][0]*delx1_delphi[1][i]) + this-> m_Parameters[10]*(delx1_delphi[0][i]*ps[1][2]+ps[1][0]*delx1_delphi[2][i]) + this->m_Parameters[11]*(delx1_delphi[1][i]*ps[1][2]+ps[1][1]*delx1_delphi[2][i]) +
           this->m_Parameters[12]*(2*ps[1][0]*delx1_delphi[0][i] - 2*ps[1][1]*delx1_delphi[1][i] ) + this->m_Parameters[12]* ( 4* ps[1][2]*delx1_delphi[2][i]- 2*ps[1][0]*delx1_delphi[0][i]- 2*ps[1][1]*delx1_delphi[1][i]);



     delx2_delphi[this->phase][6]=ps[1][0];
     delx2_delphi[this->phase][7]=ps[1][1];
     delx2_delphi[this->phase][8]=ps[1][2];
     delx2_delphi[this->phase][9]=ps[1][0]*ps[1][1];
     delx2_delphi[this->phase][10]=ps[1][0]*ps[1][2];
     delx2_delphi[this->phase][11]=ps[1][1]*ps[1][2];
     delx2_delphi[this->phase][12]=ps[1][0]*ps[1][0] -  ps[1][1]*ps[1][1];
     delx2_delphi[this->phase][13]=2*ps[1][2]*ps[1][2] -  ps[1][1]*ps[1][1]-ps[1][0]*ps[1][0];


     for(int i=0;i<NOutputDimensions;i++)
             for(int j=0;j<this->GetNumberOfLocalParameters();j++)
                 jacobian[i][j]= delx2_delphi[i][j];

     if(this->do_cubic)
     {
         for(int i=0;i<14;i++)
         {
             jacobian[this->phase][i]+= this->m_Parameters[14]*  (delx2_delphi[0][i]*ps[2][1]*ps[2][2] +  ps[2][0]* delx2_delphi[1][i]* ps[2][2] +  ps[2][0]*ps[2][1]*delx2_delphi[2][i]   ) +
                        this->m_Parameters[15] * (6* ps[2][0]*ps[2][1]*delx2_delphi[0][i]  + 3*ps[2][0]*ps[2][0]*delx2_delphi[1][i]        -  3*ps[2][1]*ps[2][1]* delx2_delphi[1][i] ) +
                        this->m_Parameters[16] * (6* ps[2][2]*ps[2][1]*delx2_delphi[2][i]  + 3*ps[2][2]*ps[2][2]*delx2_delphi[1][i]        -  3*ps[2][1]*ps[2][1]* delx2_delphi[1][i] ) +
                        this->m_Parameters[17] * (6* ps[2][0]*ps[2][2]*delx2_delphi[0][i]  + 3*ps[2][0]*ps[2][0]*delx2_delphi[2][i]        -  3*ps[2][2]*ps[2][2]* delx2_delphi[2][i] ) +
                        this->m_Parameters[18] * (6* ps[2][1]*ps[2][2]*delx2_delphi[1][i]  + 3*ps[2][1]*ps[2][1]*delx2_delphi[2][i]        -  3*ps[2][2]*ps[2][2]* delx2_delphi[2][i] ) +
                        this->m_Parameters[19] * (6* ps[2][1]*ps[2][0]*delx2_delphi[1][i]  + 3*ps[2][1]*ps[2][1]*delx2_delphi[0][i]        -  3*ps[2][0]*ps[2][0]* delx2_delphi[0][i] ) +
                        this->m_Parameters[20] * (6* ps[2][2]*ps[2][0]*delx2_delphi[2][i]  + 3*ps[2][2]*ps[2][2]*delx2_delphi[0][i]        -  3*ps[2][0]*ps[2][0]* delx2_delphi[0][i] ) ;
         }
         jacobian[this->phase][14]= ps[2][0]*ps[2][1]*ps[2][2];
         jacobian[this->phase][15]= 3*ps[2][0]*ps[2][0]*ps[2][1]- ps[2][1]*ps[2][1]*ps[2][1];
         jacobian[this->phase][16]= 3*ps[2][2]*ps[2][2]*ps[2][1]- ps[2][1]*ps[2][1]*ps[2][1];
         jacobian[this->phase][17]= 3*ps[2][0]*ps[2][0]*ps[2][2]- ps[2][2]*ps[2][2]*ps[2][2];
         jacobian[this->phase][18]= 3*ps[2][1]*ps[2][1]*ps[2][2]- ps[2][2]*ps[2][2]*ps[2][2];
         jacobian[this->phase][19]= 3*ps[2][1]*ps[2][1]*ps[2][0]- ps[2][0]*ps[2][0]*ps[2][0];
         jacobian[this->phase][20]= 3*ps[2][2]*ps[2][2]*ps[2][0]- ps[2][0]*ps[2][0]*ps[2][0];
     }

}



// Return jacobian with respect to position.
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeJacobianWithRespectToPosition(const InputPointType  &p,
                                       JacobianType & jac) const
{
       
  jac.SetSize(3,3);

  for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
          jac[i][j]=this->GetMatrix()[i][j];

  double x0= p[0];
  double y0= p[1];
  double z0= p[2];

  OutputPointType p_temp=m_Matrix * p + m_Translation;
  double x1=p_temp[0];
  double y1=p_temp[1];
  double z1=p_temp[2];



  double dely2_dely1= this->m_Parameters[7] + this->m_Parameters[9]*x1 + this->m_Parameters[11]*z1 -2*y1*this->m_Parameters[12]-2*y1*this->m_Parameters[13];
  double dely2_delx1= this->m_Parameters[6] + this->m_Parameters[9]*y1 + this->m_Parameters[10]*z1 + this->m_Parameters[12]*2*x1 - 2*x1*this->m_Parameters[13];
  double dely2_delz1= this->m_Parameters[8] + this->m_Parameters[10]*x1 + this->m_Parameters[11]*y1 + 4*z1*this->m_Parameters[13];


  double dely2_delx0= dely2_dely1*m_Matrix[1][0] + dely2_delx1*m_Matrix[0][0] + dely2_delz1*m_Matrix[2][0];
  double dely2_dely0= dely2_dely1*m_Matrix[1][1] + dely2_delx1*m_Matrix[0][1] + dely2_delz1*m_Matrix[2][1];
  double dely2_delz0= dely2_dely1*m_Matrix[1][2] + dely2_delx1*m_Matrix[0][2] + dely2_delz1*m_Matrix[2][2];

  if(!this->do_cubic)
  {
      jac[this->phase][0]=dely2_delx0;
      jac[this->phase][1]=dely2_dely0;
      jac[this->phase][2]=dely2_delz0;
  }
  else
  {
      p_temp[this->phase]=       this->m_Parameters[6]*x1      + this->m_Parameters[7] *y1      + this->m_Parameters[8] *z1+
                                 this->m_Parameters[9]*x1*y1 + this->m_Parameters[10]*x1*z1 + this->m_Parameters[11]*y1*z1+
                                 this->m_Parameters[12]*(x1*x1-y1*y1) + this->m_Parameters[13]*(2*z1*z1-x1*x1*y1*y1);

      double x2=p_temp[0];
      double y2=p_temp[1];
      double z2=p_temp[2];

      double dely3_dely2=  (int)(this->phase==1) +    this->m_Parameters[14]*x2*z2 + this->m_Parameters[15]*3*(x2*x2-y2*y2) + this->m_Parameters[16]*3*(z2*z2-y2*y2) + this->m_Parameters[18]*6*y2*z2 + this->m_Parameters[19]*6*y2*x2;
      double dely3_delx2=  (int)(this->phase==0) +    this->m_Parameters[14]*y2*z2 + this->m_Parameters[15]*6*x2*y2 + this->m_Parameters[17]*6*x2*z2 + this->m_Parameters[19]*3*(y2*y2-x2*x2) + this->m_Parameters[20]*3*(z2*z2-x2*x2);
      double dely3_delz2=  (int)(this->phase==2) +    this->m_Parameters[14]*y2*x2 + this->m_Parameters[16]*6*z2*y2 + this->m_Parameters[17]*3*(x2*x2-z2*z2) + this->m_Parameters[18]*3*(y2*y2-z2*z2) + this->m_Parameters[20]*6*z2*x2;


      double dely3_delx0= dely3_dely2*dely2_delx0 + dely3_delx2*m_Matrix[0][0]+ dely3_delz2*m_Matrix[2][0];
      double dely3_dely0= dely3_dely2*dely2_dely0 + dely3_delx2*m_Matrix[0][1]+ dely3_delz2*m_Matrix[2][1];
      double dely3_delz0= dely3_dely2*dely2_delz0 + dely3_delx2*m_Matrix[0][2]+ dely3_delz2*m_Matrix[2][2];

      jac[this->phase][0]=dely3_delx0;
      jac[this->phase][1]=dely3_dely0;
      jac[this->phase][2]=dely3_delz0;
  }

}








// Return jacobian with respect to position.
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeInverseJacobianWithRespectToPosition(const InputPointType  &p,
                                       JacobianType & jac) const
{
    
  ComputeJacobianWithRespectToPosition(p,jac);

  typedef itk::Matrix<double, 3, 3> mMatrixType;
  mMatrixType M;
  for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
          M(i,j)=jac(i,j);

  

  M=M.GetInverse();

 
  for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
          jac(i,j)=M(i,j);    
}


// Computes translation based on offset, matrix, and center
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeTranslation(void)
{
  m_Translation[0]=this->m_Parameters[0];
  m_Translation[1]=this->m_Parameters[1];
  m_Translation[2]=this->m_Parameters[2];
}



// Computes parameters - base class does nothing.  In derived classes is
//    used to convert, for example, matrix into a versor
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void OkanQuadraticTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeMatrixParameters(void)
{

}

} // namespace

#endif
 
