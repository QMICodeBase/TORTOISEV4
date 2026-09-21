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
#ifndef itkEulerPhaseScale3DTransform_hxx
#define itkEulerPhaseScale3DTransform_hxx


namespace itk
{

template <typename TParametersValueType>
EulerPhaseScale3DTransform<TParametersValueType>::EulerPhaseScale3DTransform()
  : Superclass(ParametersDimension)

{
  m_AngleX = m_AngleY = m_AngleZ = ScalarType{};

  m_PhaseAxis=1;
  m_PhaseScaleShear.fill(0);
  m_PhaseScaleShear[this->m_PhaseAxis]=1;
  this->m_FixedParameters.SetSize(SpaceDimension + 1);
  this->m_FixedParameters.Fill(0.0);
}

template <typename TParametersValueType>
EulerPhaseScale3DTransform<TParametersValueType>::EulerPhaseScale3DTransform(const MatrixType & matrix, const OutputPointType & offset)

{
  this->SetMatrix(matrix);

  OffsetType off;
  off[0] = offset[0];
  off[1] = offset[1];
  off[2] = offset[2];
  this->SetOffset(off);
  this->m_FixedParameters.SetSize(SpaceDimension + 1);
  this->m_FixedParameters.Fill(0.0);
}

template <typename TParametersValueType>
EulerPhaseScale3DTransform<TParametersValueType>::EulerPhaseScale3DTransform(unsigned int parametersDimension)
  : Superclass(parametersDimension)

{

  m_AngleX = m_AngleY = m_AngleZ = ScalarType{};
  m_PhaseAxis=1;
  m_PhaseScaleShear.fill(0);
  m_PhaseScaleShear[this->m_PhaseAxis]=1;

  this->m_FixedParameters.SetSize(SpaceDimension + 1);
  this->m_FixedParameters.Fill(0.0);
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::SetVarRotation(ScalarType angleX, ScalarType angleY, ScalarType angleZ)
{
  this->m_AngleX = angleX;
  this->m_AngleY = angleY;
  this->m_AngleZ = angleZ;
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::SetParameters(const ParametersType & parameters)
{
  itkDebugMacro("Setting parameters " << parameters);

  // Save parameters. Needed for proper operation of TransformUpdateParameters.
  if (&parameters != &(this->m_Parameters))
  {
    this->m_Parameters = parameters;
  }

  // Set angles with parameters
  m_AngleX = parameters[0];
  m_AngleY = parameters[1];
  m_AngleZ = parameters[2];

  m_PhaseScaleShear[0] = parameters[6];
  m_PhaseScaleShear[1] = parameters[7];
  m_PhaseScaleShear[2] = parameters[8];

  this->ComputeMatrix();

  // Transfer the translation part
  OutputVectorType newTranslation;
  newTranslation[0] = parameters[3];
  newTranslation[1] = parameters[4];
  newTranslation[2] = parameters[5];
  this->SetVarTranslation(newTranslation);
  this->ComputeOffset();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

  itkDebugMacro("After setting parameters ");
}

template <typename TParametersValueType>
auto
EulerPhaseScale3DTransform<TParametersValueType>::GetParameters() const -> const ParametersType &
{
  this->m_Parameters[0] = m_AngleX;
  this->m_Parameters[1] = m_AngleY;
  this->m_Parameters[2] = m_AngleZ;
  this->m_Parameters[3] = this->GetTranslation()[0];
  this->m_Parameters[4] = this->GetTranslation()[1];
  this->m_Parameters[5] = this->GetTranslation()[2];
  this->m_Parameters[6] = m_PhaseScaleShear[0];
  this->m_Parameters[7] = m_PhaseScaleShear[1];
  this->m_Parameters[8] = m_PhaseScaleShear[2];

  return this->m_Parameters;
}

template <typename TParametersValueType>
auto
EulerPhaseScale3DTransform<TParametersValueType>::GetFixedParameters() const -> const FixedParametersType &
{
  // Call the superclass GetFixedParameters so that it fills the
  // array, we ignore the returned data and add the additional
  // information to the updated array.
  Superclass::GetFixedParameters();
  this->m_FixedParameters[3] = this->m_ComputeZYX ? 1.0 : 0.0;
  return this->m_FixedParameters;
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::SetFixedParameters(const FixedParametersType & parameters)
{
  if (parameters.size() < InputSpaceDimension)
  {
    itkExceptionMacro("Error setting fixed parameters: parameters array size ("
                      << parameters.size() << ") is less than expected  (InputSpaceDimension = " << InputSpaceDimension
                      << ')');
  }

  InputPointType c;
  for (unsigned int i = 0; i < InputSpaceDimension; ++i)
  {
    c[i] = this->m_FixedParameters[i] = parameters[i];
  }
  this->SetCenter(c);
  // conditional is here for backwards compatibility: the
  // m_ComputeZYX flag was not serialized so it may or may
  // not be included as part of the fixed parameters
  if (parameters.Size() == 4)
  {
    this->m_FixedParameters[3] = parameters[3];
    this->SetComputeZYX(this->m_FixedParameters[3] != 0.0);
  }
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::SetRotation(ScalarType angleX, ScalarType angleY, ScalarType angleZ)
{
  m_AngleX = angleX;
  m_AngleY = angleY;
  m_AngleZ = angleZ;
  this->ComputeMatrix();
  this->ComputeOffset();
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::SetIdentity()
{
  Superclass::SetIdentity();
  m_AngleX = 0;
  m_AngleY = 0;
  m_AngleZ = 0;
  m_PhaseScaleShear.fill(0);
  m_PhaseScaleShear[this->m_PhaseAxis]=1;
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::ComputeMatrixParameters()
{
    vnl_matrix_fixed<double,3,3> Rmat, Smat_inv;
    Smat_inv.set_identity();

    double Kx=m_PhaseScaleShear[0];
    double Ky=m_PhaseScaleShear[1];
    double Kz=m_PhaseScaleShear[2];

    if(this->m_PhaseAxis==0)
    {
        Smat_inv(0,0)=1./Kx;
        Smat_inv(0,1)=-Ky/Kx;
        Smat_inv(0,2)=-Kz/Kx;
    }
    if(this->m_PhaseAxis==1)
    {
        Smat_inv(1,0)=-Kx/Ky;
        Smat_inv(1,1)=1./Ky;
        Smat_inv(1,2)=-Kz/Ky;
    }
    if(this->m_PhaseAxis==2)
    {
        Smat_inv(2,0)=-Kx/Kz;
        Smat_inv(2,1)=-Ky/Kz;
        Smat_inv(2,2)=1./Kz;
    }

    Rmat= Smat_inv * this->GetMatrix().GetVnlMatrix();

  if (m_ComputeZYX)
  {
    m_AngleY = -std::asin(Rmat[2][0]);
    const double C = std::cos(m_AngleY);
    if (itk::Math::abs(C) > 0.00005)
    {
      double x = Rmat[2][2] / C;
      double y = Rmat[2][1] / C;
      m_AngleX = std::atan2(y, x);
      x = Rmat[0][0] / C;
      y = Rmat[1][0] / C;
      m_AngleZ = std::atan2(y, x);
    }
    else
    {
      m_AngleX = ScalarType{};
      const double x = Rmat[1][1];
      const double y = -Rmat[0][1];
      m_AngleZ = std::atan2(y, x);
    }
  }
  else
  {
    m_AngleX = std::asin(Rmat[2][1]);
    const double A = std::cos(m_AngleX);
    if (itk::Math::abs(A) > 0.00005)
    {
      double x = Rmat[2][2] / A;
      double y = -Rmat[2][0] / A;
      m_AngleY = std::atan2(y, x);

      x = Rmat[1][1] / A;
      y = -Rmat[0][1] / A;
      m_AngleZ = std::atan2(y, x);
    }
    else
    {
      m_AngleZ = ScalarType{};
      const double x = Rmat[0][0];
      const double y = Rmat[1][0];
      m_AngleY = std::atan2(y, x);
    }
  }
  this->ComputeMatrix();
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::ComputeMatrix()
{
  // need to check if angles are in the right order
  const ScalarType     cx = std::cos(m_AngleX);
  const ScalarType     sx = std::sin(m_AngleX);
  const ScalarType     cy = std::cos(m_AngleY);
  const ScalarType     sy = std::sin(m_AngleY);
  const ScalarType     cz = std::cos(m_AngleZ);
  const ScalarType     sz = std::sin(m_AngleZ);
  const ScalarType     one = NumericTraits<ScalarType>::OneValue();
  constexpr ScalarType zero{};

  Matrix<TParametersValueType, 3, 3> RotationX;
  RotationX[0][0] = one;
  RotationX[0][1] = zero;
  RotationX[0][2] = zero;
  RotationX[1][0] = zero;
  RotationX[1][1] = cx;
  RotationX[1][2] = -sx;
  RotationX[2][0] = zero;
  RotationX[2][1] = sx;
  RotationX[2][2] = cx;

  Matrix<TParametersValueType, 3, 3> RotationY;
  RotationY[0][0] = cy;
  RotationY[0][1] = zero;
  RotationY[0][2] = sy;
  RotationY[1][0] = zero;
  RotationY[1][1] = one;
  RotationY[1][2] = zero;
  RotationY[2][0] = -sy;
  RotationY[2][1] = zero;
  RotationY[2][2] = cy;

  Matrix<TParametersValueType, 3, 3> RotationZ;
  RotationZ[0][0] = cz;
  RotationZ[0][1] = -sz;
  RotationZ[0][2] = zero;
  RotationZ[1][0] = sz;
  RotationZ[1][1] = cz;
  RotationZ[1][2] = zero;
  RotationZ[2][0] = zero;
  RotationZ[2][1] = zero;
  RotationZ[2][2] = one;

  Matrix<TParametersValueType, 3, 3> ScaleMatrix;
  ScaleMatrix.SetIdentity();
  ScaleMatrix[m_PhaseAxis][0]=m_PhaseScaleShear[0];
  ScaleMatrix[m_PhaseAxis][1]=m_PhaseScaleShear[1];
  ScaleMatrix[m_PhaseAxis][2]=m_PhaseScaleShear[2];

  // Apply the rotation first around Y then X then Z
  if (m_ComputeZYX)
  {
    this->SetVarMatrix(ScaleMatrix* RotationZ * RotationY * RotationX);
  }
  else
  {
    // Like VTK transformation order
    this->SetVarMatrix(ScaleMatrix * RotationZ * RotationX * RotationY);
  }
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::ComputeJacobianWithRespectToParameters(const InputPointType & p,
                                                                               JacobianType &         jacobian) const
{
  // need to check if angles are in the right order
  const double cx = std::cos(m_AngleX);
  const double sx = std::sin(m_AngleX);
  const double cy = std::cos(m_AngleY);
  const double sy = std::sin(m_AngleY);
  const double cz = std::cos(m_AngleZ);
  const double sz = std::sin(m_AngleZ);

  jacobian.SetSize(3, this->GetNumberOfLocalParameters());
  jacobian.Fill(0.0);

  const double px = p[0] - this->GetCenter()[0];
  const double py = p[1] - this->GetCenter()[1];
  const double pz = p[2] - this->GetCenter()[2];

  const double Kx= this->m_PhaseScaleShear[0];
  const double Ky= this->m_PhaseScaleShear[1];
  const double Kz= this->m_PhaseScaleShear[2];


  if (m_ComputeZYX)
  {
      if(m_PhaseAxis==0)
      {


        jacobian[0][0]= - py*(sx*(Ky*cz - Kx*sz) - cx*(Kz*cy + sy*(Kx*cz + Ky*sz))) - pz*(cx*(Ky*cz - Kx*sz) + sx*(Kz*cy + sy*(Kx*cz + Ky*sz)));
        jacobian[1][0]= - py*(cz*sx - cx*sy*sz) - pz*(cx*cz + sx*sy*sz);
        jacobian[2][0]= py*cx*cy - pz*cy*sx;

        jacobian[0][1]= - px*(Kz*cy + sy*(Kx*cz + Ky*sz)) - pz*cx*(Kz*sy - cy*(Kx*cz + Ky*sz)) - py*sx*(Kz*sy - cy*(Kx*cz + Ky*sz));
        jacobian[1][1]= pz*cx*cy*sz - px*sy*sz + py*cy*sx*sz;
        jacobian[2][1]= - px*cy - pz*cx*sy - py*sx*sy;

        jacobian[0][2]= pz*(sx*(Kx*cz + Ky*sz) + cx*sy*(Ky*cz - Kx*sz)) - py*(cx*(Kx*cz + Ky*sz) - sx*sy*(Ky*cz - Kx*sz)) + px*cy*(Ky*cz - Kx*sz);
        jacobian[1][2]=pz*(sx*sz + cx*cz*sy) - py*(cx*sz - cz*sx*sy) + px*cy*cz;
        jacobian[2][2]=0;



        jacobian[0][6]= pz*(sx*sz + cx*cz*sy) - py*(cx*sz - cz*sx*sy) + px*cy*cz;
        jacobian[1][6]= 0;
        jacobian[2][6]= 0;

        jacobian[0][7]= py*(cx*cz + sx*sy*sz) - pz*(cz*sx - cx*sy*sz) + px*cy*sz;
        jacobian[1][7]=      0;
        jacobian[2][7]=      0;

        jacobian[0][8]=pz*cx*cy - px*sy + py*cy*sx;
        jacobian[1][8]=      0;
        jacobian[2][8]=      0;
      }
      if(m_PhaseAxis==1)
      {
        jacobian[0][0]=py*(sx*sz + cx*cz*sy) + pz*(cx*sz - cz*sx*sy);
        jacobian[1][0]=- py*(sx*(Ky*cz - Kx*sz) - cx*(Kz*cy + sy*(Kx*cz + Ky*sz))) - pz*(cx*(Ky*cz - Kx*sz) + sx*(Kz*cy + sy*(Kx*cz + Ky*sz)));
        jacobian[2][0]=py*cx*cy - pz*cy*sx;

        jacobian[0][1]=  pz*cx*cy*cz - px*cz*sy + py*cy*cz*sx;
        jacobian[1][1]=  - px*(Kz*cy + sy*(Kx*cz + Ky*sz)) - pz*cx*(Kz*sy - cy*(Kx*cz + Ky*sz)) - py*sx*(Kz*sy - cy*(Kx*cz + Ky*sz));
        jacobian[2][1]=  - px*cy - pz*cx*sy - py*sx*sy;

        jacobian[0][2]=pz*(cz*sx - cx*sy*sz) - py*(cx*cz + sx*sy*sz) - px*cy*sz;
        jacobian[1][2]=pz*(sx*(Kx*cz + Ky*sz) + cx*sy*(Ky*cz - Kx*sz)) - py*(cx*(Kx*cz + Ky*sz) - sx*sy*(Ky*cz - Kx*sz)) + px*cy*(Ky*cz - Kx*sz);
        jacobian[2][2]=0;


        jacobian[0][6]=0;
        jacobian[1][6]=pz*(sx*sz + cx*cz*sy) - py*(cx*sz - cz*sx*sy) + px*cy*cz;
        jacobian[2][6]=0;

        jacobian[0][7]=0;
        jacobian[1][7]=py*(cx*cz + sx*sy*sz) - pz*(cz*sx - cx*sy*sz) + px*cy*sz;
        jacobian[2][7]=0;

        jacobian[0][8]=0;
        jacobian[1][8]=pz*cx*cy - px*sy + py*cy*sx;
        jacobian[2][8]=0;
      }
      if(m_PhaseAxis==2)
      {          
        jacobian[0][0]=py*(sx*sz + cx*cz*sy) + pz*(cx*sz - cz*sx*sy);
        jacobian[1][0]=- py*(cz*sx - cx*sy*sz) - pz*(cx*cz + sx*sy*sz);
        jacobian[2][0]=- py*(sx*(Ky*cz - Kx*sz) - cx*(Kz*cy + sy*(Kx*cz + Ky*sz))) - pz*(cx*(Ky*cz - Kx*sz) + sx*(Kz*cy + sy*(Kx*cz + Ky*sz)));

        jacobian[0][1]=pz*cx*cy*cz - px*cz*sy + py*cy*cz*sx;
        jacobian[1][1]=pz*cx*cy*sz - px*sy*sz + py*cy*sx*sz;
        jacobian[2][1]=- px*(Kz*cy + sy*(Kx*cz + Ky*sz)) - pz*cx*(Kz*sy - cy*(Kx*cz + Ky*sz)) - py*sx*(Kz*sy - cy*(Kx*cz + Ky*sz));

        jacobian[0][2]=pz*(cz*sx - cx*sy*sz) - py*(cx*cz + sx*sy*sz) - px*cy*sz;
        jacobian[1][2]=pz*(sx*sz + cx*cz*sy) - py*(cx*sz - cz*sx*sy) + px*cy*cz;
        jacobian[2][2]=pz*(sx*(Kx*cz + Ky*sz) + cx*sy*(Ky*cz - Kx*sz)) - py*(cx*(Kx*cz + Ky*sz) - sx*sy*(Ky*cz - Kx*sz)) + px*cy*(Ky*cz - Kx*sz);


        jacobian[0][6]=0;
        jacobian[1][6]=0;
        jacobian[2][6]=pz*(sx*sz + cx*cz*sy) - py*(cx*sz - cz*sx*sy) + px*cy*cz;

        jacobian[0][7]=0;
        jacobian[1][7]=0;
        jacobian[2][7]=py*(cx*cz + sx*sy*sz) - pz*(cz*sx - cx*sy*sz) + px*cy*sz;

        jacobian[0][8]=0;
        jacobian[1][8]=0;
        jacobian[2][8]=pz*cx*cy - px*sy + py*cy*sx;
      }
  }
  else
  {
      if(m_PhaseAxis==0)
      {                    
        jacobian[0][0]=py*(Kz*cx - sx*(Ky*cz - Kx*sz)) - pz*cy*(Kz*sx + cx*(Ky*cz - Kx*sz)) + px*sy*(Kz*sx + cx*(Ky*cz - Kx*sz));
        jacobian[1][0]=px*cx*cz*sy - pz*cx*cy*cz - py*cz*sx;
        jacobian[2][0]=py*cx - pz*cy*sx + px*sx*sy;

        jacobian[0][1]=pz*(cy*(Kx*cz + Ky*sz) - sy*(Kz*cx - sx*(Ky*cz - Kx*sz))) - px*(sy*(Kx*cz + Ky*sz) + cy*(Kz*cx - sx*(Ky*cz - Kx*sz)));
        jacobian[1][1]=pz*(cy*sz + cz*sx*sy) - px*(sy*sz - cy*cz*sx);
        jacobian[2][1]=- px*cx*cy - pz*cx*sy;

        jacobian[0][2]=px*(cy*(Ky*cz - Kx*sz) - sx*sy*(Kx*cz + Ky*sz)) + pz*(sy*(Ky*cz - Kx*sz) + cy*sx*(Kx*cz + Ky*sz)) - py*cx*(Kx*cz + Ky*sz);
        jacobian[1][2]=px*(cy*cz - sx*sy*sz) + pz*(cz*sy + cy*sx*sz) - py*cx*sz;
        jacobian[2][2]=0;


        jacobian[0][6]=px*(cy*cz - sx*sy*sz) + pz*(cz*sy + cy*sx*sz) - py*cx*sz;
        jacobian[1][6]=0;
        jacobian[2][6]=0;

        jacobian[0][7]=px*(cy*sz + cz*sx*sy) + pz*(sy*sz - cy*cz*sx) + py*cx*cz;
        jacobian[1][7]=0;
        jacobian[2][7]=0;

        jacobian[0][8]=py*sx + pz*cx*cy - px*cx*sy;
        jacobian[1][8]=0;
        jacobian[2][8]=0;

      }
      if(m_PhaseAxis==1)
      {
        jacobian[0][0]=py*sx*sz + pz*cos(m_AngleX)*cy*sz - px*cos(m_AngleX)*sy*sz;
        jacobian[1][0]=py*(Kz*cos(m_AngleX) - sx*(Ky*cz - Kx*sz)) - pz*cy*(Kz*sx + cos(m_AngleX)*(Ky*cz - Kx*sz)) + px*sy*(Kz*sx + cos(m_AngleX)*(Ky*cz - Kx*sz));
        jacobian[2][0]=py*cos(m_AngleX) - pz*cy*sx + px*sx*sy;

        jacobian[0][1]=pz*(cy*cz - sx*sy*sz) - px*(cz*sy + cy*sx*sz);
        jacobian[1][1]=pz*(cy*(Kx*cz + Ky*sz) - sy*(Kz*cos(m_AngleX) - sx*(Ky*cz - Kx*sz))) - px*(sy*(Kx*cz + Ky*sz) + cy*(Kz*cos(m_AngleX) - sx*(Ky*cz - Kx*sz)));
        jacobian[2][1]=- px*cos(m_AngleX)*cy - pz*cos(m_AngleX)*sy;

        jacobian[0][2]=- px*(cy*sz + cz*sx*sy) - pz*(sy*sz - cy*cz*sx) - py*cos(m_AngleX)*cz;
        jacobian[1][2]=px*(cy*(Ky*cz - Kx*sz) - sx*sy*(Kx*cz + Ky*sz)) + pz*(sy*(Ky*cz - Kx*sz) + cy*sx*(Kx*cz + Ky*sz)) - py*cos(m_AngleX)*(Kx*cz + Ky*sz);
        jacobian[2][2]=0;


        jacobian[0][6]=0;
        jacobian[1][6]=px*(cy*cz - sx*sy*sz) + pz*(cz*sy + cy*sx*sz) - py*cos(m_AngleX)*sz;
        jacobian[2][6]=0;

        jacobian[0][7]=0;
        jacobian[1][7]=px*(cy*sz + cz*sx*sy) + pz*(sy*sz - cy*cz*sx) + py*cos(m_AngleX)*cz;
        jacobian[2][7]=0;

        jacobian[0][8]=0;
        jacobian[1][8]=py*sx + pz*cos(m_AngleX)*cy - px*cos(m_AngleX)*sy;
        jacobian[2][8]=0;
      }
      if(m_PhaseAxis==2)
      {

        jacobian[0][0]=py*sx*sz + pz*cx*cy*sz - px*cx*sy*sz;
        jacobian[1][0]=px*cx*cz*sy - pz*cx*cy*cz - py*cz*sx;
        jacobian[2][0]=py*(Kz*cx - sx*(Ky*cz - Kx*sz)) - pz*cy*(Kz*sx + cx*(Ky*cz - Kx*sz)) + px*sy*(Kz*sx + cx*(Ky*cz - Kx*sz));

        jacobian[0][1]=pz*(cy*cz - sx*sy*sz) - px*(cz*sy + cy*sx*sz);
        jacobian[1][1]=pz*(cy*sz + cz*sx*sy) - px*(sy*sz - cy*cz*sx);
        jacobian[2][1]=pz*(cy*(Kx*cz + Ky*sz) - sy*(Kz*cx - sx*(Ky*cz - Kx*sz))) - px*(sy*(Kx*cz + Ky*sz) + cy*(Kz*cx - sx*(Ky*cz - Kx*sz)));

        jacobian[0][2]=-px*(cy*sz + cz*sx*sy) - pz*(sy*sz - cy*cz*sx) - py*cx*cz;
        jacobian[1][2]=px*(cy*cz - sx*sy*sz) + pz*(cz*sy + cy*sx*sz) - py*cx*sz;
        jacobian[2][2]=px*(cy*(Ky*cz - Kx*sz) - sx*sy*(Kx*cz + Ky*sz)) + pz*(sy*(Ky*cz - Kx*sz) + cy*sx*(Kx*cz + Ky*sz)) - py*cx*(Kx*cz + Ky*sz);


        jacobian[0][6]=0;
        jacobian[1][6]=0;
        jacobian[2][6]=px*(cy*cz - sx*sy*sz) + pz*(cz*sy + cy*sx*sz) - py*cx*sz;

        jacobian[0][7]=0;
        jacobian[1][7]=0;
        jacobian[2][7]=px*(cy*sz + cz*sx*sy) + pz*(sy*sz - cy*cz*sx) + py*cx*cz;

        jacobian[0][8]=0;
        jacobian[1][8]=0;
        jacobian[2][8]=py*sx + pz*cx*cy - px*cx*sy;
      }
  }

  // compute derivatives for the translation part
  constexpr unsigned int blockOffset = 3;
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    jacobian[dim][blockOffset + dim] = 1.0;
  }




}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::SetComputeZYX(const bool flag)
{
  if (this->m_ComputeZYX != flag)
  {
    this->m_ComputeZYX = flag;
    this->ComputeMatrix();
    this->ComputeOffset();
    // The meaning of the parameters has changed so the transform
    // has been modified even if the parameter values have not.
    this->Modified();
  }
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "AngleX: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(m_AngleX) << std::endl;
  os << indent << "AngleY: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(m_AngleY) << std::endl;
  os << indent << "AngleZ: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(m_AngleZ) << std::endl;
  os << indent << "transX: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(this->GetTranslation()[0]) << std::endl;
  os << indent << "transY: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(this->GetTranslation()[1]) << std::endl;
  os << indent << "transZ: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(this->GetTranslation()[2]) << std::endl;
  os << indent << "scalePhase0: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(m_PhaseScaleShear[0]) << std::endl;
  os << indent << "scalePhase1: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(m_PhaseScaleShear[1]) << std::endl;
  os << indent << "scalePhase2: " << static_cast<typename NumericTraits<ScalarType>::PrintType>(m_PhaseScaleShear[2]) << std::endl;

  itkPrintSelfBooleanMacro(ComputeZYX);
}

} // namespace itk

#endif

