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
    m_PhaseScale=1;
  m_PhaseAxis=1;
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
    m_PhaseScale=1;
    m_PhaseAxis=1;
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

  m_PhaseScale = parameters[6];

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
  this->m_Parameters[6] = m_PhaseScale;

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
  m_PhaseScale=1;
}

template <typename TParametersValueType>
void
EulerPhaseScale3DTransform<TParametersValueType>::ComputeMatrixParameters()
{
  if (m_ComputeZYX)
  {
    m_AngleY = -std::asin(this->GetMatrix()[2][0]);
    const double C = std::cos(m_AngleY);
    if (itk::Math::abs(C) > 0.00005)
    {
      double x = this->GetMatrix()[2][2] / C;
      double y = this->GetMatrix()[2][1] / C;
      m_AngleX = std::atan2(y, x);
      x = this->GetMatrix()[0][0] / C;
      y = this->GetMatrix()[1][0] / C;
      m_AngleZ = std::atan2(y, x);
    }
    else
    {
      m_AngleX = ScalarType{};
      const double x = this->GetMatrix()[1][1];
      const double y = -this->GetMatrix()[0][1];
      m_AngleZ = std::atan2(y, x);
    }
  }
  else
  {
    m_AngleX = std::asin(this->GetMatrix()[2][1]);
    const double A = std::cos(m_AngleX);
    if (itk::Math::abs(A) > 0.00005)
    {
      double x = this->GetMatrix()[2][2] / A;
      double y = -this->GetMatrix()[2][0] / A;
      m_AngleY = std::atan2(y, x);

      x = this->GetMatrix()[1][1] / A;
      y = -this->GetMatrix()[0][1] / A;
      m_AngleZ = std::atan2(y, x);
    }
    else
    {
      m_AngleZ = ScalarType{};
      const double x = this->GetMatrix()[0][0];
      const double y = this->GetMatrix()[1][0];
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
  ScaleMatrix[m_PhaseAxis][m_PhaseAxis]=m_PhaseScale;

  // Apply the rotation first around Y then X then Z
  if (m_ComputeZYX)
  {
    this->SetVarMatrix(RotationZ * RotationY * RotationX*ScaleMatrix);
  }
  else
  {
    // Like VTK transformation order
    this->SetVarMatrix(RotationZ * RotationX * RotationY*ScaleMatrix);
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

  const double S= m_PhaseScale;

  if (m_ComputeZYX)
  {
      if(m_PhaseAxis==0)
      {
          jacobian[0][0] = (cz * sy * cx + sz * sx) * py + (-cz * sy * sx + sz * cx) * pz;
          jacobian[1][0] = (sz * sy * cx - cz * sx) * py + (-sz * sy * sx - cz * cx) * pz;
          jacobian[2][0] = (cy * cx) * py + (-cy * sx) * pz;

          jacobian[0][1] = (-cz * sy) *  px * S  + (cz * cy * sx) * py + (cz * cy * cx) * pz;
          jacobian[1][1] = (-sz * sy) * px * S  + (sz * cy * sx) * py + (sz * cy * cx) * pz;
          jacobian[2][1] = (-cy) * px * S  + (-sy * sx) * py + (-sy * cx) * pz;

          jacobian[0][2] = (-sz * cy) * px * S  + (-sz * sy * sx - cz * cx) * py + (-sz * sy * cx + cz * sx) * pz;
          jacobian[1][2] = (cz * cy) * px * S  + (cz * sy * sx - sz * cx) * py + (cz * sy * cx + sz * sx) * pz;
          jacobian[2][2] = 0;

          jacobian[0][6]= px * cy * cz;
          jacobian[1][6]= px * cy * sz;
          jacobian[2][6]= -px * sy ;
      }
      if(m_PhaseAxis==1)
      {
          jacobian[0][0] = (cz * sy * cx + sz * sx) * py* S  + (-cz * sy * sx + sz * cx) * pz;
          jacobian[1][0] = (sz * sy * cx - cz * sx) * py* S  + (-sz * sy * sx - cz * cx) * pz;
          jacobian[2][0] = (cy * cx) * py* S  + (-cy * sx) * pz;

          jacobian[0][1] = (-cz * sy) *  px + (cz * cy * sx) * py* S  + (cz * cy * cx) * pz;
          jacobian[1][1] = (-sz * sy) * px + (sz * cy * sx) * py* S  + (sz * cy * cx) * pz;
          jacobian[2][1] = (-cy) * px + (-sy * sx) * py* S  + (-sy * cx) * pz;

          jacobian[0][2] = (-sz * cy) * px + (-sz * sy * sx - cz * cx) * py* S  + (-sz * sy * cx + cz * sx) * pz;
          jacobian[1][2] = (cz * cy) * px + (cz * sy * sx - sz * cx) * py* S  + (cz * sy * cx + sz * sx) * pz;
          jacobian[2][2] = 0;

          jacobian[0][6]= -py * (cx *sz - cz*sx*sy);
          jacobian[1][6]= py * (cx*cz + sx*sy*sz);
          jacobian[2][6]= py * cy *sx;
      }
      if(m_PhaseAxis==2)
      {
          jacobian[0][0] = (cz * sy * cx + sz * sx) * py + (-cz * sy * sx + sz * cx) * pz* S ;
          jacobian[1][0] = (sz * sy * cx - cz * sx) * py + (-sz * sy * sx - cz * cx) * pz* S ;
          jacobian[2][0] = (cy * cx) * py + (-cy * sx) * pz* S ;

          jacobian[0][1] = (-cz * sy) *  px + (cz * cy * sx) * py + (cz * cy * cx) * pz* S ;
          jacobian[1][1] = (-sz * sy) * px + (sz * cy * sx) * py + (sz * cy * cx) * pz* S ;
          jacobian[2][1] = (-cy) * px + (-sy * sx) * py + (-sy * cx) * pz* S ;

          jacobian[0][2] = (-sz * cy) * px + (-sz * sy * sx - cz * cx) * py + (-sz * sy * cx + cz * sx) * pz* S ;
          jacobian[1][2] = (cz * cy) * px + (cz * sy * sx - sz * cx) * py + (cz * sy * cx + sz * sx) * pz* S ;
          jacobian[2][2] = 0;

          jacobian[0][6]= pz* (sx*sz + cx*cz *sy);
          jacobian[1][6]= -pz* (cz*sx -cx*sy*sz);
          jacobian[2][6]= pz*cx*cy;
      }
  }
  else
  {
      if(m_PhaseAxis==0)
      {
          jacobian[0][0] = (-sz * cx * sy) * px* S  + (sz * sx) * py + (sz * cx * cy) * pz;
          jacobian[1][0] = (cz * cx * sy) * px* S  + (-cz * sx) * py + (-cz * cx * cy) * pz;
          jacobian[2][0] = (sx * sy) * px* S  + (cx)*py + (-sx * cy) * pz;

          jacobian[0][1] = (-cz * sy - sz * sx * cy) * px* S  + (cz * cy - sz * sx * sy) * pz;
          jacobian[1][1] = (-sz * sy + cz * sx * cy) * px* S  + (sz * cy + cz * sx * sy) * pz;
          jacobian[2][1] = (-cx * cy) * px* S  + (-cx * sy) * pz;

          jacobian[0][2] = (-sz * cy - cz * sx * sy) * px* S  + (-cz * cx) * py + (-sz * sy + cz * sx * cy) * pz;
          jacobian[1][2] = (cz * cy - sz * sx * sy) * px* S  + (-sz * cx) * py + (cz * sy + sz * sx * cy) * pz;
          jacobian[2][2] = 0;

          jacobian[0][6]= px* (cy*cz -sx*sy*sz);
          jacobian[1][6]= px* (cy*sz + cz*sx*sy);
          jacobian[2][6]= -px *cx*sy;
      }
      if(m_PhaseAxis==1)
      {
          jacobian[0][0] = (-sz * cx * sy) * px + (sz * sx) * py* S  + (sz * cx * cy) * pz;
          jacobian[1][0] = (cz * cx * sy) * px + (-cz * sx) * py* S  + (-cz * cx * cy) * pz;
          jacobian[2][0] = (sx * sy) * px + (cx)*py* S  + (-sx * cy) * pz;

          jacobian[0][1] = (-cz * sy - sz * sx * cy) * px + (cz * cy - sz * sx * sy) * pz;
          jacobian[1][1] = (-sz * sy + cz * sx * cy) * px + (sz * cy + cz * sx * sy) * pz;
          jacobian[2][1] = (-cx * cy) * px + (-cx * sy) * pz;

          jacobian[0][2] = (-sz * cy - cz * sx * sy) * px + (-cz * cx) * py* S  + (-sz * sy + cz * sx * cy) * pz;
          jacobian[1][2] = (cz * cy - sz * sx * sy) * px + (-sz * cx) * py* S  + (cz * sy + sz * sx * cy) * pz;
          jacobian[2][2] = 0;

          jacobian[0][6]= -py*cx*sz;
          jacobian[1][6]= py*cx*cz;
          jacobian[2][6]= py*sx;
      }
      if(m_PhaseAxis==2)
      {
          jacobian[0][0] = (-sz * cx * sy) * px + (sz * sx) * py + (sz * cx * cy) * pz* S ;
          jacobian[1][0] = (cz * cx * sy) * px + (-cz * sx) * py + (-cz * cx * cy) * pz* S ;
          jacobian[2][0] = (sx * sy) * px + (cx)*py + (-sx * cy) * pz* S ;

          jacobian[0][1] = (-cz * sy - sz * sx * cy) * px + (cz * cy - sz * sx * sy) * pz* S ;
          jacobian[1][1] = (-sz * sy + cz * sx * cy) * px + (sz * cy + cz * sx * sy) * pz* S ;
          jacobian[2][1] = (-cx * cy) * px + (-cx * sy) * pz* S ;

          jacobian[0][2] = (-sz * cy - cz * sx * sy) * px + (-cz * cx) * py + (-sz * sy + cz * sx * cy) * pz* S ;
          jacobian[1][2] = (cz * cy - sz * sx * sy) * px + (-sz * cx) * py + (cz * sy + sz * sx * cy) * pz* S ;
          jacobian[2][2] = 0;

          jacobian[0][6]= pz* (cz*sy + cy*sx*sz);
          jacobian[1][6]= pz* (sy*sz -cy*cz*sx);
          jacobian[2][6]= pz*cx*cy;
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
  itkPrintSelfBooleanMacro(ComputeZYX);
}

} // namespace itk

#endif

