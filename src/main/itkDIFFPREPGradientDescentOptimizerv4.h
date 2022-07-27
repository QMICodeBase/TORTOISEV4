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
#ifndef itkDIFFPREPGradientDescentOptimizerv4_h
#define itkDIFFPREPGradientDescentOptimizerv4_h

#include "itkGradientDescentOptimizerBasev4.h"
#include "itkOkanQuadraticTransform.h"

#ifdef USECUDA
    #include "cuda_image.h"
    #include "../cuda_src/compute_entropy.h"
#endif  

namespace itk
{

 
template<typename TInternalComputationValueType>
class ITK_TEMPLATE_EXPORT DIFFPREPGradientDescentOptimizerv4
: public GradientDescentOptimizerBasev4Template<TInternalComputationValueType>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(DIFFPREPGradientDescentOptimizerv4);

  /** Standard class type aliases. */
  using Self = DIFFPREPGradientDescentOptimizerv4;
  using Superclass = GradientDescentOptimizerBasev4Template<TInternalComputationValueType>;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;

  /** Run-time type information (and related methods). */
  itkTypeMacro(DIFFPREPGradientDescentOptimizerv4, Superclass);

  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro(Self);


  /** It should be possible to derive the internal computation type from the class object. */
  using InternalComputationValueType = TInternalComputationValueType;

  /** Derivative type */
  using DerivativeType = typename Superclass::DerivativeType;

  /** Metric type over which this class is templated */
  using MeasureType = typename Superclass::MeasureType;
  using IndexRangeType = typename Superclass::IndexRangeType;
  using ScalesType = typename Superclass::ScalesType;
  using ParametersType = typename Superclass::ParametersType;


  static constexpr itk::StopConditionObjectToObjectOptimizerEnum MAXIMUM_NUMBER_OF_ITERATIONS =
    itk::StopConditionObjectToObjectOptimizerEnum::MAXIMUM_NUMBER_OF_ITERATIONS;
  static constexpr itk::StopConditionObjectToObjectOptimizerEnum COSTFUNCTION_ERROR =
    itk::StopConditionObjectToObjectOptimizerEnum::COSTFUNCTION_ERROR;
  static constexpr itk::StopConditionObjectToObjectOptimizerEnum UPDATE_PARAMETERS_ERROR =
    itk::StopConditionObjectToObjectOptimizerEnum::UPDATE_PARAMETERS_ERROR;
  static constexpr itk::StopConditionObjectToObjectOptimizerEnum STEP_TOO_SMALL =
    itk::StopConditionObjectToObjectOptimizerEnum::STEP_TOO_SMALL;
  static constexpr itk::StopConditionObjectToObjectOptimizerEnum CONVERGENCE_CHECKER_PASSED =
    itk::StopConditionObjectToObjectOptimizerEnum::CONVERGENCE_CHECKER_PASSED;
  static constexpr itk::StopConditionObjectToObjectOptimizerEnum GRADIENT_MAGNITUDE_TOLEARANCE =
    itk::StopConditionObjectToObjectOptimizerEnum::GRADIENT_MAGNITUDE_TOLEARANCE;
  static constexpr itk::StopConditionObjectToObjectOptimizerEnum OTHER_ERROR =
    itk::StopConditionObjectToObjectOptimizerEnum::OTHER_ERROR;

  static constexpr unsigned int Nparams= itk::OkanQuadraticTransform<double>::NQUADPARAMS;


  /** Set/Get the learning rate to apply. It is overridden by
   *  automatic learning rate estimation if enabled. See main documentation.
   */
  itkSetMacro(LearningRate, TInternalComputationValueType);
  itkGetConstReferenceMacro(LearningRate, TInternalComputationValueType);



  /** Minimum convergence value for convergence checking.
   *  The convergence checker calculates convergence value by fitting to
   *  a window of the energy profile. When the convergence value reaches
   *  a small value, it would be treated as converged.
   *
   *  The default m_MinimumConvergenceValue is set to 1e-8 to pass all
   *  tests. It is suggested to use 1e-6 for less stringent convergence
   *  checking.
   */
  itkSetMacro(MinimumConvergenceValue, TInternalComputationValueType);

  /** Window size for the convergence checker.
   *  The convergence checker calculates convergence value by fitting to
   *  a window of the energy (metric value) profile.
   *
   *  The default m_ConvergenceWindowSize is set to 50 to pass all
   *  tests. It is suggested to use 10 for less stringent convergence
   *  checking.
   */
  itkSetMacro(ConvergenceWindowSize, SizeValueType);

  /** Get current convergence value.
   *  WindowConvergenceMonitoringFunction always returns output convergence
   *  value in 'TInternalComputationValueType' precision. */
  itkGetConstReferenceMacro( ConvergenceValue, TInternalComputationValueType);

  itkSetMacro(Epsilon, double);
  itkGetConstReferenceMacro(Epsilon, double);
  itkSetMacro(NumberHalves, int);
  itkGetConstReferenceMacro(NumberHalves, int);
  itkGetConstMacro(CurrentIteration, SizeValueType);


  /** Start and run the optimization. */
  void StartOptimization( bool doOnlyInitialization = false ) override;

  /** Stop the optimization. */
  void StopOptimization() override;

  /** Resume the optimization. */
  void ResumeOptimization() override;


  void SetOptimizationFlags(ParametersType fl){optimization_flags=fl;};
  void SetGradScales(ParametersType sc)
  {
      grad_params=sc;
      orig_grad_params=sc;
  }
  ParametersType GetGradScales() const
  {
      return grad_params;
  }
  void SetBrkEps(double brk){brk_eps=brk;};


  #ifdef USECUDA
      void SetFixedCudaImage(CUDAIMAGE::Pointer nimg)
      {
          fixed_img_cuda=nimg;
          entropy_f= ComputeEntropy(fixed_img_cuda, Nbins, lim_arr[0],lim_arr[1]);
      }
      void SetMovingCudaImage(CUDAIMAGE::Pointer nimg){moving_img_cuda=nimg;}
      void SetLimits(std::vector<float> aa){lim_arr=aa;}
      ParametersType GetParameters(){return this->m_Metric->GetParameters();}
      void SetNBins(int nb){Nbins=nb;}
      
  #endif


private:
  double GetGrad(std::vector<int>,DerivativeType &CurrGrad);
  std::vector< std::pair<double,double> > BracketGrad(double brk_const,DerivativeType &CurrGrad);
  double GoldenSearch(double cst,std::vector< std::pair<double,double> >&x_f_pairs, MeasureType & new_cost,DerivativeType &CurrGrad);  
  double ComputeMetric(ParametersType new_params);


protected:

  /** Advance one step following the gradient direction.
   * Includes transform update. */
  virtual void AdvanceOneStep();

  /** Modify the gradient by scales and weights over a given index range. */
  void ModifyGradientByScalesOverSubRange( const IndexRangeType& subrange ) override
  {}

  /** Modify the gradient by learning rate over a given index range. */
  void ModifyGradientByLearningRateOverSubRange( const IndexRangeType& subrange ) override
  {}



  /** Default constructor */
  DIFFPREPGradientDescentOptimizerv4();

  /** Destructor */
  ~DIFFPREPGradientDescentOptimizerv4() override = default;

  void PrintSelf( std::ostream & os, Indent indent ) const override;


  TInternalComputationValueType m_LearningRate;
  TInternalComputationValueType m_MinimumConvergenceValue;
  TInternalComputationValueType m_ConvergenceValue;

private:
  double m_Epsilon;
  double brk_eps;
  SizeValueType      m_NumberOfIterations;
  int m_NumberHalves;
  

  SizeValueType      m_CurrentIteration;
  
  
  ParametersType    optimization_flags;
  ParametersType    grad_params;
  ParametersType    orig_grad_params;
  double m_BracketParams[8];

  
  #ifdef USECUDA
      CUDAIMAGE::Pointer fixed_img_cuda{nullptr};
      CUDAIMAGE::Pointer moving_img_cuda{nullptr};
      std::vector<float> lim_arr;
      int Nbins{100};

      double entropy_f{0};
  #endif


};



} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDIFFPREPGradientDescentOptimizerv4.cxx"
#endif

#endif
