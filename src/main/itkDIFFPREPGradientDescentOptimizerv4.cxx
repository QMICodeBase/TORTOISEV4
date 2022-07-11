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
#ifndef itkDIFFPREPGradientDescentOptimizerv4_hxx
#define itkDIFFPREPGradientDescentOptimizerv4_hxx

#include "itkDIFFPREPGradientDescentOptimizerv4.h"
#include "itkOkanQuadraticTransform.h"

#ifdef USECUDA
    #include "../cuda_src/quadratic_transform_image.h"
    #include "../cuda_src/compute_mi_cuda.h"
#endif

#define G_R 0.61803399
#define G_C (1-G_R)

namespace itk
{

template<typename TInternalComputationValueType>
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::DIFFPREPGradientDescentOptimizerv4() :
  m_LearningRate( NumericTraits<TInternalComputationValueType>::OneValue() ),
  m_MinimumConvergenceValue( 1e-8 ),
  m_ConvergenceValue( NumericTraits<TInternalComputationValueType>::max() )
{
     m_Epsilon=0.00001;
     m_BracketParams[0]=0.1;
     m_BracketParams[1]=0.0001;
     m_BracketParams[2]=0.0001;
     m_BracketParams[3]=0.000001;
     m_BracketParams[4]=0.000001;
     m_BracketParams[5]=0.000000001;
     m_BracketParams[6]=0.000000001;
     m_BracketParams[7]=0.1;

     grad_params.set_size(NQUADPARAMS);

     grad_params[0]=2.5;    grad_params[1]=2.5;  grad_params[2]=2.5;
     grad_params[3]=0.04;   grad_params[4]=0.04; grad_params[5]=0.04;
     grad_params[6]=0.02;   grad_params[7]=0.02; grad_params[8]=0.02;
     grad_params[9]=0.002;   grad_params[10]=0.002; grad_params[11]=0.002;
     grad_params[12]=0.0007;  grad_params[13]=0.0007;
     grad_params[14]=0.0001;
     grad_params[15]=0.00002; grad_params[16]=0.00002; grad_params[17]=0.00002; grad_params[18]=0.00002; grad_params[19]=0.00002; grad_params[20]=0.00002;
     grad_params[21]=grad_params[0]/2.;   grad_params[22]=grad_params[1]/2.;   grad_params[23]=grad_params[2]/2.;

     orig_grad_params=grad_params;
     m_NumberOfIterations=50;
     m_NumberHalves=5;

}

template<typename TInternalComputationValueType>
void
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::StartOptimization( bool doOnlyInitialization )
{
  // Must call the superclass version for basic validation and setup.
  Superclass::StartOptimization( doOnlyInitialization );

  int Nparams= this->m_Metric->GetNumberOfParameters();
  if(this->m_Gradient.size()!=Nparams)
  {
      this->m_Gradient = DerivativeType(Nparams);
      this->m_Gradient.Fill(0.0f);
  }

  grad_params=orig_grad_params;


  #ifdef USECUDA
      if(this->moving_img_cuda)
      {
          ParametersType or_params=this->m_Metric->GetParameters();
          this->m_CurrentMetricValue= ComputeMetric(or_params);
      }
      else
      {
          this->m_CurrentMetricValue= this->m_Metric->GetValue();   //for some reason it does not update unless you compute it twice
          this->m_CurrentMetricValue= this->m_Metric->GetValue();   //maybe a bug in ITK?
      }
  #else
      this->m_CurrentMetricValue= this->m_Metric->GetValue();   //for some reason it does not update unless you compute it twice
      this->m_CurrentMetricValue= this->m_Metric->GetValue();   //maybe a bug in ITK?

  #endif


  this->m_CurrentIteration = 0;
  this->m_ConvergenceValue = NumericTraits<TInternalComputationValueType>::max();

  if( ! doOnlyInitialization )
    {
    this->ResumeOptimization();
    }
}

template<typename TInternalComputationValueType>
void
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::StopOptimization()
{
  Superclass::StopOptimization();
}


template<typename TInternalComputationValueType>
double
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::GetGrad(std::vector<int> ids, DerivativeType &CurrGrad)
{
    CurrGrad.Fill(0.0);
    ParametersType or_params=this->m_Metric->GetParameters();

    for(int v=0;v<ids.size();v++)
    {
        int curr_param_id= ids[v];
        if(optimization_flags[curr_param_id])
        {
            ParametersType temp_params=or_params;

            temp_params[curr_param_id]+=grad_params[curr_param_id];
            MeasureType fp= ComputeMetric(temp_params);
            temp_params[curr_param_id]-=2*grad_params[curr_param_id];
            MeasureType fm= ComputeMetric(temp_params);
            CurrGrad[curr_param_id]=(fp-fm)/(2*grad_params[curr_param_id]);

       }
    }

    double nrm = CurrGrad.magnitude();
    if(nrm >0)
    {
       for(int v=0;v<ids.size();v++)
       {
           int id= ids[v];
           CurrGrad[id]=CurrGrad[id]/nrm;
       }
    }


   return nrm;
}


template<typename TInternalComputationValueType>
std::vector< std::pair<double,double> >
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::DIFFPREPGradientDescentOptimizerv4::BracketGrad(double brk_const, DerivativeType &CurrGrad)
{   
    ParametersType or_params=this->m_Metric->GetParameters();

    int MAX_IT=20;
    double m_ERR_MARG=0.00001;

    double f_ini= this->m_CurrentMetricValue;
    double x_ini=0;

    double f_min = f_ini;
    double x_min = x_ini;

    double f_last;
    double x_last;

    bool bail = 0;
    double counter = 1;
    int iter=0;

    while(!bail)
    {
        //double konst = counter*counter*brk_const;
        double konst = counter*brk_const;
        ParametersType  temp_params=or_params;
        temp_params -= konst*CurrGrad;

        f_last= ComputeMetric(temp_params);
        x_last = konst;

        if(f_last <f_min)
        {
            f_min = f_last;
            x_min = x_last;
        }
        else
        {
            if(  (f_last > f_min+m_ERR_MARG)  || (iter >MAX_IT))
                bail=true;
        }
        //counter++;
        counter*=1.7;
        iter++;
    }


    std::vector< std::pair<double,double> > x_f_pairs;
    x_f_pairs.resize(3);
    x_f_pairs[0]=std::make_pair(x_ini,f_ini);
    x_f_pairs[1]=std::make_pair(x_min,f_min);
    x_f_pairs[2]=std::make_pair(x_last,f_last);

    return x_f_pairs;

}




template<typename TInternalComputationValueType>
double
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::GoldenSearch(double cst,std::vector< std::pair<double,double> > &x_f_pairs, MeasureType & new_cost, DerivativeType &CurrGrad)
{

    int MAX_IT=50;
    int counter=0;
    double TOL=cst;
    double R=  0.61803399;
    double C = 1.0 - R;

    double ax= x_f_pairs[0].first;
    double bx= x_f_pairs[1].first;
    double cx= x_f_pairs[2].first;

    double x0=ax;
    double x3=cx;
    double x1,x2;

    if (fabs(cx-bx) > fabs(bx-ax))
    {
        x1=bx;
        x2= bx+C*(cx-bx);

    }
    else
    {
        x2=bx;
        x1=bx- C*(bx-ax);
    }

    ParametersType temp_trans= this->m_Metric->GetParameters();
    temp_trans=temp_trans - x1*CurrGrad;
    MeasureType fx1= ComputeMetric(temp_trans);

    temp_trans= this->m_Metric->GetParameters();
    temp_trans= temp_trans - x2*CurrGrad;
    MeasureType fx2= ComputeMetric(temp_trans);

    while( (fabs(x3-x0) > TOL) && (counter <MAX_IT))
    {
        if(fx2 <fx1)
        {
            x0=x1;
            x1=x2;
            x2= R*x2+C*x3;

            temp_trans= this->m_Metric->GetParameters() - x2*CurrGrad;
            MeasureType xt= ComputeMetric(temp_trans);
            fx1=fx2;
            fx2=xt;
        }
        else
        {
            x3=x2;
            x2=x1;
            x1=R*x1+C*x0;
            temp_trans= this->m_Metric->GetParameters() - x1*CurrGrad;
            MeasureType xt= ComputeMetric(temp_trans);
            fx2=fx1;
            fx1=xt;
        }
        counter++;
    }


    if(fx1<fx2)
    {
        new_cost=fx1;
        return x1;
    }
    else
    {
        new_cost=fx2;
        return x2;
    }
}




template<typename TInternalComputationValueType>
double
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::ComputeMetric(ParametersType new_params)
{    
    double val;
    #ifdef USECUDA
        if(this->moving_img_cuda)
        {
            TransformType::Pointer tp= TransformType::New();
            tp->SetParameters(new_params);
            CUDAIMAGE::Pointer trans_moving_img = QuadraticTransformImageC(this->moving_img_cuda, tp,fixed_img_cuda);

           // writeImageD<ImageType3D>(fixed_img_cuda->CudaImageToITKImage(),"/qmi13_raid/okan/ABCD_Don_100_subjects/dMRIv3/data/DTIPROC_G010_INV18YX7994_2year_20181117.124339_1/tmp_DTI_corr_regT1/proc/aaaf.nii");
          //  writeImageD<ImageType3D>(trans_moving_img->CudaImageToITKImage(),"/qmi13_raid/okan/ABCD_Don_100_subjects/dMRIv3/data/DTIPROC_G010_INV18YX7994_2year_20181117.124339_1/tmp_DTI_corr_regT1/proc/aaam.nii");

            float entropy_m,entropy_j;
            ComputeJointEntropy(fixed_img_cuda,lim_arr[0],lim_arr[1],trans_moving_img,lim_arr[2],lim_arr[3],Nbins,entropy_j,entropy_m );

            val = -(this->entropy_f + entropy_m)/entropy_j;
        }
        else
        {
            ParametersType orig_params= this->m_Metric->GetParameters();
            this->m_Metric->SetParameters(new_params);
            val= this->m_Metric->GetValue();
            this->m_Metric->SetParameters(orig_params);            
        }
    #else
        ParametersType orig_params= this->m_Metric->GetParameters();
        this->m_Metric->SetParameters(new_params);
        val= this->m_Metric->GetValue();
        this->m_Metric->SetParameters(orig_params);        
    #endif
    return val;
}



template<typename TInternalComputationValueType>
void
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::ResumeOptimization()
{
  this->m_StopConditionDescription.str("");
  this->m_StopConditionDescription << this->GetNameOfClass() << ": ";
  this->InvokeEvent( StartEvent() );

  std::vector<std::vector<int> > mode_ids={{0,1,2},{3,4,5},{6,7,8},{9,10,11},{12,13},{14},{15,16,17,18,19,20},{21,22,23} };


  MeasureType last_cost= NumericTraits<MeasureType>::max();  
  int curr_halve=0;

  this->m_Stop = false;
  while( ! this->m_Stop )
  {
    // Do not run the loop if the maximum number of iterations is reached or its value is zero.
    if ( this->m_CurrentIteration >= this->m_NumberOfIterations )
      {
      this->m_StopConditionDescription << "Maximum number of iterations (" << this->m_NumberOfIterations << ") exceeded.";
      this->m_StopCondition = Superclass::MAXIMUM_NUMBER_OF_ITERATIONS;
      this->StopOptimization();
      break;
      }

    
    // Compute metric value/derivative.
    try
    {
        for(int mode=0;mode<8;mode++)
        {
            this->m_Gradient.Fill(0);
            DerivativeType m_CurrGrad= this->m_Gradient;
            
            double nrm =this->GetGrad(mode_ids[mode],m_CurrGrad);            
            if(nrm >0)
            {
                std::vector< std::pair<double,double> > x_f_pairs= this->BracketGrad(m_BracketParams[mode],m_CurrGrad);

                if(std::abs(x_f_pairs[0].first-x_f_pairs[1].first)>0)
                {
                    MeasureType new_cost;
                    double step_length= this->GoldenSearch(m_BracketParams[mode],x_f_pairs,new_cost,m_CurrGrad);                                          
                    
                    for(int i=0;i<mode_ids[mode].size();i++)
                        this->m_Gradient[mode_ids[mode][i]]= - step_length*m_CurrGrad[mode_ids[mode][i]];
                }
                else
                {                    
                    ParametersType orig_params= this->m_Metric->GetParameters();
                    ParametersType temp_change= orig_params;

                    for(int i=0;i<NQUADPARAMS;i++)
                        temp_change[i]= m_CurrGrad[i]*grad_params[i]*0.01;

                    for(int i=0;i<3;i++)
                    {
                        ParametersType temp_trans=orig_params;
                        for(int i=0;i<NQUADPARAMS;i++)
                            temp_trans[i]= orig_params[i] -temp_change[i];

                        MeasureType temp_cost= ComputeMetric(temp_trans);
                        if(temp_cost<this->m_CurrentMetricValue)
                        {
                            for(int i=0;i<mode_ids[mode].size();i++)
                                this->m_Gradient[mode_ids[mode][i]]= -temp_change[mode_ids[mode][i]];
                            break;
                        }
                        else
                            temp_change=temp_change/2.;
                    }
                }


                // Advance one step along the gradient.
                // This will modify the gradient and update the transform.
                if(this->m_Gradient.magnitude()>0)
                {
                    this->AdvanceOneStep();

                    #ifdef USECUDA
                        if(this->moving_img_cuda)
                        {
                            ParametersType or_params=this->m_Metric->GetParameters();
                            this->m_CurrentMetricValue= ComputeMetric(or_params);
                        }
                        else
                            this->m_CurrentMetricValue= this->m_Metric->GetValue();
                    #else
                        this->m_CurrentMetricValue= this->m_Metric->GetValue();
                    #endif
                }

            }
        }
        if( (last_cost -this->m_CurrentMetricValue) > m_Epsilon)
        {
            m_CurrentIteration++;
            this->InvokeEvent(IterationEvent());
            last_cost= this->m_CurrentMetricValue;
        }
        else
        {
            if(curr_halve<m_NumberHalves)
            {
                curr_halve++;
                for(int i=0;i<NQUADPARAMS;i++)
                    grad_params[i]/=1.7;
                m_CurrentIteration=0;
            }
            else
                this->StopOptimization();
        }
    }
    catch ( ExceptionObject & err )
    {
      this->m_StopCondition = Superclass::COSTFUNCTION_ERROR;
      this->m_StopConditionDescription << "Metric error during optimization";
      this->StopOptimization();

      // Pass exception to caller
      throw err;
    }

    // Check if optimization has been stopped externally.
    // (Presumably this could happen from a multi-threaded client app?)
    if ( this->m_Stop )
      {
      this->m_StopConditionDescription << "StopOptimization() called";
      break;
      }
    } //while (!m_Stop)
}

template<typename TInternalComputationValueType>
void
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::AdvanceOneStep()
{
  itkDebugMacro("AdvanceOneStep");

  // Begin threaded gradient modification.
  // Scale by gradient scales, then estimate the learning
  // rate if options are set to (using the scaled gradient),
  // then modify by learning rate. The m_Gradient variable
  // is modified in-place.

  //  this->ModifyGradientByScales();
//  this->EstimateLearningRate();
//  this->ModifyGradientByLearningRate();

  try
    {
    // Pass gradient to transform and let it do its own updating
    this->m_Metric->UpdateTransformParameters( this->m_Gradient );
    }
  catch ( ExceptionObject & err )
    {
    this->m_StopCondition = Superclass::UPDATE_PARAMETERS_ERROR;
    this->m_StopConditionDescription << "UpdateTransformParameters error";
    this->StopOptimization();

      // Pass exception to caller
    throw err;
    }

  this->InvokeEvent( IterationEvent() );
}

template<typename TInternalComputationValueType>
void
DIFFPREPGradientDescentOptimizerv4<TInternalComputationValueType>
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "LearningRate: "
    << static_cast< typename NumericTraits< TInternalComputationValueType >::PrintType >( this->m_LearningRate )
    << std::endl;
  os << indent << "MinimumConvergenceValue: " << this->m_MinimumConvergenceValue << std::endl;
  os << indent << "ConvergenceValue: "
    << static_cast< typename NumericTraits< TInternalComputationValueType >::PrintType >( this->m_ConvergenceValue )
    << std::endl;
}
}//namespace itk

#endif
