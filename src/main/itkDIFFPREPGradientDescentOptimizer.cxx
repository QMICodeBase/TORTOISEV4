
#ifndef _itkDIFFPREPGradientDescentOptimizer_hxx
#define _itkDIFFPREPGradientDescentOptimizer_hxx

#include "itkDIFFPREPGradientDescentOptimizer.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include <utility>

#include "itkResampleImageFilter.h"
#include "itkOkanQuadraticTransform.h"

namespace itk
{

DIFFPREPGradientDescentOptimizer
::DIFFPREPGradientDescentOptimizer()
{
  itkDebugMacro("Constructor");

  m_Epsilon=0.00001;
  m_CurrentIteration   =   0;
  m_Value = 0;  
  m_CurrentCost=0;
  

  m_Gradient.Fill(0.0f);
  
  
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
  image_Metric=NULL;

#ifdef USE_VTK
    newGUI=NULL;
    display=0;
#endif
    
  
}


void DIFFPREPGradientDescentOptimizer::StartOptimization(void)
{
  itkDebugMacro("StartOptimization");

  m_CurrentIteration          = 0;

  const unsigned int spaceDimension = image_Metric->GetMovingTransform()->GetNumberOfParameters();

  m_Gradient = DerivativeType(spaceDimension);
  m_Gradient.Fill(0.0f);

  grad_params=orig_grad_params;

  m_CurrentCost=  ComputeMetric(image_Metric->GetParameters() );
  

  m_BracketParams[0]=0.1;
  m_BracketParams[1]=0.0001;
  m_BracketParams[2]=0.0001;
  m_BracketParams[3]=0.000001;
  m_BracketParams[4]=0.000001;
  m_BracketParams[5]=0.000000001;
  m_BracketParams[6]=0.000000001;
  m_BracketParams[7]=0.1;

  this->ResumeOptimization();
}





double DIFFPREPGradientDescentOptimizer::ComputeMetric(ParametersType new_params)
{
   MetricType::MovingTransformType::Pointer aaa= image_Metric->GetMovingTransform();


   ParametersType orig_params= aaa->GetParameters();

   aaa->SetParameters(new_params);


#ifdef USE_VTK
   if(this->display)
   {
       newGUI->SetFixedImage(image_Metric->GetFixedImage());

       typedef itk::ResampleImageFilter<MetricType::FixedImageType, MetricType::FixedImageType> ResampleImageFilterType;
       ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
       resampleFilter->SetOutputParametersFromImage(image_Metric->GetFixedImage());
       resampleFilter->SetInput(image_Metric->GetMovingImage());
       resampleFilter->SetTransform(aaa);
       resampleFilter->Update();
       ImageType3DITK::Pointer transformed_img= resampleFilter->GetOutput();
       newGUI->SetMovingImage(transformed_img);
     //  newGUI->Refresh();

   }
#endif



   double val= image_Metric->GetValue();
   aaa->SetParameters(orig_params);
   return val;
}



double DIFFPREPGradientDescentOptimizer::GetGrad(std::vector<int> ids)
{
    m_Gradient.Fill(0.0f);

       ParametersType or_params=image_Metric->GetParameters();


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
               m_Gradient[curr_param_id]=(fp-fm)/(2*grad_params[curr_param_id]);
           }
       }

       double nrm = m_Gradient.magnitude();
       if(nrm >0)
       {
           for(int v=0;v<ids.size();v++)
           {
               int id= ids[v];
               m_Gradient[id]=m_Gradient[id]/nrm;
           }
       }

       //this->SetCurrentPosition(or_params);
       return nrm;
}



std::vector< std::pair<double,double> > DIFFPREPGradientDescentOptimizer::BracketGrad(double brk_const)
{
    //ParametersType or_params=this->GetCurrentPosition();
    ParametersType or_params=image_Metric->GetParameters();

    int MAX_IT=50;
    double m_ERR_MARG=0.00001;

    double f_ini= this->m_CurrentCost;
    double x_ini=0;

    double f_min = f_ini;
    double x_min = x_ini;

    double f_last;
    double x_last;

    bool bail = 0;
    int counter = 1;

    while(!bail)
    {
        double konst = counter*counter*brk_const;
        ParametersType temp_params= or_params;
        temp_params=temp_params- konst*m_Gradient;

        f_last= ComputeMetric(temp_params);
        x_last = konst;

        if(f_last <f_min)
        {
            f_min = f_last;
            x_min = x_last;
        }
        else
        {
            if(  (f_last > f_min+m_ERR_MARG)  || (counter >MAX_IT))
                bail=true;
        }
        counter++;
    }



    //this->SetCurrentPosition(or_params);

    std::vector< std::pair<double,double> > x_f_pairs;
    x_f_pairs.resize(3);
    x_f_pairs[0]=std::make_pair(x_ini,f_ini);
    x_f_pairs[1]=std::make_pair(x_min,f_min);
    x_f_pairs[2]=std::make_pair(x_last,f_last);

    return x_f_pairs;

}






double DIFFPREPGradientDescentOptimizer::GoldenSearch(double cst,std::vector< std::pair<double,double> > &x_f_pairs, MeasureType & new_cost)
{   

    int MAX_IT=100;
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

    ParametersType temp_trans= image_Metric->GetParameters();
    temp_trans=temp_trans - x1*m_Gradient;
    MeasureType fx1= ComputeMetric(temp_trans);

    temp_trans= image_Metric->GetParameters();
    temp_trans= temp_trans - x2*m_Gradient;
    MeasureType fx2= ComputeMetric(temp_trans);


    while( (fabs(x3-x0) > TOL) && (counter <MAX_IT))
    {
        if(fx2 <fx1)
        {
            x0=x1;
            x1=x2;
            x2= R*x2+C*x3;
                        
            temp_trans= image_Metric->GetParameters() - x2*m_Gradient;

            MeasureType xt= ComputeMetric(temp_trans);
            fx1=fx2;
            fx2=xt;
        }
        else
        {
            x3=x2;
            x2=x1;
            x1=R*x1+C*x0;
            temp_trans= image_Metric->GetParameters() - x1*m_Gradient;
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




void DIFFPREPGradientDescentOptimizer::ResumeOptimization(void)
{
    itkDebugMacro("ResumeOptimization");
    m_Stop = false;


    const unsigned int spaceDimension =image_Metric->GetMovingTransform()->GetParameters().size();

    this->InvokeEvent( StartEvent() );

    std::vector<int> mode_ids[8];
    mode_ids[0].push_back(0);
    mode_ids[0].push_back(1);
    mode_ids[0].push_back(2);
    mode_ids[1].push_back(3);
    mode_ids[1].push_back(4);
    mode_ids[1].push_back(5);
    mode_ids[2].push_back(6);
    mode_ids[2].push_back(7);
    mode_ids[2].push_back(8);
    mode_ids[3].push_back(9);
    mode_ids[3].push_back(10);
    mode_ids[3].push_back(11);
    mode_ids[4].push_back(12);
    mode_ids[4].push_back(13);
    mode_ids[5].push_back(14);
    mode_ids[6].push_back(15);
    mode_ids[6].push_back(16);
    mode_ids[6].push_back(17);
    mode_ids[6].push_back(18);
    mode_ids[6].push_back(19);
    mode_ids[6].push_back(20);
    mode_ids[7].push_back(21);
    mode_ids[7].push_back(22);
    mode_ids[7].push_back(23);


    ParametersType orig_params= image_Metric->GetParameters();;
    ParametersType temp_trans=orig_params;
    for(int i=0;i<NQUADPARAMS;i++)
        temp_trans[i]= orig_params[i] -1E-10;

    m_CurrentCost= ComputeMetric(orig_params);
    auto nn= ComputeMetric(temp_trans);





    MeasureType last_cost= m_CurrentCost;
    int curr_halve=0;

    while ( !m_Stop )
    {
      try
      {
          for(int mode=0;mode<8;mode++)
          {
              double nrm =this->GetGrad(mode_ids[mode]);

              if(nrm >0)
              {
                  std::vector< std::pair<double,double> > x_f_pairs= this->BracketGrad(m_BracketParams[mode]);

                  if(std::abs(x_f_pairs[0].first-x_f_pairs[1].first)>0)
                  {
                      MeasureType new_cost;
                      double step_length= this->GoldenSearch(m_BracketParams[mode],x_f_pairs,new_cost);                                          

                      ParametersType new_parameters=image_Metric->GetParameters();
                      for(int i=0;i<NQUADPARAMS;i++)
                          new_parameters[i]=new_parameters[i]- step_length*m_Gradient[i];


                      MetricType::MovingTransformType::Pointer aaa= image_Metric->GetMovingTransform();
                      aaa->SetParameters(new_parameters);
                      image_Metric->SetParameters(new_parameters);
                      m_Value=new_cost;
                      m_CurrentCost=new_cost;
                  }
                  else
                  {
                      ParametersType orig_params= image_Metric->GetParameters();

                      ParametersType temp_change= orig_params;

                      for(int i=0;i<NQUADPARAMS;i++)
                          temp_change[i]= m_Gradient[i]*grad_params[i]*0.01;


                      for(int i=0;i<3;i++)
                      {

                          ParametersType temp_trans=orig_params;
                          for(int i=0;i<NQUADPARAMS;i++)
                              temp_trans[i]= orig_params[i] -temp_change[i];

                          MeasureType temp_cost= ComputeMetric(temp_trans);
                          if(temp_cost<m_CurrentCost)
                          {
                              m_Value=temp_cost;
                              m_CurrentCost=temp_cost;
                              image_Metric->SetParameters(temp_trans);
                              MetricType::MovingTransformType::Pointer aaa= image_Metric->GetMovingTransform();
                              aaa->SetParameters(temp_trans);
                              break;
                          }
                          else
                              temp_change=temp_change/2.;

                      }
                  }

              }

          }
          this->SetCurrentPosition(image_Metric->GetParameters());


          if( (last_cost -m_CurrentCost) > m_Epsilon)
          {
              m_CurrentIteration++;
              this->InvokeEvent(IterationEvent());
              last_cost= m_CurrentCost;
          }
          else
          {
              if(curr_halve<m_NumberHalves)
              {
                  curr_halve++;
                  for(int i=0;i<NQUADPARAMS;i++)
                      grad_params[i]/=1.7;
                  //m_CurrentIteration=0;
              }
              else
                  this->StopOptimization();
          }
          if(m_CurrentIteration > m_NumberOfIterations)
                  this->StopOptimization();
      }
      catch ( ExceptionObject & excp )
      {
          std::cerr<< "Cost function error after "
                             << m_CurrentIteration
                             << " iterations. "
                             << excp.GetDescription();
          this->StopOptimization();

          throw excp;
      }
      if ( m_Stop )
      {
          break;
      }
    }

  
}


void
DIFFPREPGradientDescentOptimizer::StopOptimization(void)
{
  itkDebugMacro("StopOptimization");

  m_Stop = true;
  this->InvokeEvent( EndEvent() );
}


void
DIFFPREPGradientDescentOptimizer::AdvanceOneStep(void)
{
  itkDebugMacro("AdvanceOneStep");

  return;
}

void
DIFFPREPGradientDescentOptimizer::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "NumberOfIterations: "
     << m_NumberOfIterations << std::endl;
  os << indent << "CurrentIteration: "
     << m_CurrentIteration   << std::endl;
  os << indent << "Value: "
     << m_Value << std::endl;


  os << indent << "Gradient: "
     << m_Gradient << std::endl;
}
} // end namespace itk

#endif

