

#ifndef __itkDIFFPREPGradientDescentOptimizer_h
#define __itkDIFFPREPGradientDescentOptimizer_h

#include "itkIntTypes.h"
#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkImageToImageMetricv4.h"

#ifdef USE_VTK
   #include "VTKGUI.h"
#endif


#define G_R 0.61803399
#define G_C (1-G_R)

namespace itk
{

class ITK_EXPORT DIFFPREPGradientDescentOptimizer:   public SingleValuedNonLinearOptimizer
{
public:

  typedef DIFFPREPGradientDescentOptimizer Self;
  typedef SingleValuedNonLinearOptimizer          Superclass;
  typedef SmartPointer< Self >                    Pointer;
  typedef SmartPointer< const Self >              ConstPointer;
  
  


  itkNewMacro(Self);


  itkTypeMacro(DIFFPREPGradientDescentOptimizer, SingleValuedNonLinearOptimizer);
  
  typedef Superclass::CostFunctionType CostFunctionType;
  typedef CostFunctionType::Pointer    CostFunctionPointer;
  typedef ObjectToObjectMetricBaseTemplate<double>                  BaseMetricType;
  typedef ImageToImageMetricv4<itk::Image<float,3>, itk::Image<float,3>,  itk::Image<float,3>, double>  MetricType;



  

  itkSetMacro(Epsilon, double);
  itkGetConstReferenceMacro(Epsilon, double);
  itkGetConstReferenceMacro(Value, MeasureType);
  itkGetConstReferenceMacro(Gradient, DerivativeType);
  itkSetMacro(NumberOfIterations, SizeValueType);
  itkGetConstReferenceMacro(NumberOfIterations, SizeValueType);
  itkSetMacro(NumberHalves, int);
  itkGetConstReferenceMacro(NumberHalves, int);
  itkGetConstMacro(CurrentIteration, unsigned int);
  


  //const  ParametersType & GetGradientParameters(){return m_Gradient_bracket_sizes;};



 // vnl_matrix<double> jh3d(MetricType::MovingImageType::Pointer fixed_img, MetricType::MovingImageType::Pointer moving_img, const int nbins,   MetricType::FixedImageMaskType::ConstPointer mask);
 // double            nmi3d(MetricType::MovingImageType::Pointer fixed_img, MetricType::MovingImageType::Pointer moving_img, const int nbins,   MetricType::FixedImageMaskType::ConstPointer mask);

  void    StartOptimization(void);


  void    ResumeOptimization(void);


  void    StopOptimization(void);


  void SetOptimizationFlags(ParametersType fl){optimization_flags=fl;};
  void SetGradScales(ParametersType sc)
  {
      grad_params=sc;
      orig_grad_params=sc;
  };

  ParametersType GetGradScales() const
  {
      return grad_params;
  }


  //void SetHalves(int hlf){m_NumberHalves=hlf;};
  //void SetEpsilon(double eps){m_Epsilon=eps;};
  void SetBrkEps(double brk){brk_eps=brk;};


  void SetMetric(BaseMetricType::Pointer m)
  {
      image_Metric = dynamic_cast<MetricType *>( m.GetPointer() );
  };

#ifdef USE_VTK
    VTKGUI *newGUI;
    bool display;
#endif


protected:
  DIFFPREPGradientDescentOptimizer();
  virtual ~DIFFPREPGradientDescentOptimizer() {}
  void PrintSelf(std::ostream & os, Indent indent) const;


  virtual void AdvanceOneStep(void);


  virtual void StepAlongGradient(  double,const DerivativeType &)
  {
    ExceptionObject ex;

    ex.SetLocation(__FILE__);
    ex.SetDescription("This method MUST be overloaded in derived classes");
    throw ex;
  }

private:
  DIFFPREPGradientDescentOptimizer(const Self &); //purposely not
                                                         // implemented
  void operator=(const Self &);                          //purposely not

  // implemented
  
  


  
  double GetGrad(std::vector<int>);
  std::vector< std::pair<double,double> > BracketGrad(double brk_const);
  double GoldenSearch(double cst,std::vector< std::pair<double,double> >&x_f_pairs, MeasureType & new_cost);
  double ComputeMetric(ParametersType new_params);

protected:
  DerivativeType m_Gradient;   
  MeasureType m_CurrentCost;




  double m_Epsilon;
  double brk_eps;


  SizeValueType      m_NumberOfIterations;
  int m_NumberHalves;
  

  bool               m_Stop;
  MeasureType        m_Value;
  SizeValueType      m_CurrentIteration;
  
  
  ParametersType    optimization_flags;
  ParametersType    grad_params;
  ParametersType    orig_grad_params;

  double m_BracketParams[8];
  MetricType::Pointer image_Metric;





};
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDIFFPREPGradientDescentOptimizer.cxx"
#endif
#endif
