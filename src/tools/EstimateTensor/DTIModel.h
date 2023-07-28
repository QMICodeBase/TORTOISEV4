#ifndef _DTIModel_H
#define _DTIModel_H


#include "DiffusionModel.h"


struct vars_struct
{
  vnl_matrix<double> *Bmat;
  vnl_vector<double>  *signal;
  bool useWeights;
  vnl_vector<double> *weights;
};


int myNLLS_with_derivs(int m, int n, double *p, double *deviates,   double **derivs, void *vars);

class DTIModel: public DiffusionModel<DTImageType>
{
public:    
    using Superclass=DiffusionModel<DTImageType>;
    using OutputImageType= Superclass::OutputImageType;

public:
    DTIModel(){};
    ~DTIModel(){};


    void SetFittingMode(std::string s){fitting_mode=s;}


    void PerformFitting();
    ImageType3D::Pointer SynthesizeDWI(vnl_vector<double> bmat_vec);
    ImageType3D::Pointer getCSImg(){return CS_img;}
    ImageType3D::Pointer getVFImg(){return VF_img;}
    ImageType3D::Pointer getVFImg2(){return VF_img2;}
    DTImageType::Pointer getFlowImg(){return flow_tensor_img;}    

    ImageType3D::Pointer ComputeTRMap()
    {
        if(this->output_img==nullptr)
            return nullptr;

        ImageType3D::Pointer tr_img = ImageType3D::New();
        tr_img->SetRegions(this->output_img->GetLargestPossibleRegion());
        tr_img->Allocate();
        tr_img->SetSpacing(this->output_img->GetSpacing());
        tr_img->SetOrigin(this->output_img->GetOrigin());
        tr_img->SetDirection(this->output_img->GetDirection());
        tr_img->FillBuffer(0);

        itk::ImageRegionIteratorWithIndex<ImageType3D> it( tr_img,tr_img->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd(); ++it)
        {
            ImageType3D::IndexType ind3= it.GetIndex();
            OutputImageType::PixelType dt= this->output_img->GetPixel(ind3);
            float val = dt[0]+dt[3]+dt[5];
            it.Set(val);
        }
        return tr_img;
    }


private:
    void EstimateTensorWLLS();
    void EstimateTensorNLLS();
    void EstimateTensorN2();
    void EstimateTensorNT2();
    void EstimateTensorWLLSDiagonal();
    void EstimateTensorRESTORE();

    double check_condition_number(std::vector<int> &outlier_index, vnl_matrix<double> Bmatrix);
    std::vector<int> check_gradient_direction(std::vector<int> outlier_index_original,vnl_matrix<double> Bmatrix);
    float ComputeMedianB0MeanStDev(std::vector<int> b0_indices, float &median_signal_b0_std);
    DTImageType::PixelType  RobustFit(vnl_matrix<double> curr_design_matrix,vnl_vector<double> signal, double initial_A0_estimate,
                                      DTImageType::PixelType initial_dt_estimate1,  vnl_vector<double> sigstdev,
                                      std::vector<int> &b0_indices,double &A0_estimate,std::vector<int> &outlier_index,double &CS_val,float THR);

    vnl_matrix<double>  getCurrentBmatrix(ImageType3D::IndexType ind3,std::vector<int> curr_all_indices);

private:

    std::string fitting_mode;    

    ImageType3D::Pointer CS_img{nullptr};
    ImageType3D::Pointer VF_img{nullptr};
    ImageType3D::Pointer VF_img2{nullptr};
    DTImageType::Pointer flow_tensor_img{nullptr};

};



#endif
