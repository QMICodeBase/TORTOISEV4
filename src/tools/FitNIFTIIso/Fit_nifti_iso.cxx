/* fit_nitfit_iso_gw() IDL to C++
 * PURPOSE: To compute gradient non-uniformity from phantom measurements.
 */

#include "Fit_nifti_iso.h"
#include <algorithm>
#include <ctype.h>
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/read_bmatrix_file.h"
#include "vnl/algo/vnl_matrix_inverse.h"
#include "../external_src/cmpfit-1.3a/mpfit.h"
#include <boost/iostreams/device/mapped_file.hpp>
#include "omp.h"
#include <math.h>

/* Struct contains the paramerters to support function in fitting*/
struct vars_struct
{
  vnl_matrix<double> *Bmat;
  basisFuntions *Basis;
  GradCoef *E;
  double dd;
  double S0;
  vnl_vector<double> *Image1Darray;
  vnl_vector<double>*weights;
  int nmaskpts;
  mp_par *pars;
  int count = 0;


};


/* Main Pipeline Code */
FITNIFTIISO::FITNIFTIISO(int argc, char *argv[]){

    Fit_NIFTI_iso_parser *parser = new Fit_NIFTI_iso_parser(argc,argv);
    this->nii_name= parser->getInputImageName();


    std::cout << "read image: " << this->nii_name << std::endl;


    /* Get Parser Input Image, Mask Image, and Output File Name */
    std::string image_name = this->nii_name;
    /* Setup Scanner Type */
    this->scannerType = parser->getScannerType();
    if(parser->getScannerType()=="GE"){
        this->GE_type = true;
    }

    /* Setting init_flag, for single Fit    file = 0, for multiple files = 1 */
    if(parser->getExtraInputs()=="")
        this->init_flag=0;
    else
        this->init_flag=1;

    /* 4D image */
    this->set4Dimage(image_name);
    /* Get b0 image and PhysicCoordinate*/
    std::string bmtxt_name = image_name.substr(0,image_name.rfind(".nii"))+".bmtxt";
    this->Bmatrix = read_bmatrix_file(bmtxt_name);





    this->setb0Image(0);

    /*Set mask image, or create mask image */
    this->mask_filename = parser->getMaskImageName();
    this->setMaskImage(parser->getThreshold(), parser->getErodeFactor());

    /*Set Gradient Structure */
    this->R0 = parser->getDistance();
    this->computeCalibration = parser->IsEstimate_gradient_calibration();
    this->setGradientCalibrationStructure(parser->getGradCalName());
    this->setVectorOfFitting();
    this->setPermitFlagsforFitting();

    /* Read CoefFileName (TODO) */
    /*this->coef_filename = parser->getCoefFile();*/
    this->phantomD = parser->getPhantomDiffusitivity();
    this->grad_normalization = parser->getGradient_normalization();
    this->rdfit = parser->getRadiusFit();
    this->setNormalizingFlag();
    /*PROCESSING */
    this->process();
    /* we write the grad_cal txt file */
    debugArray(this->aa, "This aa");
    this->output_filename = parser->getOuputFilename();
    this->writeOutputs();

}

void FITNIFTIISO::writeOutputs(){
    if (this->output_filename != ""){
       this->write_gradcal(this->output_filename);
    }
    else{
       this->output_filename = this->nii_name.substr(0,nii_name.find(".nii"));
       if (this->computeCalibration)
           /* tp = 1*/
           this->output_filename+= std::string("_gradCpp_tp1.txt");
       else
           /*tp = 0 */
           this->output_filename+= std::string("_gradCpp_tp0.txt");

       this->write_gradcal(this->output_filename);

    }
    /* we write the grad_cal gc file */
    std::cout << this->output_filename << std::endl;
    std::string gcFN = this->output_filename.substr(0, this->output_filename.find(".txt")) + std::string(".gc");
    this->grads->write_ASB_format(this->gradCoefOutput,gcFN);

}

FITNIFTIISO::~FITNIFTIISO(){}

void FITNIFTIISO::process(){
    std::cout << "Processing......" << std::endl;
    if (this->init_flag == 0)
    {
        /* Perform init_iso_gw to get basis functions */
        basis = init_iso_gw(this->smat, this->maskImage, this->grads->get_struct());
    }

    ImageType3D::SizeType sizes= this->b0_image->GetLargestPossibleRegion().GetSize();

    /*If phantomD is not specified, compute dd near isocenter*/
    if (this->phantomD == 0)
    {
        double dd =0;
        vnl_vector<double> iso_phys(4,0);
        iso_phys[3] = 1;
        vnl_vector<double>iso_index = this->inv_smat * iso_phys;
        ImageType3D::IndexType iso_ind3;
        iso_ind3[0] =std::roundf(iso_index[0]);
        iso_ind3[1] =std::roundf(iso_index[1]);
        iso_ind3[2] =std::roundf(iso_index[2]);


        int rdfits[3]={0};

        /*if rdfit is not specfied, compute indices of magnet isocenter (get from smat) */
        if (this->rdfit == 0)        {
            std::cout << "Perform regression to compute indices of magnet isocenter, rdfit = 5x5x3" << std::endl;
           // this->phantomD = this->regression_at_point(iso_ind3);
            rdfits[0]=2;
            rdfits[1]=2;
            rdfits[2]=1;
        }
        else
        {
            std::cout << "Perform regression to compute diffusitivity inside rdfit" << std::endl;
            rdfits[0]=this->rdfit;
            rdfits[1]=this->rdfit;
            rdfits[2]=this->rdfit;
        }

        /*compute dd in roi sphere of radius rdfit about iscenter */

        int start_x_index= std::max(0,(int)iso_ind3[0]-rdfits[0]);
        int end_x_index= std::max((int)sizes[0]-1,(int)iso_ind3[0]+rdfits[0]);
        int start_y_index= std::max(0,(int)iso_ind3[1]-rdfits[1]);
        int end_y_index= std::max((int)sizes[1]-1,(int)iso_ind3[1]+rdfits[1]);
        int start_z_index= std::max(0,(int)iso_ind3[2]-rdfits[2]);
        int end_z_index= std::max((int)sizes[2]-1,(int)iso_ind3[2]+rdfits[2]);

        int Nvoxels=0;
        double DD=0;
        for(int kk=start_z_index;kk<=end_z_index;kk++)
        {
            ImageType3D::IndexType curr_ind3;
            curr_ind3[2]=kk;

            for(int jj=start_y_index;jj<=end_y_index;jj++)
            {
                curr_ind3[1]=jj;

                for(int ii=start_x_index;ii<=end_x_index;ii++)
                {
                    curr_ind3[0]=ii;

                    if(this->maskImage->GetPixel(curr_ind3)!=0)
                    {
                        int a =(kk- iso_ind3[2]);
                        int b =(jj- iso_ind3[1]);
                        int c =(ii- iso_ind3[0]);
                        double rr = std::sqrt(a*a+b*b+c*c);
                        if(this->rdfit)
                        {
                            if(rr <= this->rdfit)
                            {
                                DD+=this->regression_at_point(curr_ind3);
                                Nvoxels++;
                            }
                        }
                        else
                        {
                            DD+=this->regression_at_point(curr_ind3);
                            Nvoxels++;
                        }
                    }
                }
            }
        }
        this->phantomD=DD/Nvoxels;
    }
    /* tp and normalization are moved up to the constructor */
    this->aa(this->na-1) = this->phantomD;
    std::cout << "--- Estimated diffusitivity: "<< this->phantomD << std::endl;

    if (this->compute_normalization){
        this->normalize_image_nodif(this->nodif, this->nnodif);
    }
    else{
        /*Else only one non-diffusion weighted norm*/
        this->normalize_image_b0();
    }
    debugArray(this->fita, "this fita");
    this->PerformNonlinearityMisCalibrationFitting();
    /* Normalize gradients */
    this->normalize_gradient();
}

void FITNIFTIISO::normalize_gradient(){
    /* ;default is normalize to z-gradient */
    double initCoef =1;
    if (this->grad_normalization == "x")
        initCoef = this->gradCoefOutput.gradX_coef[0];
    if (this->grad_normalization == "y")
        initCoef = this->gradCoefOutput.gradY_coef[0];
    if (this->grad_normalization == "z")
        initCoef = this->gradCoefOutput.gradZ_coef[0];
    if (this->grad_normalization == "avg")
    {
        initCoef = (this->gradCoefOutput.gradX_coef[0] + this->gradCoefOutput.gradY_coef[0] +this->gradCoefOutput.gradZ_coef[0])/3.;
    }

    for (int kk =0; kk < this->gradCoefOutput.gradX_coef.size(); kk ++){
        this->gradCoefOutput.gradX_coef[kk] /=initCoef;
    }
    for (int kk =0; kk < this->gradCoefOutput.gradY_coef.size(); kk ++){
        this->gradCoefOutput.gradY_coef[kk] /=initCoef;
    }
    for (int kk =0; kk < this->gradCoefOutput.gradZ_coef.size(); kk ++){
        this->gradCoefOutput.gradZ_coef[kk] /=initCoef;
    }
}

void FITNIFTIISO::set4Dimage(std::string image_name){
    typedef itk::ImageFileReader<ImageType4D> ReaderType;
    ReaderType::Pointer reader=  ReaderType::New();
    reader->SetFileName(image_name);
    reader->Update();
    this->my_image = reader->GetOutput();
    ImageType4D::SizeType sizes= my_image->GetLargestPossibleRegion().GetSize();
    this->nVols= sizes[3];

}

void FITNIFTIISO::setb0Image(int volId = 0){
    ImageType3D::Pointer temp_b0_image=nullptr;
    temp_b0_image=read_3D_volume_from_4D( this->nii_name ,volId);


    this->b0_image= ImageType3D::New();
    this->b0_image->SetRegions(temp_b0_image->GetLargestPossibleRegion());
    this->b0_image->Allocate();
    this->b0_image->SetOrigin(temp_b0_image->GetOrigin());
    this->b0_image->SetDirection(temp_b0_image->GetDirection());
    this->b0_image->SetSpacing(temp_b0_image->GetSpacing());

    itk::ImageRegionIteratorWithIndex<ImageType3D> ittemp(b0_image,b0_image->GetLargestPossibleRegion());
    ittemp.GoToBegin();
    while(!ittemp.IsAtEnd())
    {
        ImageType3D::IndexType tind= ittemp.GetIndex();
        ittemp.Set(temp_b0_image->GetPixel(tind));
        ++ittemp;
    }

    /* Getting Transformation matrix from header to transform to Phys Coor.*/
   ImageType3D::DirectionType dir = b0_image->GetDirection();
   ImageType3D::SpacingType spc = b0_image->GetSpacing();
   ImageType3D::PointType origin = b0_image->GetOrigin();
   ImageType3D::SizeType sz= b0_image->GetLargestPossibleRegion().GetSize();

   vnl_vector<double> orig_vec(3,0);
   orig_vec[0]=origin[0];
   orig_vec[1]=origin[1];
   orig_vec[2]=origin[2];

   vnl_matrix<double> dicom_to_it_transformation(3,3);
   dicom_to_it_transformation.set_identity();
   dicom_to_it_transformation(0,0)=-1;
   dicom_to_it_transformation(1,1)=-1;


   vnl_matrix<double> new_direction = dicom_to_it_transformation * dir.GetVnlMatrix();
   vnl_vector<double> new_orig_vec= dicom_to_it_transformation * orig_vec;
   vnl_matrix<double> spc_mat(3,3);
   spc_mat.fill(0);
   spc_mat(0,0)= spc[0];
   spc_mat(1,1)= spc[1];
   spc_mat(2,2)= spc[2];
   vnl_matrix_fixed<double,3,3> nmat= new_direction *spc_mat;
   this->smat.set_identity();
   this->smat(0,0) = nmat(0,0);    this->smat(0,1) = nmat(0,1);     this->smat(0,2) = nmat(0,2);     this->smat(0,3) = new_orig_vec[0];
   this->smat(1,0) = nmat(1,0);    this->smat(1,1) = nmat(1,1);     this->smat(1,2) = nmat(1,2);     this->smat(1,3) = new_orig_vec[1];
   this->smat(2,0) = nmat(2,0);    this->smat(2,1) = nmat(2,1);     this->smat(2,2) = nmat(2,2);     this->smat(2,3) = new_orig_vec[2];
   /* If this is GE scanner */
   if(this->GE_type)
   {
       vnl_matrix<double> mmat = this->smat.extract(3,3,0,0);
       vnl_vector<double> mijk(3);
       mijk[0]= (sz[0]-1)/2.;
       mijk[1]= (sz[1]-1)/2.;
       mijk[2]= (sz[2]-1)/2.;

       auto oktemp = mmat * mijk;
       this->smat(2,3) = - oktemp[2];
       //this->smat(2,3) =  -((b0_image->GetLargestPossibleRegion().GetSize()[2] -1) * spc[2])/2.;
   }
   /* inverse physCoor */
   this->inv_smat= vnl_matrix_inverse<double>(this->smat);
}

void FITNIFTIISO::setMaskImage(double threshold,int erodeFactor){

    if (this->mask_filename == ""){
        std::cout << "... Creating Mask..." << std::endl;
        maskImage = compute_erode_Mask(b0_image,erodeFactor, threshold);
    }
    else{
        /*Read mask*/
        std::cout << "Reading mask_filename: " << this->mask_filename << std::endl;
        using MaskReaderType= itk::ImageFileReader<MaskImageType>;
        MaskReaderType::Pointer reader= MaskReaderType::New();
        reader->SetFileName(mask_filename);
        reader->Update();
        maskImage=reader->GetOutput();
    }

    this->npts = 0;                     // Number of pixel where mask = 1
    itk::ImageRegionIteratorWithIndex<MaskImageType> imageIterator(this->maskImage, this->maskImage->GetLargestPossibleRegion());
    imageIterator.GoToBegin();
    while(!imageIterator.IsAtEnd()){
        if (imageIterator.Get() == 0){
            ++imageIterator;
           continue;
        }
        this->npts+=1;
        ++imageIterator;
    }
}

void FITNIFTIISO::setGradientCalibrationStructure(std::string grad_filename){
    /*If grad_cal file is not specified*/
//    this->gradFn = parser->getGradCalName();
    this->gradFn = grad_filename;
    if (this->gradFn == ""){
        this->grads = new GRADCAL (); // Default setting of gradcal
    }
    else{
        this->grads = new GRADCAL(this->gradFn);
        this->computeGains = 1;
    }
    this->gradCoefOutput = this->grads->get_struct();
}


void FITNIFTIISO::setVectorOfFitting(){
    nx = grads->get_X_key().size()/2;
    ny = grads->get_Y_key().size()/2;
    nz = grads->get_Z_key().size()/2;

    na = nx + ny + nz + 2;    /* Include S0 and dd */
    /*Init aa array from coefficient array */

    this->gradCoefOutput = this->grads->get_struct();
    if (!this->computeGains){
        this-> aa = vnl_vector<double> (na,0);
        this-> fita = vnl_vector<int> (na,1);
    }
    else{
        this-> aa = vnl_vector<double> (na + 3,0);    /* Include gainx, gainy,gainz */
        this-> fita = vnl_vector<int> (na + 3,0);
        /* Set for only fitting gains */
        for (int i = 0; i < 3; i ++){
            this->fita[na+3-i-1] = 1;
            this->aa[na+3-i-1] = 1;
        }
    }

    for  (int i = 0; i < nx; i ++){
        aa(i+1) = grads->get_X_coef()[i];
    }
    for  (int i = 0; i < ny; i ++){
        aa(i+nx+1) = grads->get_Y_coef()[i];
    }
    for  (int i = 0; i < nz; i ++){
        aa(i+nx+ny+1) = grads->get_Z_coef()[i];
    }
    aa(0) = 1;
}

void FITNIFTIISO::setNormalizingFlag(){
    nnodif = 0;
    vnl_vector<double> bvals = (Bmatrix.get_column(0) + Bmatrix.get_column(3)+ Bmatrix.get_column(5))/1000;
    if (fita[0] == 0){

        for(int v=0;v<this->nVols;v++){
            if (bvals(v) <=0){
                nodif.push_back(v);
                nnodif +=1;
            }
        }
    }
    /* From here decide if we are gonna do image normalization*/
    if (nnodif > 1){
        this->compute_normalization = true;}
}

void FITNIFTIISO::setPermitFlagsforFitting(){
    /* Set FitA */
    std::vector<int> xkey = this->grads->get_X_key();
    std::vector<int> ykey = this->grads->get_X_key();
    std::vector<int> zkey = this->grads->get_X_key();

    for (int i = 0; i <= xkey.size()/2; i++){
        if ((xkey[i*2] == 1) && (xkey[i*2+1] == 1)){
            this->locx=i;
        }
    }
    for (int i = 0; i <= ykey.size()/2; i++){
        if ((ykey[i*2] == 1) && (ykey[i*2+1] == -1)){
            this->locy=i;
        }
    }
    for (int i = 0; i <= zkey.size()/2; i++){
        if ((zkey[i*2] == 1) && (zkey[i*2+1] == 0)){
            this->locz=i;
        }
    }
    /* Working on fita to define tp variabl e*/
    /* When tp = 0; then  assume that calibration is correct, default*/
    if (!this->computeCalibration)
    {
        fita[0] = 0;
        if (xkey.size() != 0){
            this->fita[locx+1] = 0;}
        if (ykey.size() != 0){
            this->fita[locy+nx+1] = 0;}
        if (zkey.size() != 0){
            this->fita[locz+nx+ny+1] = 0;}
    }
    else{
        /* When tp = 1; Compute gradient miscalibration*/
        this->fita[0] = 0;
        this->fita[na-1]=0; /*dont fit diffusitivity (system is singular if diffusivity is fit */
    }
}

void FITNIFTIISO::normalize_image_nodif(std::vector<int> nodif, int nnodif){

    itk::ImageRegionIteratorWithIndex<MaskImageType> imageIterator(this->maskImage, this->maskImage->GetLargestPossibleRegion());
    ImageType3D::IndexType index;
    imageIterator.GoToBegin();

    DTImageType4D::IndexType ind4;
    /* Note: Only Normalize index in the mask */
    while(!imageIterator.IsAtEnd())
    {
        if (imageIterator.Get() == 0){
            ++imageIterator;
           continue;
        }
        index = imageIterator.GetIndex();
        ind4[0] = index[0];
        ind4[1] = index[1];
        ind4[2] = index[2];
        ind4[3] = 1;
        double norm = 0;
        for (int v = 0; v<nnodif; v ++){
            ind4[3] = nodif[v];
            double val = this->my_image->GetPixel(ind4);
            norm += val;
        }
        norm = norm/nnodif;
        if(norm <=0)
        {
            imageIterator.Set(0);
        }
        else
        {
            for (int v = 0; v<this->nVols; v ++){
                ind4[3] = v;
                double val = this->my_image->GetPixel(ind4);
                val = val/norm;
                this->my_image->SetPixel(ind4, val);
            }
        }

        ++imageIterator;
    }
}


void FITNIFTIISO::normalize_image_b0(){
    /* Normalize image to b0, assume b0-image is the first vol */
    itk::ImageRegionIteratorWithIndex<MaskImageType> imageIterator(this->maskImage, this->maskImage->GetLargestPossibleRegion());
    ImageType3D::IndexType index;
    imageIterator.GoToBegin();

    DTImageType4D::IndexType ind4;
    /* Note: Only Normalize index in the mask */

    while(!imageIterator.IsAtEnd())
    {
        if (imageIterator.Get() == 0){
            ++imageIterator;
           continue;
        }
        index = imageIterator.GetIndex();
        ind4[0] = index[0];
        ind4[1] = index[1];
        ind4[2] = index[2];
        ind4[3] = 0;
        double norm = this->my_image->GetPixel(ind4);
        if(norm<=0)
        {
            imageIterator.Set(0);
        }
        else
        {
            for (int v = 0; v< this->nVols; v ++)
            {
                ind4[3] = v;
                double val = this->my_image->GetPixel(ind4);

                val = val/norm;
                this->my_image->SetPixel(ind4, val);
            }
        }

        ++imageIterator;
    }

}

double FITNIFTIISO::regression_at_point(ImageType3D::IndexType ind3)
{
    ImageType4D::IndexType ind4;
    ind4[0]=ind3[0];
    ind4[1]=ind3[1];
    ind4[2]=ind3[2];

    int nv = this->nVols;

    vnl_matrix<double> design_matrix(this->nVols,2);
    vnl_diag_matrix<double> weights(nv,0.0);
    vnl_matrix<double> log_signal(this->nVols,1,0);
    for(int v=0;v < nv ;v++)
    {
        ind4[3]=v;
        design_matrix(v,0)=1;
        design_matrix(v,1)= -(this->Bmatrix(v,0)+this->Bmatrix(v,3)+this->Bmatrix(v,5))/1000.;
        double val = this->my_image->GetPixel(ind4);
        log_signal(v,0)= std::log(val);
        weights(v,v)= val*val;
    }
    vnl_matrix<double> mid= design_matrix.transpose()* weights * design_matrix;
    vnl_matrix<double> D= vnl_svd<double>(mid).solve(design_matrix.transpose()*weights*log_signal);
    double dd =D(1,0);
    return dd;
}








/*; PURPOSE: Use with curfit to measure field produced by gradient coils from diffusion data*/

int IsoGWOkan(int m, int n, vnl_vector<double> &p, vnl_vector<double> &yFit,   vnl_matrix<double> &derivs, void *vars){
    /*Note: m : number of datapoint
     * n: variables;
     * deviates and derivs are outputs to fit to mp_fit
     * deviates 1D
     * derivs 2D */
    struct vars_struct  *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat = v->Bmat;
    vnl_vector<double> *weights= v->weights;
    double dd = p[n-1];

    derivs.fill(0);



    double S0 = p[0];
    GradCoef *E  = v->E;
    vnl_vector<double> *Y = v->Image1Darray;
    int  nmaskpts = v->nmaskpts;
    basisFuntions *basis = v->Basis;
    mp_par *pars=v->pars;
    v->count+=1;

    int nx = E->Xkeys.size()/2;
    int ny = E->Ykeys.size()/2;
    int nz = E->Zkeys.size()/2;
    int na = nx + ny + nz + 2;



    int nVols =  Bmat->rows();    // Number of diffusion direction


    /*std::cout << "this aa: " << std::endl;
    for (int i  = 0; i < n; i ++) {
        std::cout << "at index: " << i << " " << p[i] << std::endl;
    }*/

    int Ncores=omp_get_max_threads();

    std::vector< std::vector<int> > pids;
    pids.resize(Ncores);

    int p_per_core = nmaskpts/Ncores;

    int last=0;
    for(int c=0;c<Ncores-1;c++)
    {

        for(int i=0;i<p_per_core;i++)
            pids[c].push_back(last+i);

        last+=p_per_core;
    }
    for(int i=last;i<nmaskpts;i++)
        pids[Ncores-1].push_back(i);


    //for (int pp = 0; pp < nmaskpts ; pp ++)

#pragma omp parallel for
    for(int c=0;c<Ncores;c++)
    {
        for(int pok=0;pok<pids[c].size();pok++)
        {
            int pp= pids[c][pok];

            double axx = 1;
            double axy = 0;
            double axz = 0;
            double ayy = 1;
            double ayz = 0;
            double ayx = 0;
            double azx = 0;
            double azy = 0;
            double azz = 1;

            vnl_matrix <double> trans_mat(3,6,0);
            vnl_matrix <double> trans_mat1(3,6,0);
            vnl_matrix <double> trans_mat2(3,6,0);
            vnl_matrix <double> trans_mat3(3,6,0);
            vnl_matrix <double> trans_mat4(3,6,0);
            vnl_matrix <double> trans_mat5(3,6,0);
            vnl_matrix <double> trans_mat6(3,6,0);
            vnl_matrix <double> trans_mat7(3,6,0);
            vnl_matrix <double> trans_mat8(3,6,0);
            vnl_matrix <double> trans_mat9(3,6,0);

            vnl_matrix<double> xbasis =  basis->xbasis[pp];
            vnl_matrix<double> ybasis =  basis->ybasis[pp];
            vnl_matrix<double> zbasis =  basis->zbasis[pp];

          //  if(pp==224644)
            //    std::cout<<xbasis<<std::endl;


            for (int kk = 0; kk < nx; kk++)
            {

                axx +=p[kk + 1]*xbasis(0,kk);
                ayx +=p[kk + 1]*xbasis(1,kk);
                azx +=p[kk + 1]*xbasis(2,kk);

            }
            for (int kk = 0; kk < ny; kk++)
            {
                axy +=p[kk + nx + 1]*ybasis(0,kk);
                ayy +=p[kk + nx + 1]*ybasis(1,kk);
                azy +=p[kk + nx + 1]*ybasis(2,kk);

            }
            for (int kk = 0; kk < nz; kk++)
            {
                axz +=p[kk+nx+ny + 1]*zbasis(0,kk);
                ayz +=p[kk+nx+ny + 1]*zbasis(1,kk);
                azz +=p[kk+nx+ny + 1]*zbasis(2,kk);
            }
            // Check
            trans_mat(0,0) = axx * axx;
            trans_mat(0,1) = axx * axy;
            trans_mat(0,2) = axx * axz;
            trans_mat(0,3) = axy * axy;
            trans_mat(0,4) = axy * axz;
            trans_mat(0,5) = axz * axz;
            trans_mat(1,0) = ayx * ayx;
            trans_mat(1,1) = ayx * ayy;
            trans_mat(1,2) = ayx * ayz;
            trans_mat(1,3) = ayy * ayy;
            trans_mat(1,4) = ayy * ayz;
            trans_mat(1,5) = ayz * ayz;
            trans_mat(2,0) = azx * azx;
            trans_mat(2,1) = azx * azy;
            trans_mat(2,2) = azx * azz;
            trans_mat(2,3) = azy * azy;
            trans_mat(2,4) = azy * azz;
            trans_mat(2,5) = azz * azz;

            trans_mat1(0,0) = axx * 2;
            trans_mat1(0,1) = axy;
            trans_mat1(0,2) = axz;

            trans_mat2(0,1) = axx ;
            trans_mat2(0,3) = axy * 2;
            trans_mat2(0,4) = axz;

            trans_mat3(0,2) = axx;
            trans_mat3(0,4) = axy;
            trans_mat3(0,5) = axz * 2;


            trans_mat4(1,0) = ayx * 2;
            trans_mat4(1,1)= ayy;
            trans_mat4(1,2)= ayz;

            trans_mat5(1,1) = ayx ;
            trans_mat5(1,3) = ayy * 2;
            trans_mat5(1,4) = ayz;

            trans_mat6(1,2) = ayx ;
            trans_mat6(1,4) = ayy ;
            trans_mat6(1,5) = ayz * 2;

            trans_mat7(2,0) = azx * 2;
            trans_mat7(2,1) = azy;
            trans_mat7(2,2) = azz;

            trans_mat8(2,1) = azx ;
            trans_mat8(2,3) = azy * 2;
            trans_mat8(2,4) = azz;

            trans_mat9(2,2) = azx;
            trans_mat9(2,4) = azy;
            trans_mat9(2,5) = azz * 2 ;

            /* transfrom to b-matrice */
            vnl_matrix<double> tmp_mat =  (*Bmat) * trans_mat.transpose();
            vnl_matrix<double> tmp_mat1 = (*Bmat) * trans_mat1.transpose();
            vnl_matrix<double> tmp_mat2 = (*Bmat) * trans_mat2.transpose();
            vnl_matrix<double> tmp_mat3 = (*Bmat) * trans_mat3.transpose();
            vnl_matrix<double> tmp_mat4 = (*Bmat) * trans_mat4.transpose();
            vnl_matrix<double> tmp_mat5 = (*Bmat) * trans_mat5.transpose();
            vnl_matrix<double> tmp_mat6 = (*Bmat) * trans_mat6.transpose();
            vnl_matrix<double> tmp_mat7 = (*Bmat) * trans_mat7.transpose();
            vnl_matrix<double> tmp_mat8 = (*Bmat) * trans_mat8.transpose();
            vnl_matrix<double> tmp_mat9 = (*Bmat) * trans_mat9.transpose();


            vnl_vector<double> tmp_(tmp_mat.rows(),0);
            vnl_vector<double> tmp_1(tmp_mat1.rows(),0);
            vnl_vector<double> tmp_2(tmp_mat2.rows(),0);
            vnl_vector<double> tmp_3(tmp_mat3.rows(),0);
            vnl_vector<double> tmp_4(tmp_mat4.rows(),0);
            vnl_vector<double> tmp_5(tmp_mat5.rows(),0);
            vnl_vector<double> tmp_6(tmp_mat6.rows(),0);
            vnl_vector<double> tmp_7(tmp_mat7.rows(),0);
            vnl_vector<double> tmp_8(tmp_mat8.rows(),0);
            vnl_vector<double> tmp_9(tmp_mat9.rows(),0);
            //sum of row
            for (int r = 0; r < tmp_mat.rows(); r ++ ){
                double s[10] = {0};
                for (int c = 0; c < tmp_mat.cols(); c++){
                    s[0]+=tmp_mat[r][c];
                    s[1]+=tmp_mat1[r][c];
                    s[2]+=tmp_mat2[r][c];
                    s[3]+=tmp_mat3[r][c];
                    s[4]+=tmp_mat4[r][c];
                    s[5]+=tmp_mat5[r][c];
                    s[6]+=tmp_mat6[r][c];
                    s[7]+=tmp_mat7[r][c];
                    s[8]+=tmp_mat8[r][c];
                    s[9]+=tmp_mat9[r][c];

                }
                tmp_[r] = s[0];
                tmp_1[r] = s[1];
                tmp_2[r] = s[2];
                tmp_3[r] = s[3];
                tmp_4[r] = s[4];
                tmp_5[r] = s[5];
                tmp_6[r] = s[6];
                tmp_7[r] = s[7];
                tmp_8[r] = s[8];
                tmp_9[r] = s[9];
            }
            /* Computing according array 4D, Note: To many loops */

            double tmp0, tmp1, tmp2;
            for (int vv = 0; vv < nVols; vv++)
            {
                double temp0 = std::exp(tmp_[vv] * -dd);
                double ff = temp0 * S0;
                double tempna = (-ff) * tmp_[vv];

                /* Compute derivates output (Yi - Yfit)xWeights */


                yFit[pp + vv*nmaskpts]=ff;

             /*   if(pp ==224643)
                {
                    std::cout<<xbasis<<std::endl;
                    int ma=0;
                }
                */
                //deviates[pp + vv*nmaskpts] = ff - (*Y)[pp + vv*nmaskpts] ;



                if(derivs.rows()!=0)
                {
                    /* If you include S0, then keep the next line, else comment it */
                    if(pars[0].fixed!=1)
                        //derivs[0][pp+ vv*nmaskpts] =temp0;
                        derivs(0,pp+ vv*nmaskpts) =temp0;
                    /* If you include dd, then keep the next line, else comment it */
                    if(pars[na-1].fixed!=1)
                        //derivs[na-1][pp+ vv*nmaskpts] = tempna;
                        derivs(na-1,pp+ vv*nmaskpts) = tempna;

                    tmp0= -dd * ff*tmp_1[vv];
                    tmp1= -dd * ff*tmp_4[vv];
                    tmp2= -dd * ff*tmp_7[vv];
                    double tmp0x = tmp0;
                    double tmp1x = tmp1;
                    double tmp2x = tmp2;
                    /* DEBUG */
                    /*if ((pp == 200) && (vv == 2)){
                        std::cout << "aaaa" << std::endl;
                    }*/

                    for(int kk=0; kk<nx;kk++)
                    {
                        if(pars[kk+1].fixed==1)                        continue;

                        temp0  = tmp0*xbasis(0,kk) + tmp1*xbasis(1,kk) + tmp2*xbasis(2,kk);
                        derivs(kk+1,pp+ vv*nmaskpts) += temp0;
                    }
                    tmp0= - dd * ff*tmp_2[vv];
                    tmp1= - dd * ff*tmp_5[vv];
                    tmp2= - dd * ff*tmp_8[vv];
                    double tmp0y = tmp0;
                    double tmp1y = tmp1;
                    double tmp2y = tmp2;


                    for(int kk=0; kk<ny;kk++){
                        if(pars[kk+nx+1].fixed==1)
                            continue;
                        temp0  = tmp0*ybasis(0,kk) + tmp1*ybasis(1,kk)+ tmp2*ybasis(2,kk);
                        //derivs[kk+ nx +1][pp+ vv*nmaskpts] += temp0;
                        derivs(kk+ nx +1,pp+ vv*nmaskpts) += temp0;
                    }
                    tmp0= -dd * ff*tmp_3[vv];
                    tmp1= -dd * ff*tmp_6[vv];
                    tmp2= -dd * ff*tmp_9[vv];
                    double tmp0z = tmp0;
                    double tmp1z = tmp1;
                    double tmp2z = tmp2;
                    for(int kk=0; kk<nz;kk++){
                        if(pars[kk+nx+ny+1].fixed==1)
                            continue;
                        temp0  =  tmp0*zbasis(0,kk) + tmp1*zbasis(1,kk) + tmp2*zbasis(2,kk);
                        //derivs[kk+nx+ny+1][pp+ vv*nmaskpts] += temp0;
                        derivs(kk+nx+ny+1,pp+ vv*nmaskpts) += temp0;
                    }
                    /*
                    if ((pp == 123) &&  (vv == 20)){
                        std::cout << "at point: "<< pp << " vol: "<< vv << std::endl;
                        std::cout << tmp0x << "\t"<< tmp1x <<"\t" << tmp2x << std::endl;
                        std::cout << tmp0y << "\t"<< tmp1y <<"\t" << tmp2y << std::endl;
                        std::cout << tmp0z << "\t"<< tmp1z <<"\t" << tmp2z << std::endl;
                        std::cout << "my derivs: " << std::endl;
                        for (int i = 0; i < n; i ++){
                            if (pars[i].fixed == 1)
                                continue;
                            std::cout << "derivs at: "<< i <<" "<< derivs[i][pp + vv * nmaskpts] << std::endl;
                        }
                    }
                    */
                }
            }
        }
    }
   // derivs=-derivs;

    return 1;
}



int OkanCurveFit(int m, int n, vnl_vector<double> &p, void *vars)
{
    /*Note: m : number of datapoint
     * n: variables;
     * deviates and derivs are outputs to fit to mp_fit
     * deviates 1D
     * derivs 2D */

    struct vars_struct  *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat = v->Bmat;
    vnl_vector<double> *weights= v->weights;
    GradCoef *E  = v->E;
    int  nmaskpts = v->nmaskpts;
    basisFuntions *basis = v->Basis;
    mp_par *pars=v->pars;
    vnl_vector<double> *Y = v->Image1Darray;





    double TOL=1E-6;
    int itmax=40;
    double flambda= 0.001;


    int nterms=n;
    int nparam=0;


    vnl_vector<int> params_l(nterms,0);
    for(int i=0;i<E->Xkeys.size()/2;i++)
    {
        params_l[1+i]= E->Xkeys[2*i];
    }
    for(int i=0;i<E->Ykeys.size()/2;i++)
    {
        params_l[1+i+E->Xkeys.size()/2]= E->Ykeys[2*i];
    }
    for(int i=0;i<E->Zkeys.size()/2;i++)
    {
        params_l[1+i+E->Xkeys.size()/2+E->Ykeys.size()/2]= E->Zkeys[2*i];
    }



    std::vector<int> iparam;
    for(int a=0;a<nterms;a++)
    {
        if(pars[a].fixed==0)
        {
            iparam.push_back(a);
            nparam++;
        }
    }



   vnl_vector<unsigned int> iparam2(nparam);
   for(int i=0;i<nparam;i++)
       iparam2[i]=iparam[i];

    int nY=m;
    int nfree= nY- nparam;

    vnl_vector<double> sigma(nterms,0);
    vnl_vector<double> sigma1(nparam,0);
    vnl_vector<double> yFit(nY,0);
    vnl_matrix<double> pder(nterms,nY,0);


    vnl_matrix<double> dummy(0,0);


    bool done=0;

    for(int iter=0;iter<itmax;iter++)
    {
        if(done)
            break;

        IsoGWOkan(m, n,  p, yFit , pder, vars);
        vnl_matrix<double> pder_param= pder.get_rows(iparam2);



        vnl_vector<double> resids= (*Y) - yFit;

        vnl_vector<double> beta = pder_param * resids;


        vnl_matrix<double> alpha(nparam,nparam);
#pragma omp parallel for
        for(int m=0;m<nparam;m++)
        {
            for(int n=0;n<nparam;n++)
            {
                double sm=0;

                for(int k=0;k<nY;k++)
                {
                    sm+=pder_param(m,k) * pder_param(n,k);
                }
                alpha(m,n)=sm;
            }
        }




        for(int i=0;i<nparam;i++)
        {
            sigma1[i]=sqrt(1./alpha(i,i));
            sigma[iparam[i]]= sigma1[i];
        }

        double chisq1= resids.squared_magnitude()/nfree;
        double chisq=chisq1+1;
        std::cout<<"Error: "<< chisq1 << std::endl;

        vnl_vector<double> yfit1= yFit;


        if(chisq1 < 1.* (*Y).one_norm()/1E10/nfree   )
        {
            done=true;
            break;
        }


        vnl_vector<double> c(nparam);
        for(int i=0;i<nparam;i++)
            c[i]=1./sigma1[i];

        vnl_matrix<double> c2= outer_product(c,c);

        long lambdaCount = 0;
        vnl_vector<double> b;

        while(chisq> chisq1)
        {
            lambdaCount++;
            vnl_matrix<double> array(nparam,nparam);


            for(int m=0;m<nparam;m++)
            {
                for(int n=0;n<nparam;n++)
                    array(m,n)= alpha(m,n)/c2(m,n);

                //array(m,m)=array(m,m) *(1.+flambda);

                double fmult=params_l[iparam[m]]*params_l[iparam[m]]*(params_l[iparam[m]]+1)*(params_l[iparam[m]]+1) ;
                double lmult= fmult*fmult*fmult*fmult;

                array(m,m)=array(m,m) *(1. + flambda * lmult   );

            }

            array= vnl_matrix_inverse<double>(array);
            b=p;

            vnl_matrix<double> array_div_c(nparam,nparam);
            for(int m=0;m<nparam;m++)
            {
                for(int n=0;n<nparam;n++)
                    array_div_c(m,n)= array(m,n)/c2(m,n);
            }



            vnl_vector<double> update= array_div_c * beta;

            for(int m=0;m<nparam;m++)
            {
                b[iparam[m]]= p[iparam[m]] + update[m];
            }



            IsoGWOkan(m, n,  b, yFit , dummy, vars);
            resids= (*Y) - yFit;
            chisq= resids.squared_magnitude()/nfree;


            for(int m=0;m<nparam;m++)
            {
                sigma[iparam[m]] = sqrt(array(m,m)/alpha(m,m) );
            }

            if((!isfinite(chisq)) ||  ( (lambdaCount > 30) && (chisq >= chisq1))   )
            {
                yFit=yfit1;
                for(int m=0;m<nparam;m++)
                {
                    sigma[iparam[m]] = sigma1[m];
                }
                chisq=chisq1;

                std::cout<<"Failed to converge- CHISQ increasing without bound."<<std::endl;
                done=1;
                break;
            }

            flambda*=10;

        } //while loop

        flambda/=100.;
        for(int m=0;m<nparam;m++)
            p[iparam[m]]=b[iparam[m]];

        if( (chisq1-chisq)/chisq1   <= TOL /1E5 )
        {
            done=true;
            break;
        }
    }//for loop iter


    //sigma*=sqrt(chisq);
    return EXIT_SUCCESS;

}










/*; PURPOSE: Use with curfit to measure field produced by gradient coils from diffusion data*/


/*
int IsoGW(int m, int n, double *p, double *deviates,   double **derivs, void *vars){

    struct vars_struct  *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat = v->Bmat;
    vnl_vector<double> *weights= v->weights;
    double dd = p[n-1];
    double S0 = p[0];
    GradCoef *E  = v->E;
    vnl_vector<double> *Y = v->Image1Darray;
    int  nmaskpts = v->nmaskpts;
    basisFuntions *basis = v->Basis;
    mp_par *pars=v->pars;
    v->count+=1;

    int nx = E->Xkeys.size()/2;
    int ny = E->Ykeys.size()/2;
    int nz = E->Zkeys.size()/2;
    int na = nx + ny + nz + 2;

    std::vector <double>xbasis;
    std::vector <double>ybasis;
    std::vector <double>zbasis;

    int nVols =  Bmat->rows();    // Number of diffusion direction
    double sm = 0;


    for (int pp = 0; pp < nmaskpts ; pp ++){
        double axx = 1;
        double axy = 0;
        double axz = 0;
        double ayy = 1;
        double ayz = 0;
        double ayx = 0;
        double azx = 0;
        double azy = 0;
        double azz = 1;

        vnl_matrix <double> trans_mat(3,6,0);
        vnl_matrix <double> trans_mat1(3,6,0);
        vnl_matrix <double> trans_mat2(3,6,0);
        vnl_matrix <double> trans_mat3(3,6,0);
        vnl_matrix <double> trans_mat4(3,6,0);
        vnl_matrix <double> trans_mat5(3,6,0);
        vnl_matrix <double> trans_mat6(3,6,0);
        vnl_matrix <double> trans_mat7(3,6,0);
        vnl_matrix <double> trans_mat8(3,6,0);
        vnl_matrix <double> trans_mat9(3,6,0);


        for (int kk = 0; kk < nx; kk++){
            xbasis =  basis->xbasis[pp*nx+kk];
            axx +=p[kk + 1]*xbasis[0];
            ayx +=p[kk + 1]*xbasis[1];
            azx +=p[kk + 1]*xbasis[2];

        }
        for (int kk = 0; kk < ny; kk++){
            ybasis =  basis->ybasis[pp*ny+kk];
            axy +=p[kk + nx + 1]*ybasis[0];
            ayy +=p[kk + nx + 1]*ybasis[1];
            azy +=p[kk + nx + 1]*ybasis[2];

        }
        for (int kk = 0; kk < nz; kk++){
            zbasis = basis->zbasis[pp*nz+kk];
            axz +=p[kk+nx+ny + 1]*zbasis[0];
            ayz +=p[kk+nx+ny + 1]*zbasis[1];
            azz +=p[kk+nx+ny + 1]*zbasis[2];
        }
        // Check
        trans_mat(0,0) = axx * axx;
        trans_mat(0,1) = axx * axy;
        trans_mat(0,2) = axx * axz;
        trans_mat(0,3) = axy * axy;
        trans_mat(0,4) = axy * axz;
        trans_mat(0,5) = axz * axz;
        trans_mat(1,0) = ayx * ayx;
        trans_mat(1,1) = ayx * ayy;
        trans_mat(1,2) = ayx * ayz;
        trans_mat(1,3) = ayy * ayy;
        trans_mat(1,4) = ayy * ayz;
        trans_mat(1,5) = ayz * ayz;
        trans_mat(2,0) = azx * azx;
        trans_mat(2,1) = azx * azy;
        trans_mat(2,2) = azx * azz;
        trans_mat(2,3) = azy * azy;
        trans_mat(2,4) = azy * azz;
        trans_mat(2,5) = azz * azz;

        trans_mat1(0,0) = axx * 2;
        trans_mat1(0,1) = axy;
        trans_mat1(0,2) = axz;

        trans_mat2(0,1) = axx ;
        trans_mat2(0,3) = axy * 2;
        trans_mat2(0,4) = axz;

        trans_mat3(0,2) = axx;
        trans_mat3(0,4) = axy;
        trans_mat3(0,5) = axz * 2;


        trans_mat4(1,0) = ayx * 2;
        trans_mat4(1,1)= ayy;
        trans_mat4(1,2)= ayz;

        trans_mat5(1,1) = ayx ;
        trans_mat5(1,3) = ayy * 2;
        trans_mat5(1,4) = ayz;

        trans_mat6(1,2) = ayx ;
        trans_mat6(1,4) = ayy ;
        trans_mat6(1,5) = ayz * 2;

        trans_mat7(2,0) = azx * 2;
        trans_mat7(2,1) = azy;
        trans_mat7(2,2) = azz;

        trans_mat8(2,1) = azx ;
        trans_mat8(2,3) = azy * 2;
        trans_mat8(2,4) = azz;

        trans_mat9(2,2) = azx;
        trans_mat9(2,4) = azy;
        trans_mat9(2,5) = azz * 2 ;

        // transfrom to b-matrice
        vnl_matrix<double> tmp_mat =  (*Bmat) * trans_mat.transpose();
        vnl_matrix<double> tmp_mat1 = (*Bmat) * trans_mat1.transpose();
        vnl_matrix<double> tmp_mat2 = (*Bmat) * trans_mat2.transpose();
        vnl_matrix<double> tmp_mat3 = (*Bmat) * trans_mat3.transpose();
        vnl_matrix<double> tmp_mat4 = (*Bmat) * trans_mat4.transpose();
        vnl_matrix<double> tmp_mat5 = (*Bmat) * trans_mat5.transpose();
        vnl_matrix<double> tmp_mat6 = (*Bmat) * trans_mat6.transpose();
        vnl_matrix<double> tmp_mat7 = (*Bmat) * trans_mat7.transpose();
        vnl_matrix<double> tmp_mat8 = (*Bmat) * trans_mat8.transpose();
        vnl_matrix<double> tmp_mat9 = (*Bmat) * trans_mat9.transpose();


        vnl_vector<double> tmp_(tmp_mat.rows(),0);
        vnl_vector<double> tmp_1(tmp_mat1.rows(),0);
        vnl_vector<double> tmp_2(tmp_mat2.rows(),0);
        vnl_vector<double> tmp_3(tmp_mat3.rows(),0);
        vnl_vector<double> tmp_4(tmp_mat4.rows(),0);
        vnl_vector<double> tmp_5(tmp_mat5.rows(),0);
        vnl_vector<double> tmp_6(tmp_mat6.rows(),0);
        vnl_vector<double> tmp_7(tmp_mat7.rows(),0);
        vnl_vector<double> tmp_8(tmp_mat8.rows(),0);
        vnl_vector<double> tmp_9(tmp_mat9.rows(),0);
        //sum of row
        for (int r = 0; r < tmp_mat.rows(); r ++ ){
            double s[10] = {0};
            for (int c = 0; c < tmp_mat.cols(); c++){
                s[0]+=tmp_mat[r][c];
                s[1]+=tmp_mat1[r][c];
                s[2]+=tmp_mat2[r][c];
                s[3]+=tmp_mat3[r][c];
                s[4]+=tmp_mat4[r][c];
                s[5]+=tmp_mat5[r][c];
                s[6]+=tmp_mat6[r][c];
                s[7]+=tmp_mat7[r][c];
                s[8]+=tmp_mat8[r][c];
                s[9]+=tmp_mat9[r][c];

            }
            tmp_[r] = s[0];
            tmp_1[r] = s[1];
            tmp_2[r] = s[2];
            tmp_3[r] = s[3];
            tmp_4[r] = s[4];
            tmp_5[r] = s[5];
            tmp_6[r] = s[6];
            tmp_7[r] = s[7];
            tmp_8[r] = s[8];
            tmp_9[r] = s[9];
        }
        //Computing according array 4D, Note: To many loops

        double tmp0, tmp1, tmp2;
        for (int vv = 0; vv < nVols; vv++){
            double temp0 = std::exp(tmp_[vv] * -dd);
            double ff = temp0 * S0;
            double tempna = (-ff) * tmp_[vv];

            // Compute derivates output (Yi - Yfit)xWeights


            deviates[pp + vv*nmaskpts] = ff - (*Y)[pp + vv*nmaskpts] ;
            //deviates[pp + vv*nmaskpts] =  (*Y)[pp + vv*nmaskpts] -ff;

            sm += deviates[pp + vv*nmaskpts]*deviates[pp + vv*nmaskpts];
            if(derivs!=nullptr)
            {
                // If you include S0, then keep the next line, else comment it
                if(pars[0].fixed!=1)
                    derivs[0][pp+ vv*nmaskpts] =temp0;
                // If you include dd, then keep the next line, else comment it
                if(pars[na-1].fixed!=1)
                    derivs[na-1][pp+ vv*nmaskpts] = tempna;

                tmp0= -dd * ff*tmp_1[vv];
                tmp1= -dd * ff*tmp_4[vv];
                tmp2= -dd * ff*tmp_7[vv];
                double tmp0x = tmp0;
                double tmp1x = tmp1;
                double tmp2x = tmp2;


                for(int kk=0; kk<nx;kk++){
                    if(pars[kk+1].fixed==1)
                        continue;
                    xbasis =  basis->xbasis[pp*nx+kk];
                    temp0  = tmp0*xbasis[0] + tmp1*xbasis[1] + tmp2*xbasis[2];
                    derivs[kk+1][pp+ vv*nmaskpts] += temp0;
                }
                tmp0= - dd * ff*tmp_2[vv];
                tmp1= - dd * ff*tmp_5[vv];
                tmp2= - dd * ff*tmp_8[vv];
                double tmp0y = tmp0;
                double tmp1y = tmp1;
                double tmp2y = tmp2;


                for(int kk=0; kk<ny;kk++){
                    if(pars[kk+nx+1].fixed==1)
                        continue;
                    ybasis =  basis->ybasis[pp*ny+kk];
                    temp0  = tmp0*ybasis[0] + tmp1*ybasis[1] + tmp2*ybasis[2];
                    derivs[kk+ nx +1][pp+ vv*nmaskpts] += temp0;
                }
                tmp0= -dd * ff*tmp_3[vv];
                tmp1= -dd * ff*tmp_6[vv];
                tmp2= -dd * ff*tmp_9[vv];
                double tmp0z = tmp0;
                double tmp1z = tmp1;
                double tmp2z = tmp2;
                for(int kk=0; kk<nz;kk++){
                    if(pars[kk+nx+ny+1].fixed==1)
                        continue;
                    zbasis =  basis->zbasis[pp*nz+kk];
                    temp0  =  tmp0*zbasis[0] + tmp1*zbasis[1] + tmp2*zbasis[2];
                    derivs[kk+nx+ny+1][pp+ vv*nmaskpts] += temp0; 
                }

            }
        }
    }

    std::cout << "Yerror: "<< sm/m << std::endl;
    return 1;
}
*/


/*; PURPOSE: Use with curfit to measure field produced by gradient coils from diffusion data*/

/*
int IsoGW1(int m, int n, double *p, double *deviates,   double **derivs, void *vars){
    //Note: m : number of datapoint
     //* n: variables;
     //* deviates and derivs are outputs to fit to mp_fit
     //* deviates 1D
     // derivs 2D

    // This function is different from IsoGw()
    // This performs fitting to compute Gains, not Coeffs
    struct vars_struct  *v = (struct vars_struct *) vars;
    vnl_matrix<double> *Bmat = v->Bmat;
    vnl_vector<double> *weights= v->weights;
    double dd = p[n-1];
    double S0 = p[0];
    GradCoef *E  = v->E;
    vnl_vector<double> *Y = v->Image1Darray;
    int  nmaskpts = v->nmaskpts;
    basisFuntions *basis = v->Basis;
    mp_par *pars=v->pars;

    int nx = E->Xkeys.size()/2;
    int ny = E->Ykeys.size()/2;
    int nz = E->Zkeys.size()/2;
    int na = nx + ny + nz + 2;

    std::vector <double>xbasis;
    std::vector <double>ybasis;
    std::vector <double>zbasis;

    int nVols =  Bmat->rows();    // Number of diffusion direction
    double sm = 0;
    for (int pp = 0; pp < nmaskpts ; pp ++){
        double axx = 1;
        double axy = 0;
        double axz = 0;
        double ayy = 1;
        double ayz = 0;
        double ayx = 0;
        double azx = 0;
        double azy = 0;
        double azz = 1;

        vnl_matrix <double> trans_mat(3,6,0);
        vnl_matrix <double> trans_mat1(3,6,0);
        vnl_matrix <double> trans_mat2(3,6,0);
        vnl_matrix <double> trans_mat3(3,6,0);
        vnl_matrix <double> trans_mat4(3,6,0);
        vnl_matrix <double> trans_mat5(3,6,0);
        vnl_matrix <double> trans_mat6(3,6,0);
        vnl_matrix <double> trans_mat7(3,6,0);
        vnl_matrix <double> trans_mat8(3,6,0);
        vnl_matrix <double> trans_mat9(3,6,0);


        for (int kk = 0; kk < nx; kk++){
            xbasis =  basis->xbasis[pp*nx+kk];
            axx +=p[kk + 1]*xbasis[0] * p[na];
            ayx +=p[kk + 1]*xbasis[1] * p[na];
            azx +=p[kk + 1]*xbasis[2] * p[na];

        }
        for (int kk = 0; kk < ny; kk++){
            ybasis =  basis->ybasis[pp*ny+kk] ;
            axy +=p[kk + nx + 1]*ybasis[0] * p[na+1];
            ayy +=p[kk + nx + 1]*ybasis[1] * p[na+1];
            azy +=p[kk + nx + 1]*ybasis[2] * p[na+1];

        }
        for (int kk = 0; kk < nz; kk++){
            zbasis = basis->zbasis[pp*nz+kk] ;
            axz +=p[kk+nx+ny + 1]*zbasis[0] * p[na+2];
            ayz +=p[kk+nx+ny + 1]*zbasis[1] * p[na+2];
            azz +=p[kk+nx+ny + 1]*zbasis[2] * p[na+2];
        }

        // Check
        trans_mat(0,0) = axx * axx;
        trans_mat(0,1) = axx * axy;
        trans_mat(0,2) = axx * axz;
        trans_mat(0,3) = axy * axy;
        trans_mat(0,4) = axy * axz;
        trans_mat(0,5) = axz * axz;
        trans_mat(1,0) = ayx * ayx;
        trans_mat(1,1) = ayx * ayy;
        trans_mat(1,2) = ayx * ayz;
        trans_mat(1,3) = ayy * ayy;
        trans_mat(1,4) = ayy * ayz;
        trans_mat(1,5) = ayz * ayz;
        trans_mat(2,0) = azx * azx;
        trans_mat(2,1) = azx * azy;
        trans_mat(2,2) = azx * azz;
        trans_mat(2,3) = azy * azy;
        trans_mat(2,4) = azy * azz;
        trans_mat(2,5) = azz * azz;

        trans_mat1(0,0) = axx * 2;
        trans_mat1(0,1) = axy;
        trans_mat1(0,2) = axz;

        trans_mat2(0,1) = axx ;
        trans_mat2(0,3) = axy * 2;
        trans_mat2(0,4) = axz;

        trans_mat3(0,2) = axx;
        trans_mat3(0,4) = axy;
        trans_mat3(0,5) = axz * 2;


        trans_mat4(1,0) = ayx * 2;
        trans_mat4(1,1)= ayy;
        trans_mat4(1,2)= ayz;

        trans_mat5(1,1) = ayx ;
        trans_mat5(1,3) = ayy * 2;
        trans_mat5(1,4) = ayz;

        trans_mat6(1,2) = ayx ;
        trans_mat6(1,4) = ayy ;
        trans_mat6(1,5) = ayz * 2;

        trans_mat7(2,0) = azx * 2;
        trans_mat7(2,1) = azy;
        trans_mat7(2,2) = azz;

        trans_mat8(2,1) = azx ;
        trans_mat8(2,3) = azy * 2;
        trans_mat8(2,4) = azz;

        trans_mat9(2,2) = azx;
        trans_mat9(2,4) = azy;
        trans_mat9(2,5) = azz * 2 ;

        // transfrom to b-matrice
        vnl_matrix<double> tmp_mat =  (*Bmat) * trans_mat.transpose();
        vnl_matrix<double> tmp_mat1 = (*Bmat) * trans_mat1.transpose();
        vnl_matrix<double> tmp_mat2 = (*Bmat) * trans_mat2.transpose();
        vnl_matrix<double> tmp_mat3 = (*Bmat) * trans_mat3.transpose();
        vnl_matrix<double> tmp_mat4 = (*Bmat) * trans_mat4.transpose();
        vnl_matrix<double> tmp_mat5 = (*Bmat) * trans_mat5.transpose();
        vnl_matrix<double> tmp_mat6 = (*Bmat) * trans_mat6.transpose();
        vnl_matrix<double> tmp_mat7 = (*Bmat) * trans_mat7.transpose();
        vnl_matrix<double> tmp_mat8 = (*Bmat) * trans_mat8.transpose();
        vnl_matrix<double> tmp_mat9 = (*Bmat) * trans_mat9.transpose();


        vnl_vector<double> tmp_(tmp_mat.rows(),0);
        vnl_vector<double> tmp_1(tmp_mat1.rows(),0);
        vnl_vector<double> tmp_2(tmp_mat2.rows(),0);
        vnl_vector<double> tmp_3(tmp_mat3.rows(),0);
        vnl_vector<double> tmp_4(tmp_mat4.rows(),0);
        vnl_vector<double> tmp_5(tmp_mat5.rows(),0);
        vnl_vector<double> tmp_6(tmp_mat6.rows(),0);
        vnl_vector<double> tmp_7(tmp_mat7.rows(),0);
        vnl_vector<double> tmp_8(tmp_mat8.rows(),0);
        vnl_vector<double> tmp_9(tmp_mat9.rows(),0);
        //sum of row
        for (int r = 0; r < tmp_mat.rows(); r ++ ){
            double s[10] = {0};
            for (int c = 0; c < tmp_mat.cols(); c++){
                s[0]+=tmp_mat[r][c];
                s[1]+=tmp_mat1[r][c];
                s[2]+=tmp_mat2[r][c];
                s[3]+=tmp_mat3[r][c];
                s[4]+=tmp_mat4[r][c];
                s[5]+=tmp_mat5[r][c];
                s[6]+=tmp_mat6[r][c];
                s[7]+=tmp_mat7[r][c];
                s[8]+=tmp_mat8[r][c];
                s[9]+=tmp_mat9[r][c];

            }
            tmp_[r] = s[0];
            tmp_1[r] = s[1];
            tmp_2[r] = s[2];
            tmp_3[r] = s[3];
            tmp_4[r] = s[4];
            tmp_5[r] = s[5];
            tmp_6[r] = s[6];
            tmp_7[r] = s[7];
            tmp_8[r] = s[8];
            tmp_9[r] = s[9];
        }
        // Computing according array 4D, Note: To many loops
        double tmp0, tmp1, tmp2;

        for (int vv = 0; vv < nVols; vv++){
            double temp0 = std::exp(tmp_[vv] * -dd);
            double ff = temp0 * S0;
            double tempna = (-ff) * tmp_[vv];

            // Compute derivates output (Yi - Yfit)xWeights
            deviates[pp + vv*nmaskpts] = ff - (*Y)[pp + vv*nmaskpts] ;
            sm += deviates[pp + vv*nmaskpts];

            if(derivs!=nullptr)
            {

                // If you include S0, then keep the next line, else comment it
                if(pars[0].fixed!=1)
                    derivs[0][pp+ vv*nmaskpts] =temp0;
                // If you include dd, then keep the next line, else comment it
                if(pars[na-1].fixed!=1)
                    derivs[na-1][pp+ vv*nmaskpts] = tempna;

                tmp0= -dd * ff*tmp_1[vv];
                tmp1= -dd * ff*tmp_4[vv];
                tmp2= -dd * ff*tmp_7[vv];

                for(int kk=0; kk<nx;kk++){
                    xbasis =  basis->xbasis[pp*nx+kk];
                    temp0  = (tmp0*xbasis[0] + tmp1*xbasis[1] + tmp2*xbasis[2]) * p[na];
                    derivs[na][pp+ vv*nmaskpts] += temp0*p[kk+1];
                }
                tmp0= - dd * ff*tmp_2[vv];
                tmp1= - dd * ff*tmp_5[vv];
                tmp2= - dd * ff*tmp_8[vv];

                for(int kk=0; kk<ny;kk++){
                    ybasis =  basis->ybasis[pp*ny+kk] ;
                    temp0  = (tmp0*ybasis[0] + tmp1*ybasis[1] + tmp2*ybasis[2]) * p[na+1];
                    derivs[na+1][pp+ vv*nmaskpts] += temp0*p[kk+nx+1];
                }
                tmp0= -dd * ff*tmp_3[vv];
                tmp1= -dd * ff*tmp_6[vv];
                tmp2= -dd * ff*tmp_9[vv];


                for(int kk=0; kk<nz;kk++){
                    zbasis =  basis->xbasis[pp*nz+kk];
                    temp0  =  (tmp0*zbasis[0] + tmp1*zbasis[1] + tmp2*zbasis[2])  * p[na+2];
                    derivs[na+2][pp+ vv*nmaskpts] += temp0*p[kk+nx+ny+1];
                }


            }

        }
    }
//    std::cout <<p[na] <<"\t"<<p[na+1] << "\t"<< p[na+2] <<std::endl;
    std::cout << "Yerror: "<< sm/m << std::endl;
    return 1;
}
*/

void FITNIFTIISO::PerformNonlinearityMisCalibrationFitting()
{
    double nser = double(this->nVols);
    int pts = this->npts;

    /* In curvefit, tol = 1e-6 */
    /* Creating and 1D array of 4D image */
    vnl_vector <double> Image1DArray(nser*pts, 0);
    double indice = 0;
    DTImageType4D::IndexType ind4;
    MaskImageType::IndexType ind3;

    itk::ImageRegionIteratorWithIndex<MaskImageType> imageIterator(this->maskImage, this->maskImage->GetLargestPossibleRegion());
    /* Iterate over the volume */
    /* Convert 4D to 1D image */
    for (int v =0; v < nser ; v ++) {
        imageIterator.GoToBegin();
        while(!imageIterator.IsAtEnd()){
            if (imageIterator.Get() == 0){
                ++imageIterator;
               continue;
            }
            ind3 = imageIterator.GetIndex();

            ind4[0] = ind3[0];
            ind4[1] = ind3[1];
            ind4[2] = ind3[2];
            ind4[3] = v;

           Image1DArray(indice) = this->my_image->GetPixel(ind4);
            ++imageIterator;
            indice ++;
        }
    }
    mp_config_struct config;
    config.maxiter=500;
    config.ftol=1E-10;
    config.xtol=1E-10;
    config.gtol=1E-10;
    config.epsfcn=MP_MACHEP0;
    config.stepfactor=100;
    config.covtol=1E-14;
    config.maxfev=0;
    config.nprint=0;
    config.douserscale=0;
    config.nofinitecheck=0;

    mp_result_struct my_results_struct;
    vnl_vector<double> my_resids(nser * pts);
    my_results_struct.resid= my_resids.data_block();
    my_results_struct.xerror=nullptr;
    my_results_struct.covar=nullptr;

    vnl_matrix <double> Bmatrx = this->Bmatrix /1000;
    vars_struct my_struct;
    GradCoef gradA = this->gradCoefOutput;
    my_struct.weights=NULL;
    my_struct.Bmat= &(Bmatrx);
    my_struct.Image1Darray = &(Image1DArray);
    my_struct.Basis= &(this->basis);
    my_struct.nmaskpts = pts;
    my_struct.E = &(gradA);


    int nxcoeffs = my_struct.Basis->xbasis[0].cols();
    int nycoeffs = my_struct.Basis->ybasis[0].cols();
    int nzcoeffs = my_struct.Basis->zbasis[0].cols();
    int ncoeffs = nxcoeffs+nycoeffs+nzcoeffs + 2;


    // Set first coeff = 1
    if (this->computeGains){
        /*IsoGW1() */
        std::cout << "Perform Fitting to compute Gains" << std::endl;
        config.maxiter=500;
        config.ftol=1E-10;
        mp_par *pars=new mp_par[ncoeffs + 3];
        double *coeffs = new double[ncoeffs + 3];

        my_struct.pars= pars;
        coeffs[0] = 1;
        for(int nn=0;nn<ncoeffs + 3; nn++)
        {
            memset(&(pars[nn]), 0, sizeof(mp_par));
            pars[nn].side=3;
            coeffs[nn+1]=this->aa[nn+1];
            if (this->fita[nn] == 0){
                pars[nn].fixed = 1;
            }
         }

       // int status = mpfit(IsoGW1, pts * nser, ncoeffs + 3, coeffs, pars, &config, (void *) &my_struct, &my_results_struct);
        for(int ii=0;ii<ncoeffs+3;ii++)
            this->aa [ii] = coeffs[ii];

        this->phantomD = coeffs[ncoeffs -1 ];
        for (int i = 0; i < nxcoeffs; i ++){
            this->gradCoefOutput.gradX_coef[i + 1] *= coeffs[na] ;
        }
        for (int i = 0; i < nycoeffs; i ++){
            this->gradCoefOutput.gradY_coef[i+ nxcoeffs + 1] *= coeffs[na+1] ;
        }
        for (int i = 0; i < nzcoeffs; i ++){
            this->gradCoefOutput.gradZ_coef[i+ nxcoeffs +  nycoeffs + 1] *= coeffs[na+2] ;
        }
        delete[] pars;
        delete[] coeffs;
    }
    else{
        /*IsoGw()*/
        config.ftol=1E-10;
        config.xtol=1E-10;
        config.gtol=1E-10;
        config.epsfcn=MP_MACHEP0;
        config.stepfactor=100.0;
        config.covtol=1E-14;

        mp_par *pars=new mp_par[ncoeffs];
        double *coeffs = new double[ncoeffs];
        my_struct.pars= pars;
        coeffs[0] = 1;
        for(int nn=0;nn<ncoeffs;nn++)
        {
            memset(&(pars[nn]), 0, sizeof(mp_par));
            pars[nn].side=3;
            //pars[nn].side=0;
           // coeffs[nn+1]=this->aa[nn+1];
             coeffs[nn]=this->aa[nn];
            if (this->fita[nn] == 0)
            {
                pars[nn].fixed = 1;
            }
        }


        /* To Estimate aa = coeffs */
        /* Estimate over 4D image */
        //int status = mpfit(IsoGW, pts * nser, ncoeffs, coeffs, pars, &config, (void *) &my_struct, &my_results_struct);
        //for(int ii=0;ii<ncoeffs;ii++)
        //    this->aa [ii] = coeffs[ii];

        OkanCurveFit(pts * nser, ncoeffs, this->aa, (void *) &my_struct);


       /* After fitting */
        this->gradCoefOutput.gradX_coef.clear();
        this->gradCoefOutput.gradY_coef.clear();
        this->gradCoefOutput.gradZ_coef.clear();
       // this->phantomD = coeffs[ncoeffs -1 ];
         this->phantomD = this->aa[ncoeffs -1 ];


        if (this->locx != -1){
            this->aa[locx+1] +=1;
        }
        if (this->locy != -1){
            this->aa[locy+nx+1] +=1;
        }
        if (this->locy != -1){
            this->aa[locz+nx+ny+1] +=1;
        }
        std::cout << "xcoef after fitting: ";
         for (int i = 0; i < nxcoeffs; i ++){
            this->gradCoefOutput.gradX_coef.push_back(aa[i+1]);
            std::cout << aa[i+1] << "\t";
        }
        std::cout << std::endl;
        std::cout << "ycoef after fitting: ";
        for (int i = 0; i < nycoeffs; i ++){
            this->gradCoefOutput.gradY_coef.push_back(aa[i+ nxcoeffs + 1]);
            std::cout << aa[i + nxcoeffs + 1] << "\t";
        }
        std::cout << std::endl;
        std::cout << "zcoef after fitting: ";
        for (int i = 0; i < nzcoeffs; i ++){
            this->gradCoefOutput.gradZ_coef.push_back(aa[i+nxcoeffs + nycoeffs + 1]);
            std::cout << aa[i + nxcoeffs + nycoeffs + 1] << "\t";
        }
        std::cout << std::endl;

        delete[] pars;
        delete[] coeffs;
    }
}

void FITNIFTIISO::write_gradcal(std::string tempFilename){

    std::ofstream reg_outfile(tempFilename.c_str());
    std::cout << "Save to file: " << tempFilename << std::endl;
    int nxcoeffs = this->gradCoefOutput.gradX_coef.size();
    int nycoeffs = this->gradCoefOutput.gradY_coef.size();
    int nzcoeffs = this->gradCoefOutput.gradZ_coef.size();

    /* X gradients */
    reg_outfile << "\"XCOEF:\"[";
    for (int i = 0; i < nxcoeffs-1; i ++){
        reg_outfile<<this->gradCoefOutput.gradX_coef[i]<<", ";
    }
    reg_outfile<< this->gradCoefOutput.gradX_coef[nxcoeffs-1] << "]," << std::endl;

    reg_outfile << "\"XKEY:\"[";
    for (int i = 0; i < nxcoeffs-1; i ++){
        reg_outfile<<this->gradCoefOutput.Xkeys[2*i]<<", ";
        reg_outfile<<this->gradCoefOutput.Xkeys[2*i + 1]<<", ";
    }
    reg_outfile<<this->gradCoefOutput.Xkeys[2*(nxcoeffs-1)]<<", ";
    reg_outfile<<this->gradCoefOutput.Xkeys[2*(nxcoeffs-1) + 1] << "]," << std::endl;

    /* Y gradients */
    reg_outfile << "\"YCOEF:\"[";
    for (int i = 0; i < nycoeffs-1; i ++){
        reg_outfile<<this->gradCoefOutput.gradY_coef[i]<<", ";
    }
    reg_outfile<< this->gradCoefOutput.gradY_coef[nycoeffs-1] << "]," << std::endl;

    reg_outfile << "\"YKEY:\"[";
    for (int i = 0; i < nycoeffs-1; i ++){
        reg_outfile<<this->gradCoefOutput.Ykeys[2*i]<<", ";
        reg_outfile<<this->gradCoefOutput.Ykeys[2*i + 1]<<", ";
    }
    reg_outfile<<this->gradCoefOutput.Ykeys[2*(nycoeffs-1)]<<", ";
    reg_outfile<<this->gradCoefOutput.Ykeys[2*(nycoeffs-1) + 1] << "]," << std::endl;

    /* Z gradients */
    reg_outfile << "\"ZCOEF:\"[";
    for (int i = 0; i < nzcoeffs-1; i ++){
        reg_outfile<<this->gradCoefOutput.gradZ_coef[i]<<", ";
    }
    reg_outfile<< this->gradCoefOutput.gradZ_coef[nzcoeffs-1] << "]," << std::endl;
    reg_outfile << "\"ZKEY:\"[";
    for (int i = 0; i < nzcoeffs-1; i ++){
        reg_outfile<<this->gradCoefOutput.Zkeys[2*i]<<", ";
        reg_outfile<<this->gradCoefOutput.Zkeys[2*i + 1]<<", ";
    }
    reg_outfile<<this->gradCoefOutput.Zkeys[2*(nzcoeffs-1)]<<", ";
    reg_outfile<<this->gradCoefOutput.Zkeys[2*(nzcoeffs-1) + 1] << "]," << std::endl;

    reg_outfile << "\"R0\": " << this->R0 << std::endl;
    reg_outfile.close();

//    return temp_reg_settings_filename;
}
