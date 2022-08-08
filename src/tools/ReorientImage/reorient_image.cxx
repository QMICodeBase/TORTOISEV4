#include "reorient_image_parser.h"
#include "defines.h"
#include "itkSpatialOrientation.h"
#include "itkOrientImageFilter.h"
#include "itkExtractImageFilter.h"
#include "../utilities/read_bmatrix_file.h"


#include <boost/algorithm/string.hpp>
#include <string>
#include "itkOkanQuadraticTransform.h"


typedef itk::OkanQuadraticTransform<double,3,3> OkanQuadraticTransformType;

ImageType4D::Pointer MakeDetOne(ImageType4D::Pointer dwis,vnl_matrix<double> & Bmatrix, vnl_matrix_fixed<double,3,3> &trans_matrix,vnl_matrix_fixed<double,3,3> &output_dir , std::string input_orient,std::string output_orient)
{

    vnl_matrix_fixed<double,3,3> input_actual_dir;
    if(input_orient=="")
    {
        input_actual_dir= dwis->GetDirection().GetVnlMatrix().extract(3,3);
        std::cout<<"Orig dir: "<< input_actual_dir<<std::endl;
    }
    else
    {
        boost::to_upper(input_orient);

        for(int dim=0;dim<3;dim++)
        {
            vnl_vector<double> vec(3);
            vec.fill(0);
            char f= input_orient.at(dim);
            if(f=='L')
                vec[0]=1;
            if(f=='R')
                vec[0]=-1;
            if(f=='P')
                vec[1]=1;
            if(f=='A')
                vec[1]=-1;
            if(f=='S')
                vec[2]=1;
            if(f=='I')
                vec[2]=-1;

            input_actual_dir.set_row(dim,vec);
        }
    }
    input_actual_dir= input_actual_dir.transpose();

    vnl_matrix_fixed<double,3,3> output_actual_dir;
    boost::to_upper(output_orient);
    for(int dim=0;dim<3;dim++)
    {
        vnl_vector<double> vec(3);
        vec.fill(0);
        char f= output_orient.at(dim);
        if(f=='L')
            vec[0]=1;
        if(f=='R')
            vec[0]=-1;
        if(f=='P')
            vec[1]=1;
        if(f=='A')
            vec[1]=-1;
        if(f=='S')
            vec[2]=1;
        if(f=='I')
            vec[2]=-1;

        output_actual_dir.set_row(dim,vec);

    }
    output_actual_dir= output_actual_dir.transpose();
    output_dir= output_actual_dir;

    vnl_matrix<double> aa=vnl_matrix_inverse<double>(input_actual_dir.transpose());

    trans_matrix= aa * output_actual_dir ;
    trans_matrix= trans_matrix.transpose();




    double mdet = vnl_determinant<double>(trans_matrix);
    if(fabs(mdet+1)<0.1)
    {
        vnl_matrix<double> flipM(3,3);
        flipM.set_identity();
        flipM(1,1)=-1;

        trans_matrix= trans_matrix *flipM;


        vnl_matrix<double> new_dir= input_actual_dir.transpose() * flipM;
        ImageType4D::DirectionType img_dir;
        img_dir.SetIdentity();
        img_dir(0,0)=new_dir(0,0);img_dir(0,1)=new_dir(0,1);img_dir(0,2)=new_dir(0,2);
        img_dir(1,0)=new_dir(1,0);img_dir(1,1)=new_dir(1,1);img_dir(1,2)=new_dir(1,2);
        img_dir(2,0)=new_dir(2,0);img_dir(2,1)=new_dir(2,1);img_dir(2,2)=new_dir(2,2);


        ImageType4D::IndexType new_orig_index;
        new_orig_index.Fill(0);
        new_orig_index[1]=dwis->GetLargestPossibleRegion().GetSize()[1]-1;


        ImageType4D::PointType new_orig;
        dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);



        ImageType4D::Pointer new_str= ImageType4D::New();
        new_str->SetRegions(dwis->GetLargestPossibleRegion());
        new_str->Allocate();
        new_str->SetSpacing(dwis->GetSpacing());
        new_str->SetDirection(dwis->GetDirection());
        new_str->SetOrigin(dwis->GetOrigin());

        itk::ImageRegionIteratorWithIndex<ImageType4D> it(new_str,new_str->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            ImageType4D::IndexType index= it.GetIndex();
            index[1]= new_str->GetLargestPossibleRegion().GetSize()[1]-1-index[1];
            it.Set(dwis->GetPixel(index));
            ++it;
        }


        Bmatrix.set_column(1, -1.*Bmatrix.get_column(1));
        Bmatrix.set_column(4, -1.*Bmatrix.get_column(4));

        return new_str;
    }


    return dwis;
}


ImageType4D::Pointer ReorientDWIs(ImageType4D::Pointer dwis,vnl_matrix<double> trans_matrix, vnl_matrix_fixed<double,3,3> output_dir)
{
    ImageType4D::SpacingType new_spc;
    new_spc[3]=1;

    ImageType4D::SizeType new_sizes;
    new_sizes[3]= dwis->GetLargestPossibleRegion().GetSize()[3];

    ImageType4D::IndexType new_orig_index;
    new_orig_index.Fill(0);

    for(int d=0;d<3;d++)
    {
        vnl_vector<double> row = trans_matrix.get_row(d);
        new_spc[d]= fabs(row[0]*dwis->GetSpacing()[0] +row[1]*dwis->GetSpacing()[1]+row[2]*dwis->GetSpacing()[2]);
        new_sizes[d] = (int)(round(fabs(row[0]*dwis->GetLargestPossibleRegion().GetSize()[0] +row[1]*dwis->GetLargestPossibleRegion().GetSize()[1]+row[2]*dwis->GetLargestPossibleRegion().GetSize()[2])));

        if(row[0]==-1)
        {
            new_orig_index[0]=dwis->GetLargestPossibleRegion().GetSize()[0]-1;
        }
        if(row[1]==-1)
        {
            new_orig_index[1]=dwis->GetLargestPossibleRegion().GetSize()[1]-1;
        }
        if(row[2]==-1)
        {
            new_orig_index[2]=dwis->GetLargestPossibleRegion().GetSize()[2]-1;
        }
    }

    ImageType4D::PointType new_orig;    
    dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);    


    ImageType4D::IndexType start; start.Fill(0);
    ImageType4D::RegionType reg(start,new_sizes);
    ImageType4D::DirectionType dir;
    dir.SetIdentity();
    dir(0,0)= output_dir(0,0);dir(0,1)= output_dir(0,1);dir(0,2)= output_dir(0,2);
    dir(1,0)= output_dir(1,0);dir(1,1)= output_dir(1,1);dir(1,2)= output_dir(1,2);
    dir(2,0)= output_dir(2,0);dir(2,1)= output_dir(2,1);dir(2,2)= output_dir(2,2);

    ImageType4D::Pointer new_dwis=ImageType4D::New();
    new_dwis->SetRegions(reg);
    new_dwis->Allocate();
    new_dwis->SetSpacing(new_spc);


    vnl_vector<double> new_orig2(3);
    new_orig2.fill(0);
    for(int d=0;d<3;d++)
    {
        vnl_vector<double>  row= trans_matrix.get_row(d);
        if(row[0]==-1)
            new_orig2[d]=dwis->GetLargestPossibleRegion().GetSize()[0]-1;
        if(row[1]==-1)
            new_orig2[d]=dwis->GetLargestPossibleRegion().GetSize()[1]-1;
        if(row[2]==-1)
            new_orig2[d]=dwis->GetLargestPossibleRegion().GetSize()[2]-1;
    }


    itk::ImageRegionIteratorWithIndex<ImageType4D> it(dwis,dwis->GetLargestPossibleRegion());
    it.GoToBegin();
    vnl_vector<double> ind3(3);
    while(!it.IsAtEnd())
    {
        ImageType4D::IndexType old_index= it.GetIndex();
        ind3[0]=old_index[0];
        ind3[1]=old_index[1];
        ind3[2]=old_index[2];

        ImageType4D::IndexType new_index;
        vnl_vector<double>  vec= trans_matrix * ind3;


        new_index[0]= new_orig2[0]+ vec[0];
        new_index[1]= new_orig2[1]+ vec[1];
        new_index[2]= new_orig2[2]+ vec[2];
        new_index[3]= old_index[3];

        new_dwis->SetPixel(new_index,it.Get());
        ++it;
    }


    return new_dwis;
}

vnl_matrix<double>  RotateBmatrix(vnl_matrix<double>Bmatrix,vnl_matrix_fixed<double,3,3> trans_matrix)
{
    int Nvols= Bmatrix.rows();

    vnl_matrix<double> new_Bmatrix(Nvols,6);

    vnl_matrix<double> B= trans_matrix.transpose();

    double ax,ay,az;
    if(B(2,0)<1)
    {
        if(B(2,0)>-1)
        {
            ay= asin(-B(2,0));
            az=atan2(B(1,0),B(0,0));
            ax= atan2(B(2,1),B(2,2));
        }
        else
        {
            ay= 1.5707964;
            az= -atan2(-B(1,2),B(1,1));
            ax= 0;
        }
    }
    else
    {
        ay= -1.5707964;
        az= atan2(-B(1,2),B(1,1));
        ax= 0;
    }


    OkanQuadraticTransformType::Pointer finalTransform= OkanQuadraticTransformType::New();
    finalTransform->SetPhase(1);
    finalTransform->SetIdentity();
    OkanQuadraticTransformType::ParametersType finalparams= finalTransform->GetParameters();
    finalparams[0]=0;
    finalparams[1]=0;
    finalparams[2]=0;
    finalparams[3]=-ax;
    finalparams[4]=-ay;
    finalparams[5]=az;
    finalTransform->SetParameters(finalparams);

    vnl_matrix<double> rot_mat= finalTransform->GetMatrix().GetTranspose();

    for(int vol=0;vol<Nvols;vol++)
    {
        vnl_vector<double> Bmatrixvec=Bmatrix.get_row(vol);
        vnl_matrix<double> curr_Bmat(3,3);

        curr_Bmat(0,0)=Bmatrixvec[0];
        curr_Bmat(0,1)=Bmatrixvec[1]/2.;
        curr_Bmat(1,0)=Bmatrixvec[1]/2.;
        curr_Bmat(0,2)=Bmatrixvec[2]/2.;
        curr_Bmat(2,0)=Bmatrixvec[2]/2.;
        curr_Bmat(1,1)=Bmatrixvec[3];
        curr_Bmat(2,1)=Bmatrixvec[4]/2.;
        curr_Bmat(1,2)=Bmatrixvec[4]/2.;
        curr_Bmat(2,2)=Bmatrixvec[5];


        vnl_matrix<double> rotated_curr_Bmat = rot_mat  * curr_Bmat  * rot_mat.transpose();

        vnl_vector<double> rot_bmat_vec(6);
        rot_bmat_vec[0]= rotated_curr_Bmat(0,0);
        rot_bmat_vec[1]= 2*rotated_curr_Bmat(0,1);
        rot_bmat_vec[2]= 2*rotated_curr_Bmat(0,2);
        rot_bmat_vec[3]= rotated_curr_Bmat(1,1);
        rot_bmat_vec[4]= 2*rotated_curr_Bmat(1,2);
        rot_bmat_vec[5]= rotated_curr_Bmat(2,2);

        new_Bmatrix.set_row(vol,rot_bmat_vec);
    }

    return new_Bmatrix;


}


int main(int argc, char * argv[])
{
    Reorient_Image_PARSER *parser= new Reorient_Image_PARSER(argc,argv);

    std::string nii_name=parser->getInputImageName();
    std::string bmtxt_name = nii_name.substr(0,nii_name.rfind(".nii"))+".bmtxt";
    std::string json_name = nii_name.substr(0,nii_name.rfind(".nii"))+".json";

    ImageType4D::Pointer  dwis = readImageD<ImageType4D>(nii_name);
    vnl_matrix<double>  Bmatrix= read_bmatrix_file(bmtxt_name);


    vnl_matrix_fixed<double,3,3> trans_matrix, output_dir;
    std::string input_orient= parser->getOriginalOrientation();
    std::string output_orient= parser->getDesiredOrientation();
    

    dwis =  MakeDetOne(dwis,Bmatrix, trans_matrix,output_dir ,  input_orient, output_orient);


    vnl_vector<double> bb0, bb1;
    bb0.set_size(4);
    bb1.set_size(4);
    itk::ContinuousIndex<double,4> bound_start_index, bound_end_index;
    typename ImageType4D::PointType bound_start_point,bound_end_point;



    bound_start_index[0]=-0.5;
    bound_start_index[1]=-0.5;
    bound_start_index[2]=-0.5;

    bound_end_index[0]=dwis->GetLargestPossibleRegion().GetSize()[0] + 0.5;
    bound_end_index[1]=dwis->GetLargestPossibleRegion().GetSize()[1] + 0.5;
    bound_end_index[2]=dwis->GetLargestPossibleRegion().GetSize()[2] + 0.5;

    dwis->TransformContinuousIndexToPhysicalPoint(bound_start_index,bound_start_point);
    dwis->TransformContinuousIndexToPhysicalPoint(bound_end_index,bound_end_point);

    bb0[0]= bound_start_point[0];
    bb0[1]= bound_start_point[1];
    bb0[2]= bound_start_point[2];

    bb1[0]= bound_end_point[0];
    bb1[1]= bound_end_point[1];
    bb1[2]= bound_end_point[2];


    std::cout << " bb = {[" << bb0 << "], [" << bb1 << "]}; "<<std::endl;

    ImageType4D::Pointer new_dwis = ReorientDWIs(dwis,trans_matrix,output_dir);

    std::string phase;
    json mjson;
    if(fs::exists(json_name))
    {

        std::ifstream json_stream(json_name);
        json_stream >> mjson;
        json_stream.close();

        std::string json_PE= mjson["PhaseEncodingDirection"];      //get phase encoding direction
        if(json_PE.find("j")!=std::string::npos)
            phase="vertical";
        else
            if(json_PE.find("i")!=std::string::npos)
                phase="horizontal";
            else
                phase="slice";
    }
    if(parser->getPhase()!="")
        phase=parser->getPhase();


    vnl_vector<double> curr_phase(3);
    curr_phase.fill(0);
    if(phase=="horizontal")
        curr_phase[0]=1;
    if(phase=="vertical")
        curr_phase[1]=1;
    if(phase=="slice")
        curr_phase[2]=1;
    vnl_vector<double> new_phase= trans_matrix * curr_phase;
        

    if(fabs(new_phase[0])>0.9)
        phase=std::string("horizontal");
    if(fabs(new_phase[1])>0.9)
        phase=std::string("vertical");
    if(fabs(new_phase[2])>0.9)
        phase=std::string("slice");

    vnl_matrix<double> new_Bmatrix= RotateBmatrix(Bmatrix,trans_matrix);


    if(phase=="horizontal")
       mjson["PhaseEncodingDirection"]="i+";
    if(phase=="vertical")
       mjson["PhaseEncodingDirection"]="j+";
    if(phase=="slice")
       mjson["PhaseEncodingDirection"]="k+";


    std::string new_nii_name;
    std::string new_bmtxt_name;
    std::string new_json_name;

    std::string output_name = parser->getOutputName();
    if(output_name=="")
    {
        new_nii_name=  nii_name.substr(0,nii_name.rfind(".nii"))+"_reoriented.nii";
        new_bmtxt_name=  nii_name.substr(0,nii_name.rfind(".nii"))+"_reoriented.bmtxt";
        new_json_name=  nii_name.substr(0,nii_name.rfind(".nii"))+"_reoriented.json";
    }
    else
    {
        new_nii_name=output_name;
        new_bmtxt_name= new_nii_name.substr(0,new_nii_name.rfind(".nii"))+".bmtxt";
        new_json_name= new_nii_name.substr(0,new_nii_name.rfind(".nii"))+".json";
    }


    writeImageD<ImageType4D>(new_dwis,new_nii_name);


    std::ofstream out_bmtxt(new_bmtxt_name);
    out_bmtxt  << new_Bmatrix << std::endl;
    out_bmtxt.close();

    std::ofstream out_json(new_json_name);
    out_json << std::setw(4) << mjson << std::endl;
    out_json.close();

    std::cout<<"Done reorienting DWIs..."<<std::endl;

    return EXIT_SUCCESS;
}
