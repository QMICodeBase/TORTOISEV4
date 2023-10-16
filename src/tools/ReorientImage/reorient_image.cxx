#include "reorient_image_parser.h"
#include "defines.h"
#include "itkSpatialOrientation.h"
#include "itkOrientImageFilter.h"
#include "itkExtractImageFilter.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/math_utilities.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkIdentityTransform.h"


#include <boost/algorithm/string.hpp>
#include <string>




vnl_vector_fixed<int,3>  GetPermutation(std::string input_orient,std::string output_orient)
{
    boost::to_upper(input_orient);
    boost::to_upper(output_orient);

    vnl_vector_fixed<int,3> permutes;

    for(int dim=0;dim<3;dim++)
    {
        char fo= output_orient.at(dim);

        if(fo=='L')
        {
            if(input_orient.at(0)=='L')
                permutes[dim]=1;
            if(input_orient.at(0)=='R')
                permutes[dim]=-1;
            if(input_orient.at(1)=='L')
                permutes[dim]=2;
            if(input_orient.at(1)=='R')
                permutes[dim]=-2;
            if(input_orient.at(2)=='L')
                permutes[dim]=3;
            if(input_orient.at(2)=='R')
                permutes[dim]=-3;
        }
        if(fo=='R')
        {
            if(input_orient.at(0)=='R')
                permutes[dim]=1;
            if(input_orient.at(0)=='L')
                permutes[dim]=-1;
            if(input_orient.at(1)=='R')
                permutes[dim]=2;
            if(input_orient.at(1)=='L')
                permutes[dim]=-2;
            if(input_orient.at(2)=='R')
                permutes[dim]=3;
            if(input_orient.at(2)=='L')
                permutes[dim]=-3;
        }
        if(fo=='I')
        {
            if(input_orient.at(0)=='I')
                permutes[dim]=1;
            if(input_orient.at(0)=='S')
                permutes[dim]=-1;
            if(input_orient.at(1)=='I')
                permutes[dim]=2;
            if(input_orient.at(1)=='S')
                permutes[dim]=-2;
            if(input_orient.at(2)=='I')
                permutes[dim]=3;
            if(input_orient.at(2)=='S')
                permutes[dim]=-3;
        }
        if(fo=='S')
        {
            if(input_orient.at(0)=='S')
                permutes[dim]=1;
            if(input_orient.at(0)=='I')
                permutes[dim]=-1;
            if(input_orient.at(1)=='S')
                permutes[dim]=2;
            if(input_orient.at(1)=='I')
                permutes[dim]=-2;
            if(input_orient.at(2)=='S')
                permutes[dim]=3;
            if(input_orient.at(2)=='I')
                permutes[dim]=-3;
        }
        if(fo=='A')
        {
            if(input_orient.at(0)=='A')
                permutes[dim]=1;
            if(input_orient.at(0)=='P')
                permutes[dim]=-1;
            if(input_orient.at(1)=='A')
                permutes[dim]=2;
            if(input_orient.at(1)=='P')
                permutes[dim]=-2;
            if(input_orient.at(2)=='A')
                permutes[dim]=3;
            if(input_orient.at(2)=='P')
                permutes[dim]=-3;
        }
        if(fo=='P')
        {
            if(input_orient.at(0)=='P')
                permutes[dim]=1;
            if(input_orient.at(0)=='A')
                permutes[dim]=-1;
            if(input_orient.at(1)=='P')
                permutes[dim]=2;
            if(input_orient.at(1)=='A')
                permutes[dim]=-2;
            if(input_orient.at(2)=='P')
                permutes[dim]=3;
            if(input_orient.at(2)=='A')
                permutes[dim]=-3;
        }
    }
    return permutes;
}


std::string DirectionToOrient(vnl_matrix_fixed<double,4,4> dir2)
{
    vnl_matrix_fixed<double,4,4> dir = dir2;
    for(int r=0;r<3;r++)
    {
        for(int c=0;c<3;c++)
        {
            if(fabs(dir(r,c))>0.8)
            {
                dir(r,c) = sgn<double>(dir(r,c));
            }
            else
            {
                if(fabs(dir(r,c))<0.2)
                {
                    dir(r,c)=0;
                }
                else
                {
                    std::cout<<"Image too oblique to determine orientation. Exiting..."<<std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    std::string orient;
    orient.resize(3);

    for(int d=0;d<3;d++)
    {
        vnl_vector<double> row = dir.get_column(d);
        if(row[0]<0)
            orient.at(d)='R';
        if(row[0]>0)
            orient.at(d)='L';
        if(row[1]<0)
            orient.at(d)='A';
        if(row[1]>0)
            orient.at(d)='P';
        if(row[2]<0)
            orient.at(d)='I';
        if(row[2]>0)
            orient.at(d)='S';
    }

    return orient;


}





ImageType4D::Pointer  ReorientDWIs(ImageType4D::Pointer dwis, vnl_vector_fixed<int,3> permutes)
{
    vnl_matrix_fixed<double,4,4> new_dir; new_dir.set_identity();
    ImageType4D::SpacingType new_spc; new_spc.Fill(1);
    ImageType4D::PointType new_orig; new_orig.Fill(0);
    ImageType4D::SizeType new_size;
    ImageType4D::IndexType new_orig_index; new_orig_index.Fill(0);

    new_size[3] = dwis->GetLargestPossibleRegion().GetSize()[3];

    vnl_matrix_fixed<double,4,4> orig_dir = dwis->GetDirection().GetVnlMatrix();

    for(int d=0;d<3;d++)
    {
        new_size[d] =dwis->GetLargestPossibleRegion().GetSize()[abs(permutes[d])-1];
        new_spc[d] =dwis->GetSpacing()[abs(permutes[d])-1];

        if(permutes[d]>0)
            new_orig_index[abs(permutes[d])-1]=0;
        else
            new_orig_index[abs(permutes[d])-1] =  dwis->GetLargestPossibleRegion().GetSize()[abs(permutes[d])-1] -1;


        auto aa= (double)(sgn<int>(permutes[d])) * orig_dir.get_column(abs(permutes[d])-1);
        new_dir(0,d)= aa[0];
        new_dir(1,d)= aa[1];
        new_dir(2,d)= aa[2];

    }

    dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);




    using InterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType4D>;
    using ResamplerType = itk::ResampleImageFilter<ImageType4D,ImageType4D>;
    using TransformType = itk::IdentityTransform<double,4>;
    InterpolatorType::Pointer interp = InterpolatorType::New();
    interp->SetInputImage(dwis);

    TransformType::Pointer id_trans=TransformType::New();
    id_trans->SetIdentity();


    ImageType4D::IndexType start; start.Fill(0);
    ImageType4D::RegionType reg(start,new_size);

    ImageType4D::Pointer new_dwis = ImageType4D::New();
    new_dwis->SetRegions(reg);
    new_dwis->SetDirection(new_dir);
    new_dwis->SetOrigin(new_orig);
    new_dwis->SetSpacing(new_spc);




    ResamplerType::Pointer resampler = ResamplerType::New();
    resampler->SetInput(dwis);
    resampler->SetInterpolator(interp);
    resampler->SetTransform(id_trans);
    resampler->SetOutputParametersFromImage(new_dwis);
    resampler->Update();
    return resampler->GetOutput();

}




vnl_matrix<double>  RotateBmatrix(vnl_matrix<double>Bmatrix,vnl_vector_fixed<int,3>  permutes)
{
    vnl_matrix<double> new_Bmatrix=Bmatrix;

    for(int v=0;v<Bmatrix.rows();v++)
    {
        vnl_vector<double> bmat_vec= Bmatrix.get_row(v);
        vnl_vector<double> new_bmat_vec= bmat_vec;

        {
            if(abs(permutes[0])==1)
                new_bmat_vec[0]= bmat_vec[0];
            if(abs(permutes[0])==2)
                new_bmat_vec[0]= bmat_vec[3];
            if(abs(permutes[0])==3)
                new_bmat_vec[0]= bmat_vec[5];
        }
        {
            if(abs(permutes[1])==1)
                new_bmat_vec[3]= bmat_vec[0];
            if(abs(permutes[1])==2)
                new_bmat_vec[3]= bmat_vec[3];
            if(abs(permutes[1])==3)
                new_bmat_vec[3]= bmat_vec[5];
        }
        {
            if(abs(permutes[2])==1)
                new_bmat_vec[5]= bmat_vec[0];
            if(abs(permutes[2])==2)
                new_bmat_vec[5]= bmat_vec[3];
            if(abs(permutes[2])==3)
                new_bmat_vec[5]= bmat_vec[5];
        }

        {
            if(abs(permutes[0])==1   &&  abs(permutes[1])==2 )
            {
                new_bmat_vec[1] =  bmat_vec[1] * sgn(permutes[0]) * sgn(permutes[1]);
            }
            if(abs(permutes[0])==1   &&  abs(permutes[1])==3 )
            {
                new_bmat_vec[1] =  bmat_vec[2] * sgn(permutes[0]) * sgn(permutes[1]);
            }

            if(abs(permutes[0])==2   &&  abs(permutes[1])==1 )
            {
                new_bmat_vec[1] =  bmat_vec[1] * sgn(permutes[0]) * sgn(permutes[1]);
            }
            if(abs(permutes[0])==2   &&  abs(permutes[1])==3 )
            {
                new_bmat_vec[1] =  bmat_vec[4] * sgn(permutes[0]) * sgn(permutes[1]);
            }

            if(abs(permutes[0])==3   &&  abs(permutes[1])==1 )
            {
                new_bmat_vec[1] =  bmat_vec[2] * sgn(permutes[0]) * sgn(permutes[1]);
            }
            if(abs(permutes[0])==3   &&  abs(permutes[1])==2 )
            {
                new_bmat_vec[1] =  bmat_vec[4] * sgn(permutes[0]) * sgn(permutes[1]);
            }
        }



        {
            if(abs(permutes[0])==1   &&  abs(permutes[2])==2 )
            {
                new_bmat_vec[2] =  bmat_vec[1] * sgn(permutes[0]) * sgn(permutes[2]);
            }
            if(abs(permutes[0])==1   &&  abs(permutes[2])==3 )
            {
                new_bmat_vec[2] =  bmat_vec[2] * sgn(permutes[0]) * sgn(permutes[2]);
            }

            if(abs(permutes[0])==2   &&  abs(permutes[2])==1 )
            {
                new_bmat_vec[2] =  bmat_vec[1] * sgn(permutes[0]) * sgn(permutes[2]);
            }
            if(abs(permutes[0])==2   &&  abs(permutes[2])==3 )
            {
                new_bmat_vec[2] =  bmat_vec[4] * sgn(permutes[0]) * sgn(permutes[2]);
            }

            if(abs(permutes[0])==3   &&  abs(permutes[2])==1 )
            {
                new_bmat_vec[2] =  bmat_vec[2] * sgn(permutes[0]) * sgn(permutes[2]);
            }
            if(abs(permutes[0])==3   &&  abs(permutes[2])==2 )
            {
                new_bmat_vec[2] =  bmat_vec[4] * sgn(permutes[0]) * sgn(permutes[2]);
            }
        }



        {
            if(abs(permutes[1])==1   &&  abs(permutes[2])==2 )
            {
                new_bmat_vec[4] =  bmat_vec[1] * sgn(permutes[1]) * sgn(permutes[2]);
            }
            if(abs(permutes[1])==1   &&  abs(permutes[2])==3 )
            {
                new_bmat_vec[4] =  bmat_vec[2] * sgn(permutes[1]) * sgn(permutes[2]);
            }

            if(abs(permutes[1])==2   &&  abs(permutes[2])==1 )
            {
                new_bmat_vec[4] =  bmat_vec[1] * sgn(permutes[1]) * sgn(permutes[2]);
            }
            if(abs(permutes[1])==2   &&  abs(permutes[2])==3 )
            {
                new_bmat_vec[4] =  bmat_vec[4] * sgn(permutes[1]) * sgn(permutes[2]);
            }

            if(abs(permutes[1])==3   &&  abs(permutes[2])==1 )
            {
                new_bmat_vec[4] =  bmat_vec[2] * sgn(permutes[1]) * sgn(permutes[2]);
            }
            if(abs(permutes[1])==3   &&  abs(permutes[2])==2 )
            {
                new_bmat_vec[4] =  bmat_vec[4] * sgn(permutes[1]) * sgn(permutes[2]);
            }
        }

        new_Bmatrix.set_row(v,new_bmat_vec);
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

    if(input_orient=="")
        input_orient = DirectionToOrient(dwis->GetDirection().GetVnlMatrix());
    if(output_orient=="")
        output_orient="LPS";





    vnl_vector_fixed<int,3> permutes = GetPermutation(input_orient,output_orient);

    ImageType4D::Pointer new_dwis = ReorientDWIs(dwis,permutes);





    {
        vnl_vector<double> bb0, bb1;
        bb0.set_size(3);
        bb1.set_size(3);
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


          std::cout << "  Orig bb = {[" << bb0 << "], [" << bb1 << "]}; "<<std::endl;
    }
    {
        vnl_vector<double> bb0, bb1;
        bb0.set_size(3);
        bb1.set_size(3);
        itk::ContinuousIndex<double,4> bound_start_index, bound_end_index;
        typename ImageType4D::PointType bound_start_point,bound_end_point;


        bound_start_index[0]=-0.5;
        bound_start_index[1]=-0.5;
        bound_start_index[2]=-0.5;

        bound_end_index[0]=new_dwis->GetLargestPossibleRegion().GetSize()[0] + 0.5;
        bound_end_index[1]=new_dwis->GetLargestPossibleRegion().GetSize()[1] + 0.5;
        bound_end_index[2]=new_dwis->GetLargestPossibleRegion().GetSize()[2] + 0.5;

        new_dwis->TransformContinuousIndexToPhysicalPoint(bound_start_index,bound_start_point);
        new_dwis->TransformContinuousIndexToPhysicalPoint(bound_end_index,bound_end_point);

          bb0[0]= bound_start_point[0];
          bb0[1]= bound_start_point[1];
          bb0[2]= bound_start_point[2];

          bb1[0]= bound_end_point[0];
          bb1[1]= bound_end_point[1];
          bb1[2]= bound_end_point[2];


          std::cout << "  New bb = {[" << bb0 << "], [" << bb1 << "]}; "<<std::endl;
    }


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


    if(phase=="vertical")
    {
        if(fabs(permutes[1])==1)
            phase="horizontal";
        else if(fabs(permutes[1])==2)
            phase="vertical";
        else if(fabs(permutes[1])==3)
            phase="slice";
    }
    if(phase=="horizontal")
    {
        if(fabs(permutes[0])==1)
            phase="horizontal";
        else if(fabs(permutes[0])==2)
            phase="vertical";
        else if(fabs(permutes[0])==3)
            phase="slice";
    }
    if(phase=="slice")
    {
        if(fabs(permutes[0])==1)
            phase="horizontal";
        else if(fabs(permutes[0])==2)
            phase="vertical";
        else if(fabs(permutes[0])==3)
            phase="slice";
    }


    vnl_matrix<double> new_Bmatrix= RotateBmatrix(Bmatrix,permutes);


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
