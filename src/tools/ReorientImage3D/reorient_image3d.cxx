#include "reorient_image_parser.h"
#include "itkSpatialOrientation.h"
#include "itkOrientImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"


#include "../utilities/math_utilities.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkIdentityTransform.h"


#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
namespace fs = boost::filesystem;

#include <string>

typedef itk::Image<float,3> ImageType3D;

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


std::string DirectionToOrient(vnl_matrix_fixed<double,3,3> dir2)
{
    vnl_matrix_fixed<double,3,3> dir = dir2;
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

ImageType3D::Pointer  ReorientDWIs(ImageType3D::Pointer dwis, vnl_vector_fixed<int,3> permutes)
{
    vnl_matrix_fixed<double,3,3> new_dir;
    ImageType3D::SpacingType new_spc;
    ImageType3D::PointType new_orig;
    ImageType3D::SizeType new_size;
    ImageType3D::IndexType new_orig_index;

    vnl_matrix_fixed<double,3,3> orig_dir = dwis->GetDirection().GetVnlMatrix();

    for(int d=0;d<3;d++)
    {
        new_size[d] =dwis->GetLargestPossibleRegion().GetSize()[abs(permutes[d])-1];
        new_spc[d] =dwis->GetSpacing()[abs(permutes[d])-1];

        if(permutes[d]>0)
            new_orig_index[abs(permutes[d])-1]=0;
        else
            new_orig_index[abs(permutes[d])-1] =  dwis->GetLargestPossibleRegion().GetSize()[abs(permutes[d])-1] -1;


        new_dir.set_column(d,  (double)(sgn<int>(permutes[d])) * orig_dir.get_column(abs(permutes[d])-1));
    }

    dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);




    using InterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType3D>;
    using ResamplerType = itk::ResampleImageFilter<ImageType3D,ImageType3D>;
    using TransformType = itk::IdentityTransform<double,3>;
    InterpolatorType::Pointer interp = InterpolatorType::New();
    interp->SetInputImage(dwis);

    TransformType::Pointer id_trans=TransformType::New();
    id_trans->SetIdentity();


        ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,new_size);

    ImageType3D::Pointer new_dwis = ImageType3D::New();
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


int main(int argc, char * argv[])
{
    Reorient_Image_PARSER *parser= new Reorient_Image_PARSER(argc,argv);

    std::string nifti_filename = parser->getInputImageName();

    typedef itk::ImageFileReader<ImageType3D> ReaderType;
    ReaderType::Pointer rd= ReaderType::New();
    rd->SetFileName(nifti_filename);
    rd->Update();
    ImageType3D::Pointer  dwis = rd->GetOutput();


    vnl_matrix_fixed<double,3,3> trans_matrix, output_dir;
    std::string input_orient= parser->getOriginalOrientation();
    std::string output_orient= parser->getDesiredOrientation();

    if(input_orient=="")
        input_orient = DirectionToOrient(dwis->GetDirection().GetVnlMatrix());
    if(output_orient=="")
    {
        if(parser->getDesiredOrientationFromReferenceImage()!="")
        {
            ReaderType::Pointer rd2= ReaderType::New();
            rd2->SetFileName(parser->getDesiredOrientationFromReferenceImage());
            rd2->Update();
            ImageType3D::Pointer  ref_img = rd2->GetOutput();

            output_orient= DirectionToOrient(ref_img->GetDirection().GetVnlMatrix());
        }
        else
            output_orient="LPS";
    }


    vnl_vector_fixed<int,3> permutes = GetPermutation(input_orient,output_orient);

    ImageType3D::Pointer new_dwis = ReorientDWIs(dwis,permutes);



    {
        vnl_vector<double> bb0, bb1;
        bb0.set_size(3);
        bb1.set_size(3);
        itk::ContinuousIndex<double,3> bound_start_index, bound_end_index;
        typename ImageType3D::PointType bound_start_point,bound_end_point;


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
        itk::ContinuousIndex<double,3> bound_start_index, bound_end_index;
        typename ImageType3D::PointType bound_start_point,bound_end_point;


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







    std::string output_name = parser->getOutputName();
    if(output_name=="")
    {
        fs::path NIFTI_path(parser->getInputImageName());
        fs::path proc_folder = NIFTI_path.parent_path();
        fs::path output_proc_path;
        fs::path basename = NIFTI_path.stem();
        fs::path new_listname= proc_folder/ (basename.string() + std::string("_reoriented.nii")) ;

        output_name = new_listname.string();
    }

    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr=WrType::New();
    wr->SetInput(new_dwis);
    wr->SetFileName(output_name);
    wr->Update();




    std::cout<<"Done reorienting the NIFTI file..."<<std::endl;

    return EXIT_SUCCESS;
}
