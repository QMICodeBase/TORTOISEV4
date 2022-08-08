#include "reorient_image_parser.h"
#include "itkSpatialOrientation.h"
#include "itkOrientImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
namespace fs = boost::filesystem;

#include <string>

typedef itk::Image<float,3> ImageType3D;


ImageType3D::Pointer MakeDetOne(ImageType3D::Pointer dwis,vnl_matrix_fixed<double,3,3> &trans_matrix,vnl_matrix_fixed<double,3,3> &output_dir , std::string input_orient,std::string output_orient)
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

    vnl_matrix_fixed<double,3,3> output_actual_dir(3,3);
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
        ImageType3D::DirectionType img_dir;
        img_dir.SetIdentity();
        img_dir(0,0)=new_dir(0,0);img_dir(0,1)=new_dir(0,1);img_dir(0,2)=new_dir(0,2);
        img_dir(1,0)=new_dir(1,0);img_dir(1,1)=new_dir(1,1);img_dir(1,2)=new_dir(1,2);
        img_dir(2,0)=new_dir(2,0);img_dir(2,1)=new_dir(2,1);img_dir(2,2)=new_dir(2,2);


        ImageType3D::IndexType new_orig_index;        
        new_orig_index.Fill(0);
        new_orig_index[1]=dwis->GetLargestPossibleRegion().GetSize()[1]-1;

        ImageType3D::PointType new_orig;
        dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);


        ImageType3D::Pointer new_str= ImageType3D::New();
        new_str->SetRegions(dwis->GetLargestPossibleRegion());
        new_str->Allocate();
        new_str->SetSpacing(dwis->GetSpacing());        
        new_str->SetDirection(img_dir);
        new_str->SetOrigin(new_orig);

        itk::ImageRegionIteratorWithIndex<ImageType3D> it(new_str,new_str->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            ImageType3D::IndexType index= it.GetIndex();
            index[1]= new_str->GetLargestPossibleRegion().GetSize()[1]-1-index[1];
            it.Set(dwis->GetPixel(index));
            ++it;
        }

        return new_str;
    }


    return dwis;
}



ImageType3D::Pointer ReorientDWIs(ImageType3D::Pointer dwis,vnl_matrix_fixed<double,3,3> trans_matrix, vnl_matrix_fixed<double,3,3> output_dir)
{
    ImageType3D::SpacingType new_spc;    
    ImageType3D::SizeType new_sizes;    
    ImageType3D::IndexType new_orig_index;
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

    ImageType3D::PointType new_orig;    
    dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);


    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,new_sizes);
    ImageType3D::DirectionType dir;    
    dir(0,0)=output_dir(0,0);dir(0,1)=output_dir(0,1);dir(0,2)=output_dir(0,2);
    dir(1,0)=output_dir(1,0);dir(1,1)=output_dir(1,1);dir(1,2)=output_dir(1,2);
    dir(2,0)=output_dir(2,0);dir(2,1)=output_dir(2,1);dir(2,2)=output_dir(2,2);

    ImageType3D::Pointer new_dwis=ImageType3D::New();
    new_dwis->SetRegions(reg);
    new_dwis->Allocate();
    new_dwis->SetSpacing(new_spc);
    new_dwis->SetDirection(dir);
    new_dwis->SetOrigin(new_orig);


    ImageType3D::IndexType new_orig2;
    new_orig2.Fill(0);
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


    itk::ImageRegionIteratorWithIndex<ImageType3D> it(dwis,dwis->GetLargestPossibleRegion());
    it.GoToBegin();
    vnl_vector<double> ind3(3);
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType old_index= it.GetIndex();
        ind3[0]=old_index[0];
        ind3[1]=old_index[1];
        ind3[2]=old_index[2];

        ImageType3D::IndexType new_index;
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



int main(int argc, char * argv[])
{
    Reorient_Image_PARSER *parser= new Reorient_Image_PARSER(argc,argv);

    std::string nifti_filename = parser->getInputImageName();

    typedef itk::ImageFileReader<ImageType3D> ReaderType;
    ReaderType::Pointer rd= ReaderType::New();
    rd->SetFileName(nifti_filename);
    rd->Update();
    ImageType3D::Pointer  dwis = rd->GetOutput();


    //ImageType3D::Pointer  dwis = readImageD<ImageType3D>(nifti_filename);


    vnl_matrix_fixed<double,3,3> trans_matrix, output_dir;
    std::string input_orient= parser->getOriginalOrientation();
    std::string output_orient= parser->getDesiredOrientation();

    dwis =  MakeDetOne(dwis,trans_matrix,output_dir ,  input_orient, output_orient);



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


      std::cout << " bb = {[" << bb0 << "], [" << bb1 << "]}; "<<std::endl;      



    ImageType3D::Pointer new_dwis = ReorientDWIs(dwis,trans_matrix,output_dir);



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

    //writeImageD<ImageType3D>(new_dwis,output_name);








    std::cout<<"Done reorienting the NIFTI file..."<<std::endl;

    return EXIT_SUCCESS;
}
