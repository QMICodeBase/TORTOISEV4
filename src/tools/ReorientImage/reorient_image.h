#ifndef _REORIENTIMAGE_H
#define _REORIENTIMAGE_H

#include "defines.h"


ImageType3D::Pointer ReorientImg(ImageType3D::Pointer dwis,vnl_matrix_fixed<double,3,3> trans_matrix, vnl_matrix_fixed<double,3,3> output_dir)
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


ImageType3D::Pointer ReorientImage3D(ImageType3D::Pointer img, std::string input_orientation_string, std::string output_orientation_string)
{

    vnl_matrix_fixed<double,3,3> input_actual_dir;
    if(input_orientation_string=="")
    {
        input_actual_dir= img->GetDirection().GetVnlMatrix().extract(3,3);
    }
    else
    {
        boost::to_upper(input_orientation_string);
        for(int dim=0;dim<3;dim++)
        {
            vnl_vector<double> vec(3);
            vec.fill(0);
            char f= input_orientation_string.at(dim);
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
    boost::to_upper(output_orientation_string);
    for(int dim=0;dim<3;dim++)
    {
        vnl_vector<double> vec(3);
        vec.fill(0);
        char f= output_orientation_string.at(dim);
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
    vnl_matrix_fixed<double,3,3> output_dir= output_actual_dir;

    if(  (output_dir - input_actual_dir).absolute_value_sum() <1E-6)
        return img;



    vnl_matrix<double> aa=vnl_matrix_inverse<double>(input_actual_dir.transpose());
    vnl_matrix_fixed<double,3,3> trans_matrix= aa * output_actual_dir ;
    trans_matrix= trans_matrix.transpose();

    ImageType3D::Pointer new_dwis = ReorientImg(img,trans_matrix,output_dir);
    return new_dwis;
}


#endif

