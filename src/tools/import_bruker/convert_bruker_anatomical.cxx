#include "convert_bruker_anatomical.h"
#include "combine_listfiles.h"
#include "write_bmatrix_file.h"
#include "TORTOISE_Utilities.h"
#include "read_3Dvolume_from_4D.h"

#include "itkPermuteAxesImageFilter.h"

#include <sstream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <boost/endian/conversion.hpp>

#include "itkImportImageFilter.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>



ConvertBrukerAnatomical::ConvertBrukerAnatomical( int argc , char * argv[] )
{
    parser= new BrukerAnatomical_PARSER(argc,argv);


    grad_orient.fill(0);

    slice_orientation=std::string("");
    read_orientation=std::string("");

    std::string input_folder = parser->getInputDataFolder();
    if(input_folder.at(input_folder.length()-1)==boost::filesystem::path::preferred_separator)
        input_folder= input_folder.substr(0,input_folder.length()-1);

    fs::path data_dir(input_folder);

    std::string output_filename = parser->getOutputFilename();


    if(output_filename=="")
    {
        fs::path input_folder_path(parser->getInputDataFolder());
        output_filename=(input_folder_path.parent_path() / input_folder_path.filename()).string() +std::string(".nii");
    }
    else
    {
        fs::path output_filename_path(output_filename);
        if(!fs::exists(output_filename_path.parent_path()))
        {
            fs::create_directories(output_filename_path.parent_path());
        }
    }



    if(fs::exists(data_dir / std::string("pdata")))
    {
        bool error_code = this->convert_bruker_anatomical(input_folder,output_filename );

        if(error_code==0)
        {
            std::cout<<"ConvertBruker with folder " << input_folder<< " failed. Exiting!!"<<std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        fs::directory_iterator it(data_dir);
        fs::directory_iterator endit;

        while(it != endit)
        {
            if(fs::is_directory(*it))
            {
                fs::path curr_input_folder= it->path();
                output_filename=(curr_input_folder.parent_path() / curr_input_folder.filename()).string() +std::string(".nii");

                bool error_code = this->convert_bruker_anatomical(curr_input_folder.string(),output_filename);
                if(error_code==0)
                {
                    std::cout<<"ConvertBruker with folder " << curr_input_folder<< " failed."<<std::endl;
                }
            }
            ++it;
        }
    }
} 




ConvertBrukerAnatomical::~ConvertBrukerAnatomical()
{
}


void ConvertBrukerAnatomical::get_grad_orient(std::string method_file)
{

    std::ifstream infile(method_file.c_str());
    std::string line;
    while (std::getline(infile, line))
    {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace),line.end());
        if(line.find("##$")==0)
        {
            int temp_pos1= line.find("$")+1;
            int temp_pos2= line.find("=");

            std::string var_name= line.substr(temp_pos1, temp_pos2-temp_pos1);
            std::transform(var_name.begin(), var_name.end(),var_name.begin(), ::toupper);

            if(var_name==std::string("PVM_RGVALUE"))
            {
                RG=atof(line.substr(temp_pos2+1).c_str());
            }

            if(var_name=="PVM_SPACKARRSLICEORIENT")
            {
                infile >> this->slice_orientation;
            }

            if(var_name=="PVM_SPACKARRREADORIENT")
            {
                infile >> this->read_orientation;
            }
            if(var_name=="PVM_SPACKARRGRADORIENT")
            {
                infile >> this->grad_orient;
            }
            if(var_name=="EFFECTIVETE")
            {
                std::string ok_temp=line.substr(temp_pos2+2);
                ok_temp=ok_temp.substr(0,ok_temp.length()-1);

                this->NTE = atoi(ok_temp.c_str());
            }
        }
    }
    infile.close();
}

void ConvertBrukerAnatomical::get_dimension_order(RECO_struct reco_struct,ACQP_struct acqp_struct,int dimension_order[])
{
    if(reco_struct.RECO_TRANSPOSITION==0)
    {
        dimension_order[0]=0;
        dimension_order[1]=1;
        dimension_order[2]=2;
        return;
    }

    if(acqp_struct.ACQ_DIM ==2)
    {
        dimension_order[0]=1;
        dimension_order[1]=0;
        dimension_order[2]=2;
    }
    else
    {
        if(reco_struct.RECO_TRANSPOSITION==1)
        {
            dimension_order[0]=1;
            dimension_order[1]=0;
            dimension_order[2]=2;
        }

        if(reco_struct.RECO_TRANSPOSITION==2)
        {
            dimension_order[0]=0;
            dimension_order[1]=2;
            dimension_order[2]=1;
        }

        if(reco_struct.RECO_TRANSPOSITION==3)
        {
            dimension_order[0]=2;
            dimension_order[1]=1;
            dimension_order[2]=0;

        }
    }

}

bool ConvertBrukerAnatomical::convert_bruker_anatomical(std::string input_folder, std::string output_filename)
{
    fs::path path(input_folder);

    fs::path acqp_file = path / std::string("acqp");
    if(!fs::exists(acqp_file))
    {
        std::cout<<"ACQP file not found in folder "<< input_folder<<std::endl;
        return 0;
    }

    fs::path method_file = path / std::string("method");
    if(!fs::exists(method_file))
    {
        std::cout<<"METHOD file not found in folder "<< input_folder<<std::endl;
        return 0;
    }

    fs::path pdata_folder = path / std::string("pdata");
    if(!fs::exists(pdata_folder))
    {
        std::cout<<"PDATA folder not found in folder "<< input_folder<<std::endl;
        return 0;
    }

    get_grad_orient(method_file.string());

    fs::directory_iterator it(pdata_folder);
    fs::directory_iterator endit;

    while(it != endit)
    {
        if(fs::is_directory(*it))
        {
            fs::path reco_file= it->path() / std::string("reco");
            std::string fname = it->path().filename().string();

            if(!fs::exists(reco_file))
            {
                std::cout<<"RECO file not found: "<< reco_file.string()<<std::endl;
                return 0;
            }

            fs::path twodseq_file= it->path() / std::string("2dseq");
            if(!fs::exists(twodseq_file))
            {
                std::cout<<"2dseq file not found: "<< twodseq_file.string()<<std::endl;
                return 0;
            }

            RECO_struct reco_struct= read_reco_header(reco_file.string());
            ACQP_struct acqp_struct= read_acqp_header(acqp_file.string(),reco_struct);
            if(acqp_struct.RG==1)
                acqp_struct.RG=RG;


            int dimension_order[3];
            get_dimension_order(reco_struct,acqp_struct,dimension_order);


            int a,b,c;
            a=reco_struct.RECO_SIZE[dimension_order[0]];
            b=reco_struct.RECO_SIZE[dimension_order[1]];
            c=reco_struct.RECO_SIZE[dimension_order[2]];

            reco_struct.RECO_SIZE[0]=a;
            reco_struct.RECO_SIZE[1]=b;
            reco_struct.RECO_SIZE[2]=c;

            float x,y,z;
            x=reco_struct.RECO_FOV[dimension_order[0]];
            y=reco_struct.RECO_FOV[dimension_order[1]];
            z=reco_struct.RECO_FOV[dimension_order[2]];
            reco_struct.RECO_FOV[0]=x;
            reco_struct.RECO_FOV[1]=y;
            reco_struct.RECO_FOV[2]=z;


            ImageType4DITK::Pointer DWI_image= get_bruker_image_data(twodseq_file.string(),reco_struct,acqp_struct);

            vnl_matrix<double> dir3(3,3);
            dir3.set_identity();

            if(this->slice_orientation=="axial")
            {
                if(this->read_orientation=="L_R")
                {
                    dir3= this->grad_orient;
                }
                else
                {
                    dir3.set_row(0, this->grad_orient.get_row(1));
                    dir3.set_row(1, this->grad_orient.get_row(0));
                    dir3.set_row(2, this->grad_orient.get_row(2));
                }
            }

            if(this->slice_orientation=="sagittal")
            {
                if(this->read_orientation=="H_F")
                {
                    dir3.set_row(0, this->grad_orient.get_row(1));
                    dir3.set_row(1, this->grad_orient.get_row(0));
                    dir3.set_row(2, this->grad_orient.get_row(2));
                }
                else
                {
                    dir3.set_row(0, this->grad_orient.get_row(0));
                    dir3.set_row(1, this->grad_orient.get_row(1));
                    dir3.set_row(2, this->grad_orient.get_row(2));
                }
            }

            if(this->slice_orientation=="coronal")
            {
                if(this->read_orientation=="H_F")
                {
                    dir3.set_row(0, this->grad_orient.get_row(1));
                    dir3.set_row(1, this->grad_orient.get_row(0));
                    dir3.set_row(2, this->grad_orient.get_row(2));
                }
                else
                {
                    dir3.set_row(0, this->grad_orient.get_row(0));
                    dir3.set_row(1, this->grad_orient.get_row(1));
                    dir3.set_row(2, this->grad_orient.get_row(2));
                }
            }


            dir3=dir3.transpose();

            if(parser->getConvertToAnatomicalHeader()==1)
            {
                if(this->slice_orientation=="axial")
                {
                    vnl_matrix<double> trmat(3,3);
                    trmat.fill(0);
                    trmat(0,0)=1;
                    trmat(1,2)=-1;
                    trmat(2,1)=-1;

                    dir3= trmat*dir3;
                }

                if(this->slice_orientation=="coronal")
                {
                    vnl_matrix<double> trmat(3,3);
                    trmat.fill(0);
                    trmat(0,0)=1;
                    trmat(1,2)=1;
                    trmat(2,1)=1;

                    dir3= trmat*dir3;
                }

                if(this->slice_orientation=="sagittal")
                {
                    vnl_matrix<double> trmat(3,3);
                    trmat.fill(0);
                    trmat(0,0)=1;
                    trmat(1,2)=1;
                    trmat(2,1)=-1;

                    dir3= trmat*dir3;
                }
            }

            vnl_matrix<double> dir4(4,4);
            dir4.set_identity();
            dir4(0,0)=dir3(0,0);dir4(0,1)=dir3(0,1);dir4(0,2)=dir3(0,2);
            dir4(1,0)=dir3(1,0);dir4(1,1)=dir3(1,1);dir4(1,2)=dir3(1,2);
            dir4(2,0)=dir3(2,0);dir4(2,1)=dir3(2,1);dir4(2,2)=dir3(2,2);

            //  std::cout<<dir<<std::endl;
            DWI_image->SetDirection(dir4);

            double mdet= vnl_determinant<double>(dir4);
            if(fabs(mdet+1)<0.1)
            {
                vnl_matrix<double> flipM(4,4);
                flipM.set_identity();
                flipM(1,1)=-1;


                vnl_matrix<double> new_dir = DWI_image->GetDirection().GetVnlMatrix() *flipM;

                ImageType4DITK::Pointer dwis2=ImageType4DITK::New();
                dwis2->SetRegions(DWI_image->GetLargestPossibleRegion());
                dwis2->Allocate();
                dwis2->SetSpacing(DWI_image->GetSpacing());
                dwis2->SetOrigin(DWI_image->GetOrigin());
                dwis2->SetDirection(new_dir);

                itk::ImageRegionIteratorWithIndex<ImageType4DITK> it(dwis2,dwis2->GetLargestPossibleRegion());
                it.GoToBegin();
                while(!it.IsAtEnd())
                {
                    ImageType4D::IndexType index =it.GetIndex();
                    index[1]= DWI_image->GetLargestPossibleRegion().GetSize()[1]-1 -index[1];
                    it.Set(DWI_image->GetPixel(index));
                    ++it;
                }
                DWI_image=dwis2;

            }


            std::string ofc=output_filename.substr(0,output_filename.find(".nii")) + std::string("_")+it->path().filename().string()+ std::string(".nii");

            typedef itk::ImageFileWriter<ImageType4DITK> WriterType;
            WriterType::Pointer wr=WriterType::New();
            wr->SetFileName(ofc);
            wr->SetInput(DWI_image);
            wr->Update();

            ++it;
        }
    }




    std::cout<<"Done importing " <<input_folder<<std::endl;
    return 1;
}





ImageType4DITK::Pointer ConvertBrukerAnatomical::get_bruker_image_data(std::string twodseq_file,RECO_struct reco_struct,ACQP_struct acqp_struct)
{
    int *image_data_int=NULL;
    short int * image_data_short=NULL;

    int nvols,nslices;

    bool is_BIG =is_big_endian();


    long fsize= fs::file_size(twodseq_file);

    if(acqp_struct.ACQ_DIM==2)
    {
        nslices=acqp_struct.Nslices;
    }
    else
        nslices=reco_struct.RECO_SIZE[2];


    nvols =  fsize/ reco_struct.RECO_SIZE[0]/reco_struct.RECO_SIZE[1]/nslices;

    if(reco_struct.RECO_WORDTYPE == std::string("_16BIT_SGN_INT"))
        nvols/=2;
    else
        nvols/=4;

    if(reco_struct.RECO_WORDTYPE != std::string("_16BIT_SGN_INT"))
    {
        image_data_int = new int[reco_struct.RECO_SIZE[0]*reco_struct.RECO_SIZE[1]*nslices*nvols];
    }
    else
    {
        image_data_short = new short[reco_struct.RECO_SIZE[0]*reco_struct.RECO_SIZE[1]*nslices*nvols];
    }

    long total_els= reco_struct.RECO_SIZE[0]*reco_struct.RECO_SIZE[1]*nslices*nvols;

    if( (image_data_int || image_data_short) == 0)
    {
        std::cout<<"Error allocating memory.. Exiting..."<<std::endl;
        exit(EXIT_FAILURE);
    }

    FILE *fp=fopen(twodseq_file.c_str(),"rb");
    if(reco_struct.RECO_WORDTYPE != std::string("_16BIT_SGN_INT"))
    {
        fread(image_data_int,sizeof(int),total_els,fp);
    }
    else
    {
        long count =fread(image_data_short,sizeof(short),total_els,fp);
    }
    fclose(fp);

    if(reco_struct.RECO_BYTE_ORDER== std::string("littleEndian"))
    {
        if(is_BIG)
        {
            if(reco_struct.RECO_WORDTYPE != std::string("_16BIT_SGN_INT"))
            {
                for(int i=0;i<total_els;i++)
                    image_data_int[i]=boost::endian::endian_reverse(image_data_int[i]);
            }
            else
            {
                for(int i=0;i<total_els;i++)
                    image_data_short[i]=boost::endian::endian_reverse(image_data_short[i]);
            }

        }
    }
    else
    {
        if(!is_BIG)
        {
            if(reco_struct.RECO_WORDTYPE != std::string("_16BIT_SGN_INT"))
            {
                for(int i=0;i<total_els;i++)
                    image_data_int[i]=boost::endian::endian_reverse(image_data_int[i]);
            }
            else
            {
                for(int i=0;i<total_els;i++)
                    image_data_short[i]=boost::endian::endian_reverse(image_data_short[i]);
            }
        }
    }

    float *image_data=NULL;
    image_data= new float[total_els];

    if(image_data==NULL)
    {
        std::cout<<"Not enough memory... Exiting..."<<std::endl;
        exit(EXIT_FAILURE);
    }


    if(reco_struct.RECO_WORDTYPE != std::string("_16BIT_SGN_INT"))
    {
        if(reco_struct.RECO_MAP_MODE==std::string("ABSOLUTE_MAPPING"))
        {
            for(int i=0;i<total_els;i++)
                image_data[i]=1.0* image_data_int[i]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(0,0);
        }
        else
        {
            if(reco_struct.RECO_MAP_SLOPE.columns()==1 )
            {
                long cnt=0;
                int cnt2=-1;
                for(int l=0;l<nvols;l++)
                    for(int k=0;k<nslices;k++)
                    {
                        cnt2++;
                        for(int j=0;j<reco_struct.RECO_SIZE[1];j++)
                            for(int i=0;i<reco_struct.RECO_SIZE[0];i++)
                            {
                                image_data[cnt]=1.0*image_data_int[cnt]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(cnt2,0);
                                cnt++;
                            }
                    }
            }
            else
            {
                if(reco_struct.RECO_MAP_SLOPE.columns()==nvols && acqp_struct.ACQ_DIM==3)
                {
                    long cnt=0;
                    for(int l=0;l<nvols;l++)
                        for(int k=0;k<nslices;k++)
                            for(int j=0;j<reco_struct.RECO_SIZE[1];j++)
                                for(int i=0;i<reco_struct.RECO_SIZE[0];i++)
                                {
                                    image_data[cnt]=1.0*image_data_int[cnt]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(l,0);
                                    cnt++;
                                }

                }
                else
                {
                    long cnt=0;
                    for(int l=0;l<nvols;l++)
                        for(int k=0;k<nslices;k++)
                            for(int j=0;j<reco_struct.RECO_SIZE[1];j++)
                                for(int i=0;i<reco_struct.RECO_SIZE[0];i++)
                                {
                                    image_data[cnt]=1.0*image_data_int[cnt]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(l,k);
                                    cnt++;
                                }
                }
            }
        }
    }
    else
    {
        if(reco_struct.RECO_MAP_MODE==std::string("ABSOLUTE_MAPPING"))
        {
            for(int i=0;i<total_els;i++)
                image_data[i]=1.0* image_data_short[i]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(0,0);
        }
        else
        {
            if(reco_struct.RECO_MAP_SLOPE.columns()==1 )
            {
                long cnt=0;
                int cnt2=-1;
                for(int l=0;l<nvols;l++)
                    for(int k=0;k<nslices;k++)
                    {
                        cnt2++;
                        for(int j=0;j<reco_struct.RECO_SIZE[1];j++)
                            for(int i=0;i<reco_struct.RECO_SIZE[0];i++)
                            {
                                image_data[cnt]=1.0*image_data_short[cnt]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(cnt2,0);
                                cnt++;
                            }
                    }
            }
            else
            {
                if(reco_struct.RECO_MAP_SLOPE.columns()==nvols && acqp_struct.ACQ_DIM==3)
                {
                    long cnt=0;
                    for(int l=0;l<nvols;l++)
                        for(int k=0;k<nslices;k++)
                            for(int j=0;j<reco_struct.RECO_SIZE[1];j++)
                                for(int i=0;i<reco_struct.RECO_SIZE[0];i++)
                                {
                                    image_data[cnt]=1.0*image_data_short[cnt]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(l,0);
                                    cnt++;
                                }

                }
                else
                {
                    long cnt=0;
                    for(int l=0;l<nvols;l++)
                        for(int k=0;k<nslices;k++)
                            for(int j=0;j<reco_struct.RECO_SIZE[1];j++)
                                for(int i=0;i<reco_struct.RECO_SIZE[0];i++)
                                {
                                    image_data[cnt]=1.0*image_data_short[cnt]/acqp_struct.RG/reco_struct.RECO_MAP_SLOPE(l,k);
                                    cnt++;
                                }
                }
            }
        }
    }


    if(image_data_int)
    {
        delete[] image_data_int;
        image_data_int=NULL;
    }

    if(image_data_short)
    {
        delete[] image_data_short;
        image_data_short=NULL;
    }


    if(acqp_struct.software_version==6)
    {
        for(int i=0;i<total_els;i++)
            image_data[i]*=10000.;
    }


    ImageType4DITK::Pointer output_img=nullptr;



    typedef itk::ImportImageFilter< float, 4 >   ImportFilterType;
    ImportFilterType::Pointer importFilter = ImportFilterType::New();
    ImportFilterType::SizeType  size;
    ImportFilterType::IndexType start;
    start.Fill( 0 );
    ImportFilterType::SpacingType res;

    if(nvols==1 || acqp_struct.ACQ_DIM==3 || acqp_struct.ACQ_obj_order == nslices)
    {
        size[0]=reco_struct.RECO_SIZE[0];
        size[1]=reco_struct.RECO_SIZE[1];
        size[2]=nslices;
        size[3]=nvols;
    }
    else
    {
        size[0]=reco_struct.RECO_SIZE[0];
        size[1]=reco_struct.RECO_SIZE[1];
        size[2]=nvols;
        size[3]=nslices;
    }



    ImportFilterType::RegionType region;
    region.SetIndex( start );
    region.SetSize(  size  );

    importFilter->SetRegion( region );


    if(nvols==1 || acqp_struct.ACQ_DIM==3 || acqp_struct.ACQ_obj_order == nslices)
    {
        res[0]= reco_struct.RECO_FOV[0]/size[0];
        res[1]= reco_struct.RECO_FOV[1]/size[1];
        res[2]= reco_struct.RECO_FOV[2]/size[2];
        res[3]=1;

        if(acqp_struct.ACQ_DIM!=3)
        {
            res[2]= 1.* acqp_struct.ACQ_SLICE_THICK;
        }
    }
    else
    {
        res[0]= reco_struct.RECO_FOV[0]/size[0];
        res[1]= reco_struct.RECO_FOV[1]/size[1];
        res[2]= 1;
        res[3]=reco_struct.RECO_FOV[2]/size[2];

        if(acqp_struct.ACQ_DIM!=3)
        {
            res[3]= 1.* acqp_struct.ACQ_SLICE_THICK;
        }

    }


    importFilter->SetSpacing(res );

    ImageType4DITK::PointType orig;
    orig[0]= -((double)size[0]-1)* res[0]/2;
    orig[1]= -((double)size[1]-1)* res[1]/2;
    orig[2]= -((double)size[2]-1)* res[2]/2;
    orig[3]= -((double)size[2]-1)* res[3]/2;;

    importFilter->SetOrigin( orig );


    importFilter->SetImportPointer( image_data, total_els, false );
    importFilter->Update();

    output_img=importFilter->GetOutput();


    if(nvols!=1 && acqp_struct.ACQ_DIM==2 && acqp_struct.ACQ_obj_order != nslices)
    {
        typedef itk::PermuteAxesImageFilter <ImageType4DITK>    PermuteAxesImageFilterType;
        itk::FixedArray<unsigned int, 4> order;
        order[0] = 0;
        order[1] = 1;
        order[2] = 3;
        order[3] = 2;

        PermuteAxesImageFilterType::Pointer permuteAxesFilter          = PermuteAxesImageFilterType::New();
        permuteAxesFilter->SetInput(output_img);
        permuteAxesFilter->SetOrder(order);
        permuteAxesFilter->Update();
        output_img= permuteAxesFilter->GetOutput();


    }
    return output_img;
}



RECO_struct ConvertBrukerAnatomical::read_reco_header(std::string reco_file)
{
    RECO_struct reco_struct;

      std::ifstream infile(reco_file.c_str());
      std::string line;
      while (std::getline(infile, line))
      {
          line.erase(std::remove_if(line.begin(), line.end(), ::isspace),line.end());
          if(line.find("##TITLE")!=std::string::npos)
          {
              int temp_pos2= line.find("=");
              reco_struct.version=line.substr(temp_pos2+1 );
          }


          if(line.find("##$")==0)
          {
              int temp_pos1= line.find("$")+1;
              int temp_pos2= line.find("=");

              std::string var_name= line.substr(temp_pos1, temp_pos2-temp_pos1);
              std::transform(var_name.begin(), var_name.end(),var_name.begin(), ::toupper);


              if(var_name==std::string("RECO_WORDTYPE"))
              {
                  reco_struct.RECO_WORDTYPE=line.substr(temp_pos2+1);
              }

              if(var_name==std::string("RECO_BYTE_ORDER"))
              {
                  reco_struct.RECO_BYTE_ORDER=line.substr(temp_pos2+1);
              }

              if(var_name==std::string("RECO_SIZE"))
              {

                  int ndims= atoi(line.substr(temp_pos2+2).c_str());
                  std::getline(infile, line);

                  int pos1= line.find(" ");
                  reco_struct.RECO_SIZE[0]=   atoi(line.substr(0,pos1).c_str());
                  if(ndims>2)
                  {
                      int pos2= line.find(' ' ,pos1+1);
                      reco_struct.RECO_SIZE[1]=   atoi(line.substr(pos1+1,pos2-pos1).c_str());
                      reco_struct.RECO_SIZE[2]=   atoi(line.substr(pos2).c_str());
                  }
                  else
                  {
                      reco_struct.RECO_SIZE[1]=   atoi(line.substr(pos1+1).c_str());
                      reco_struct.RECO_SIZE[2]=1;
                  }
              }

              if(var_name==std::string("RECO_MAP_MODE"))
              {
                  reco_struct.RECO_MAP_MODE=line.substr(temp_pos2+1);
              }

              if(var_name==std::string("RECO_TRANSPOSITION"))
              {
                  std::getline(infile, line);
                  if(line.at(0)=='@')
                  {
                      std::string ok_temp= line.substr(line.find('(')+1);
                      reco_struct.RECO_TRANSPOSITION= atoi(ok_temp.c_str());

                  }
                  else
                  {
                      reco_struct.RECO_TRANSPOSITION= atoi(line.c_str());
                  }

              }

              if(var_name==std::string("RECO_FOV"))
              {
                  int ndims= atoi(line.substr(temp_pos2+2).c_str());
                  std::getline(infile, line);

                  int pos1= line.find(' ');
                  reco_struct.RECO_FOV[0]=   atof(line.substr(0,pos1).c_str())*10;

                  if(ndims>2)
                  {
                      int pos2= line.find(' ' ,pos1+1);
                      reco_struct.RECO_FOV[1]=   atof(line.substr(pos1+1,pos2-pos1).c_str())*10;
                      reco_struct.RECO_FOV[2]=   atof(line.substr(pos2+1).c_str())*10;
                  }
                  else
                  {
                      reco_struct.RECO_FOV[1]=   atof(line.substr(pos1+1).c_str())*10;
                      reco_struct.RECO_FOV[2]=-1;
                  }
              }

              if(var_name==std::string("RECO_MAP_SLOPE"))
              {
                  if(reco_struct.version.find("ParaVision360")!=std::string::npos)
                  {
                      std::string tmp_line;
                      std::getline(infile, tmp_line);

                      reco_struct.RECO_MAP_SLOPE.set_size(1,1);

                      int tpos1= tmp_line.find('(');
                      int tpos2= tmp_line.find(')');

                      tmp_line= tmp_line.substr(tpos1+1,tpos2-tpos1-1);
                      reco_struct.RECO_MAP_SLOPE(0,0)= atof(tmp_line.c_str());
                  }
                  else
                  {
                      int cpos= line.find(',',temp_pos2+1);
                      if(cpos== std::string::npos)
                      {
                          int other_dim=  atoi(line.substr(temp_pos2+2).c_str());
                          reco_struct.RECO_MAP_SLOPE.set_size(other_dim,1);
                          infile>>std::setprecision(16)>>reco_struct.RECO_MAP_SLOPE;
                      }
                      else
                      {
                          int dim3= atoi(line.substr(temp_pos2+2,cpos-temp_pos2-1).c_str());
                          int other_dim=  atoi(line.substr(cpos+1).c_str());
                          reco_struct.RECO_MAP_SLOPE.set_size(other_dim,dim3);
                          infile>>std::setprecision(16)>>reco_struct.RECO_MAP_SLOPE;
                      }
                  }

              }

              if(var_name==std::string("RECO_IR_SCALE"))
              {
                  reco_struct.RECO_IR_SCALE=atoi(line.substr(temp_pos2+1).c_str());
              }

          }
      }


      infile.close();

      return reco_struct;
}



ACQP_struct ConvertBrukerAnatomical::read_acqp_header(std::string acqp_file, RECO_struct reco_struct)
{
    ACQP_struct acqp_struct;

    acqp_struct.RG=1.;

    std::ifstream infile(acqp_file.c_str());
    std::string line;
    while (std::getline(infile, line))
    {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace),line.end());
        if(line.find("##OWNER")!=std::string::npos)
        {
            std::getline(infile, line);
            if(line.find("/opt")==std::string::npos)
                std::getline(infile, line);

            int pos = line.find("PV");
            acqp_struct.software_version =line.at(pos+2) -'0';

        }



        if(line.find("##$")==0)
        {
            int temp_pos1= line.find("$")+1;
            int temp_pos2= line.find("=");

            std::string var_name= line.substr(temp_pos1, temp_pos2-temp_pos1);
            std::transform(var_name.begin(), var_name.end(),var_name.begin(), ::toupper);


            if(var_name==std::string("ACQ_SCALING_READ"))
            {
                acqp_struct.ACQ_SCALING[0]= atof(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("ACQ_SCALING_PHASE"))
            {
                acqp_struct.ACQ_SCALING[1]= atof(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("ACQ_SCALING_SLICE"))
            {
                acqp_struct.ACQ_SCALING[2]= atof(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("ACQ_SLICE_SEPN_MODE"))
            {
                acqp_struct.ACQ_SLICE_SEPN_MODE= line.substr(temp_pos2+1);
            }

            if(var_name==std::string("ACQ_DIM"))
            {
                acqp_struct.ACQ_DIM= atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("ACQ_SLICE_SEPN"))
            {
                std::getline(infile, line);
                acqp_struct.ACQ_SLICE_SEPN= atof(line.c_str());
            }

            if(var_name==std::string("ACQ_NS_LIST_SIZE"))
            {
                acqp_struct.ACQ_NS_LIST_SIZE= atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("NI"))
            {
                acqp_struct.NI= atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("NA"))
            {
                acqp_struct.NA= atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("NR"))
            {
                acqp_struct.NR= atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("ACQ_NR_COMPLETED"))
            {
                acqp_struct.ACQ_NR_COMPLETED= atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("ACQ_SLICE_THICK"))
            {
                acqp_struct.ACQ_SLICE_THICK= atof(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("RG"))
            {
                acqp_struct.RG= atof(line.substr(temp_pos2+1).c_str());
            }


            if(var_name==std::string("NSLICES"))
            {
                acqp_struct.Nslices= atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("ACQ_OBJ_ORDER"))
            {
                std::string ok_temp=line.substr(temp_pos2+2);
                ok_temp=ok_temp.substr(0,ok_temp.length()-1);

                acqp_struct.ACQ_obj_order= atoi(ok_temp.c_str());
            }






            if(var_name==std::string("ACQ_GRAD_MATRIX"))
            {
                int num_matrices=atoi(line.substr(temp_pos2+2).c_str());
                int cpos= line.find(',',temp_pos2);

                int dim1 = atoi(line.substr(cpos+1).c_str());
                cpos= line.find(',',cpos+1);
                int dim2 = atoi(line.substr(cpos+1).c_str());
                acqp_struct.dircos_matrix.set_size(dim1,dim2);
                infile>>acqp_struct.dircos_matrix;


                for(int r=0;r<dim1;r++)
                {
                    for(int c=0;c<dim2;c++)
                    {
                        if(acqp_struct.dircos_matrix(r,c)==-0)
                            acqp_struct.dircos_matrix(r,c)=0;
                    }
                }

                acqp_struct.phase=std::string("vertical");
                acqp_struct.transpos=0;

                if(acqp_struct.ACQ_DIM==2 && reco_struct.RECO_TRANSPOSITION>0)
                {
                    acqp_struct.transpos=1;           //transpose read/phase
                    vnl_vector<double> tempv= acqp_struct.dircos_matrix.get_row(0);
                    acqp_struct.dircos_matrix.set_row(0,acqp_struct.dircos_matrix.get_row(1));
                    acqp_struct.dircos_matrix.set_row(1,tempv);
                    acqp_struct.phase=std::string("horizontal");
                }
                else
                {
                    if(acqp_struct.ACQ_DIM==3)
                    {
                        if(reco_struct.RECO_TRANSPOSITION==1)
                        {
                            acqp_struct.transpos=1;           //transpose read/phase
                            vnl_vector<double> tempv= acqp_struct.dircos_matrix.get_row(0);
                            acqp_struct.dircos_matrix.set_row(0,acqp_struct.dircos_matrix.get_row(1));
                            acqp_struct.dircos_matrix.set_row(1,tempv);
                            acqp_struct.phase=std::string("horizontal");
                        }
                        else
                        {
                            if(reco_struct.RECO_TRANSPOSITION==2)
                            {
                                acqp_struct.transpos=2;  //transpose phase/slice
                                vnl_vector<double> tempv= acqp_struct.dircos_matrix.get_row(1);
                                acqp_struct.dircos_matrix.set_row(1,acqp_struct.dircos_matrix.get_row(2));
                                acqp_struct.dircos_matrix.set_row(2,tempv);
                                acqp_struct.phase=std::string("slice");
                            }
                            else
                            {
                                acqp_struct.transpos=3;  //transpose read/slice
                                vnl_vector<double> tempv= acqp_struct.dircos_matrix.get_row(0);
                                acqp_struct.dircos_matrix.set_row(0,acqp_struct.dircos_matrix.get_row(2));
                                acqp_struct.dircos_matrix.set_row(2,tempv);
                            }
                        }
                    }
                }
                acqp_struct.image_orientation = acqp_struct.get_bruker_orientation();
            }
        }
    }

    if(acqp_struct.NI>1 && reco_struct.RECO_SIZE[2]==1)
        reco_struct.RECO_SIZE[2]=acqp_struct.NI;

    infile.close();


    return acqp_struct;
}
