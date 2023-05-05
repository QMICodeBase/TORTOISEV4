#include "convert_bruker.h"
#include "combine_listfiles.h"
#include "write_bmatrix_file.h"
#include "TORTOISE_Utilities.h"
#include "read_3Dvolume_from_4D.h"


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

#include "itkOkanQuadraticTransform.h"


typedef itk::OkanQuadraticTransform<double,3,3> OkanQuadraticTransformType;


ConvertBruker::ConvertBruker( int argc , char * argv[] )
{
    parser= new Bruker_PARSER(argc,argv);



    std::string input_folder = parser->getInputDataFolder();
    if(input_folder.at(input_folder.length()-1)==boost::filesystem::path::preferred_separator)
        input_folder= input_folder.substr(0,input_folder.length()-1);

    fs::path data_dir(input_folder);

    std::string output_folder = parser->getOutputProcFolder();
    fs::path output_proc;


    if(output_folder== std::string(""))
    {
        output_proc = data_dir/ std::string("..") /  (data_dir.filename().string() + std::string("_proc"));
    }
    else
    {
        if(output_folder.find("_proc")==std::string::npos)
        {
            if(output_folder.at(output_folder.length()-1)=='/')
                output_folder=output_folder.substr(0,output_folder.length()-1);

             output_folder= output_folder + std::string("_proc");
             output_proc= fs::path(output_folder);
        }
        else
        {
            fs::path dummy(output_folder);
            output_proc=dummy;
        }
    }



    std::vector<std::string> new_nifti_files;
    std::string final_listfilename;

    if(fs::exists(data_dir / std::string("pdata")))
    {
        bool error_code = this->convert_bruker(input_folder,output_proc.string(),new_nifti_files);
        final_listfilename = new_nifti_files[0].substr(0,new_nifti_files[0].find(".nii")) + std::string(".list");
        if(error_code==0)
        {
            std::cout<<"ConvertBruker with folder " << input_folder<< " failed. Exiting!!"<<std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::vector<fs::path> folders;

        fs::directory_iterator it(data_dir);
        fs::directory_iterator endit;

        while(it != endit)
        {
            if(fs::is_directory(*it))
            {
                fs::path curr_input_folder= it->path();
                folders.push_back(curr_input_folder);
            }
            ++it;
        }

        std::sort(folders.begin(), folders.end());

        for(int f=0;f<folders.size();f++)
        {
            fs::path curr_input_folder= folders[f];
            fs::path toutput_proc = curr_input_folder / std::string("..") / (curr_input_folder.filename().string() + std::string("_proc"));
            bool error_code = this->convert_bruker(curr_input_folder.string(),toutput_proc.string(),new_nifti_files);
            if(error_code==0)
            {
                std::cout<<"ConvertBruker with folder " << curr_input_folder<< " failed."<<std::endl;
            }
        }
        final_listfilename =combine_listfiles(new_nifti_files,output_proc);

    }


    std::string info_file= final_listfilename.substr(0,final_listfilename.find(".list")) + std::string(".info");
    std::ofstream info(info_file.c_str());
    for(int i=0;i<argc;i++)
    {
        info<< argv[i] << " ";
    }
    info<<std::endl<<std::endl;

    info<<"TORTOISE version: "<<GetTORTOISEVersion()<<std::endl<<std::endl;

    info<<"Raw data format: Bruker 2DSEQ"<<std::endl;
    info<<"Number of series in the dataset: " <<new_nifti_files.size()<<std::endl;
    if(new_nifti_files.size()>0)
    {
        LISTFILE list(final_listfilename);
        ImageType3D::Pointer first_vol= read_3D_volume_from_4D(list.GetDataFile(),0);
        info<<"Number of volumes: "<<list.GetBmatrix().rows()<<std::endl;
        info<<"Size: " << first_vol->GetLargestPossibleRegion().GetSize()[0]<< " "<<first_vol->GetLargestPossibleRegion().GetSize()[1]<< " "<<first_vol->GetLargestPossibleRegion().GetSize()[2]<<std::endl;
        info<<"Resolution: "<< first_vol->GetSpacing()[0]<< " "<<first_vol->GetSpacing()[1]<< " "<<first_vol->GetSpacing()[2]<<std::endl;
    }
    info.close();

} 




ConvertBruker::~ConvertBruker()
{
}


std::vector<int> ConvertBruker::get_phase(METHOD_struct &method_struct,RECO_struct &reco_struct, VISU_struct &visu_struct)
{
    std::vector<int> phase_vector;
    if(visu_struct.dim==2)
    {
        if(reco_struct.RECO_TRANSPOSITION==0)
        {
            phase_vector.push_back(1);
        }
        else
        {
            phase_vector.push_back(0);
        }
    }
    else
    {
        if(reco_struct.RECO_TRANSPOSITION==0)
        {
            phase_vector.push_back(1);
        }
        if(reco_struct.RECO_TRANSPOSITION==1)
        {
            phase_vector.push_back(0);
        }
        if(reco_struct.RECO_TRANSPOSITION==2)
        {
            phase_vector.push_back(2);
        }

    }

    if(method_struct.PVM_EPI_BlipDir!=0)
    {
        phase_vector.push_back(method_struct.PVM_EPI_BlipDir);
    }
    else
    {
        if(method_struct.phase_mode=="Forward")
            phase_vector.push_back(1);
        else
            phase_vector.push_back(-1);
    }

    return phase_vector;

}

bool ConvertBruker::convert_bruker(std::string input_folder, std::string output_folder, std::vector<std::string> &new_nifti_files)
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


    METHOD_struct method_struct= read_method_file(method_file.string());


    if(!boost::filesystem::exists(output_folder))
        boost::filesystem::create_directory(output_folder);


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


            fs::path visu_file= it->path() / std::string("visu_pars");
            if(!fs::exists(visu_file))
            {
                std::cout<<"VISU_PARS file not found: "<< visu_file.string()<<std::endl;
                return 0;
            }
            VISU_struct visu_struct= read_visu_file(visu_file.string());


            std::vector<int> phase_vector= get_phase(method_struct,reco_struct,visu_struct);
            int convert_to_anat= parser->getConvertToAnatomicalHeader();
            ImageType4DITK::Pointer DWI_image= get_bruker_image_data(twodseq_file.string(),reco_struct,method_struct,visu_struct,convert_to_anat,phase_vector);


            //bool use_gradients =parser->getUseGradientsInsteadOfBMatrix();
            bool use_gradients=false;
            vnl_matrix<double> Bmatrix = get_Bmatrix(reco_struct,method_struct,visu_struct,phase_vector,use_gradients);



            std::string phase;
            if(phase_vector[0]==0)
                phase="horizontal";
            if(phase_vector[0]==1)
                phase="vertical";
            if(phase_vector[0]==2)
                phase="slice";

            if(parser->getConvertToAnatomicalHeader()==2)
            {
                vnl_matrix<double> trans_matrix = ReorientToAnatomical(DWI_image,Bmatrix);

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
            }


            fs::path new_proc(output_folder);

            std::string basename= new_proc.stem().string();
            if(output_folder.rfind("_proc")== output_folder.length()-5)
            {
                basename = basename.substr(0,basename.find(("_proc")));
            }

            std::string new_listf= (new_proc / (basename+ std::string(".list"))).string();
            std::string new_niif= (new_proc / (basename+ std::string(".nii"))).string();


            typedef itk::ImageFileWriter<ImageType4DITK> WriterType;
            WriterType::Pointer wr=WriterType::New();
            wr->SetFileName(new_niif);
            wr->SetInput(DWI_image);
            wr->Update();




            LISTFILE newlist;
            newlist.SetListFileName(new_listf);
            newlist.SetBmatrix(Bmatrix);
            newlist.SetBigDelta(method_struct.BIG_DELTA);
            newlist.SetSmallDelta(method_struct.small_delta);
            newlist.SetPhaseEncodingDir(phase);
            newlist.WriteListFileWithBMatrix();


            new_nifti_files.push_back(newlist.GetDataFile());

            ++it;
        }
    }

    std::cout<<"Done importing " <<input_folder<<std::endl;
    return 1;
}

ImageType4DITK::Pointer MakeDetOne(ImageType4DITK::Pointer dwis,vnl_matrix<double> & Bmatrix,vnl_matrix<double> &trans_matrix)
{

    vnl_matrix<double> actual_dir(3,3);
    actual_dir= dwis->GetDirection().GetVnlMatrix().extract(3,3);
    trans_matrix=actual_dir;


    double mdet = vnl_determinant<double>(actual_dir);
    if(fabs(mdet+1)<0.1)
    {
        vnl_matrix<double> flipM(3,3);
        flipM.set_identity();
        flipM(1,1)=-1;

        trans_matrix= trans_matrix *flipM;

        vnl_matrix<double> new_dir= actual_dir * flipM;
        ImageType4DITK::DirectionType img_dir;
        img_dir.SetIdentity();
        img_dir(0,0)=new_dir(0,0);img_dir(0,1)=new_dir(0,1);img_dir(0,2)=new_dir(0,2);
        img_dir(1,0)=new_dir(1,0);img_dir(1,1)=new_dir(1,1);img_dir(1,2)=new_dir(1,2);
        img_dir(2,0)=new_dir(2,0);img_dir(2,1)=new_dir(2,1);img_dir(2,2)=new_dir(2,2);



        ImageType4DITK::DirectionType DPdir=  dwis->GetDirection();
        ImageType4DITK::PointType DPorig=  dwis->GetOrigin();

        ImageType4DITK::IndexType new_orig_index;
        new_orig_index.Fill(0);
        new_orig_index[1]=dwis->GetLargestPossibleRegion().GetSize()[1]-1;


        ImageType4DITK::PointType new_orig;
        dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);



        ImageType4DITK::Pointer new_str= ImageType4DITK::New();
        new_str->SetRegions(dwis->GetLargestPossibleRegion());
        new_str->Allocate();
        new_str->SetSpacing(dwis->GetSpacing());
        new_str->SetDirection(img_dir);
        new_str->SetOrigin(new_orig);

        itk::ImageRegionIteratorWithIndex<ImageType4DITK> it(new_str,new_str->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            ImageType4DITK::IndexType index= it.GetIndex();
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


ImageType4DITK::Pointer ConvertBruker::ReorientDWIs(ImageType4DITK::Pointer dwis,vnl_matrix<double> trans_matrix)
{
    ImageType4DITK::SpacingType new_spc;
      new_spc[3]=1;

      ImageType4DITK::SizeType new_sizes;
      new_sizes[3]= dwis->GetLargestPossibleRegion().GetSize()[3];

      ImageType4DITK::IndexType new_orig_index;
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

      ImageType4DITK::PointType new_orig;
      dwis->TransformIndexToPhysicalPoint(new_orig_index,new_orig);


      ImageType4DITK::IndexType start; start.Fill(0);
      ImageType4DITK::RegionType reg(start,new_sizes);
      ImageType4DITK::DirectionType dir;
      dir.SetIdentity();

      ImageType4DITK::Pointer new_dwis=ImageType4DITK::New();
      new_dwis->SetRegions(reg);
      new_dwis->Allocate();
      new_dwis->SetSpacing(new_spc);
      new_dwis->SetDirection(dir);
      new_dwis->SetOrigin(new_orig);


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


      itk::ImageRegionIteratorWithIndex<ImageType4DITK> it(dwis,dwis->GetLargestPossibleRegion());
      it.GoToBegin();
      vnl_vector<double> ind3(3);
      while(!it.IsAtEnd())
      {
          ImageType4DITK::IndexType old_index= it.GetIndex();
          ind3[0]=old_index[0];
          ind3[1]=old_index[1];
          ind3[2]=old_index[2];

          ImageType4DITK::IndexType new_index;
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

vnl_matrix<double>  ConvertBruker::RotateBmatrix(vnl_matrix<double>Bmatrix,vnl_matrix<double> trans_matrix)
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

vnl_matrix<double>  ConvertBruker::ReorientToAnatomical(ImageType4DITK::Pointer &DWI_image,vnl_matrix<double> &Bmatrix)
{
    vnl_matrix<double> temp_Bmatrix= Bmatrix;
    vnl_matrix<double> trans_matrix;

    ImageType4DITK::Pointer temp_dwis= MakeDetOne(DWI_image, temp_Bmatrix,trans_matrix);

    DWI_image = ReorientDWIs(temp_dwis,trans_matrix);
    Bmatrix= RotateBmatrix(temp_Bmatrix,trans_matrix);

    return trans_matrix;


}


vnl_matrix<double>  ConvertBruker::get_Bmatrix(RECO_struct &reco_struct,METHOD_struct &method_struct,VISU_struct visu_struct,std::vector<int> &phase_vector,bool use_gradients)
{
    if(!use_gradients)
    {
        return method_struct.Bmatrix;
    }

}










void ConvertBruker::write_bruker_listfile(std::string output_folder,std::string fname,std::string phase,double sdelta,double bDELTA)
{
    std::string listfile= output_folder + std::string("/") + fname +std::string(".list");

    std::ofstream outfile(listfile.c_str());
    outfile<< "<!-- TORTOISE initialization file -->"<<std::endl;
    outfile<< "<!-- do not remove the above comment line -->"<<std::endl<<std::endl<<std::endl;

    outfile<< "<!-- The 4D NIFTI image containing the b=0 volumes and the DWIs -->"<<std::endl;
    outfile<< "<data_file>"<< output_folder + std::string("/") + fname +std::string(".nii")<<"</data_file>" <<std::endl<<std::endl;

    outfile<<"<!-- ORIENTATION OF PHASE ENCODING (vertical, horizontal) -->"<<std::endl;
    outfile<<"<phase_encode_direction>"<<  phase  << "</phase_encode_direction>"<<std::endl<<std::endl;

    outfile<<"<!-- NAME OF BMATRIX FILE -->"<<std::endl;
    outfile<<"<bmatrixfile>"<<output_folder + std::string("/") + fname +std::string(".bmtxt")<<"</bmatrixfile>"<<std::endl<<std::endl;

    outfile<<"<!-- small delta of diffusion weighting -->"<<std::endl;
    outfile<<"<small_delta>"<<sdelta<<"</small_delta>"<<std::endl<<std::endl;

    outfile<<"<!-- BIG DELTA of diffusion weighting -->"<<std::endl;
    outfile<<"<big_delta>"<<bDELTA<<"</big_delta>"<<std::endl<<std::endl;

    outfile<<std::endl;
    outfile.close();



}




ImageType4DITK::Pointer ConvertBruker::get_bruker_image_data(std::string twodseq_file,RECO_struct &reco_struct, METHOD_struct &method_struct, VISU_struct &visu_struct, int convert_to_anatomical,std::vector<int> &phase_vector)
{
    int *image_data_int=NULL;
    short int * image_data_short=NULL;


    bool is_BIG =is_big_endian();

    int Nechoes=1;
    int cycle= method_struct.N_A0 + method_struct.NdiffExp * method_struct.Ndir;
    int nreps= method_struct.N_reps;
    int N_input_channel=1;

    int dims[3];
    ImageType4DITK::SizeType final_dims;
    dims[0]= visu_struct.size[0];
    dims[1]= visu_struct.size[1];
    dims[2]=1;
    final_dims[0]=dims[0];
    final_dims[1]=dims[1];
    final_dims[2]=visu_struct.nslices;
    if(visu_struct.dim==3)
    {
        dims[2]=visu_struct.size[2];
        final_dims[2]=dims[2];
    }
    final_dims[3]=cycle;

    long total_els= (long)final_dims[0]*final_dims[1]*final_dims[2]*Nechoes*cycle*method_struct.N_reps*N_input_channel;


    if(visu_struct.word_type==std::string("_16BIT_SGN_INT"))
    {
        image_data_short = new short[total_els];
    }
    else
    {
        image_data_int = new int[total_els];
    }


    if( (image_data_int==nullptr && image_data_short==nullptr))
    {
        std::cout<<"Error allocating memory.. Exiting..."<<std::endl;
        exit(EXIT_FAILURE);
    }

    FILE *fp=fopen(twodseq_file.c_str(),"rb");
    if( visu_struct.word_type !=std::string("_16BIT_SGN_INT"))
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
                for(long i=0;i<total_els;i++)
                    image_data_int[i]=boost::endian::endian_reverse(image_data_int[i]);
            }
            else
            {
                for(long i=0;i<total_els;i++)
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
                for(long i=0;i<total_els;i++)
                    image_data_int[i]=boost::endian::endian_reverse(image_data_int[i]);
            }
            else
            {
                for(long i=0;i<total_els;i++)
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
        for(long i=0;i<total_els;i++)
            image_data[i]=1.0* image_data_int[i]/method_struct.RG * visu_struct.slope;

    }
    else
    {
        for(long i=0;i<total_els;i++)
            image_data[i]=1.0* image_data_short[i]/method_struct.RG * visu_struct.slope;

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


    typedef itk::ImportImageFilter< float, 4 >   ImportFilterType;
    ImportFilterType::Pointer importFilter = ImportFilterType::New();

    ImportFilterType::IndexType start;
    start.Fill( 0 );
    ImportFilterType::RegionType region;
    region.SetIndex( start );
    region.SetSize(  final_dims  );
    importFilter->SetRegion( region );   

    ImportFilterType::SpacingType spc;
    spc[0]= 1.*visu_struct.extent[0]/visu_struct.size[0];
    spc[1]= 1.*visu_struct.extent[1]/visu_struct.size[1];
    if(visu_struct.dim==2)
        spc[2]=visu_struct.slice_thickness;
    else
        spc[2]= 1.*visu_struct.extent[2]/visu_struct.size[2];
    spc[3]=1;
    importFilter->SetSpacing(spc);

    ImageType4DITK::PointType orig;
    orig[0]=visu_struct.pos[0];
    orig[1]=visu_struct.pos[1];
    orig[2]=visu_struct.pos[2];
    orig[3]=0;
    importFilter->SetOrigin( orig );

    ImageType4DITK::DirectionType dir;
    dir.SetIdentity();
    dir(0,0)= visu_struct.orientation(0,0);dir(0,1)= visu_struct.orientation(0,1);dir(0,2)= visu_struct.orientation(0,2);
    dir(1,0)= visu_struct.orientation(1,0);dir(1,1)= visu_struct.orientation(1,1);dir(1,2)= visu_struct.orientation(1,2);
    dir(2,0)= visu_struct.orientation(2,0);dir(2,1)= visu_struct.orientation(2,1);dir(2,2)= visu_struct.orientation(2,2);

    if(phase_vector[1]==-1)
    {
        ImageType4DITK::DirectionType id;
        id.SetIdentity();
        id(phase_vector[0],phase_vector[0])=-1;
        dir= id*dir;
    }


    dir= dir.GetTranspose();

    if(convert_to_anatomical>0)
    {
        if(method_struct.SliceOrient=="axial")
        {
            ImageType4DITK::DirectionType  trmat;
            trmat.Fill(0); trmat(3,3)=1;
            trmat(0,0)=1;
            trmat(1,2)=1;
            trmat(2,1)=-1;

            dir= trmat*dir;
        }
        if(method_struct.SliceOrient=="coronal")
        {
            ImageType4DITK::DirectionType  trmat;
            trmat.Fill(0); trmat(3,3)=1;
            trmat(0,0)=1;
            trmat(1,2)=-1;
            trmat(2,1)=1;

            dir= trmat*dir;
        }
        if(method_struct.SliceOrient=="sagittal")
        {
            ImageType4DITK::DirectionType  trmat;
            trmat.Fill(0); trmat(3,3)=1;
            trmat(0,0)=1;
            trmat(1,2)=1;
            trmat(2,1)=-1;

            dir= trmat*dir;
        }
    }

    importFilter->SetDirection(dir);
    importFilter->SetImportPointer( image_data, total_els, true);
    importFilter->Update();
    ImageType4DITK::Pointer dwis= importFilter->GetOutput();


    double mdet= vnl_determinant<double>(dir.GetVnlMatrix());
    if(fabs(mdet+1)<0.1)
    {
        vnl_matrix<double> flipM(4,4);
        flipM.set_identity();
        flipM(phase_vector[0],phase_vector[0])=-1;


        vnl_matrix<double> new_dir = dwis->GetDirection().GetVnlMatrix() *flipM;

        ImageType4DITK::Pointer dwis2=ImageType4DITK::New();
        dwis2->SetRegions(dwis->GetLargestPossibleRegion());
        dwis2->Allocate();
        dwis2->SetSpacing(dwis->GetSpacing());
        dwis2->SetOrigin(dwis->GetOrigin());
        dwis2->SetDirection(new_dir);

        itk::ImageRegionIteratorWithIndex<ImageType4DITK> it(dwis2,dwis2->GetLargestPossibleRegion());
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            ImageType4DITK::IndexType index =it.GetIndex();
            index[phase_vector[0]]= dwis->GetLargestPossibleRegion().GetSize()[phase_vector[0]]-1 -index[phase_vector[0]];
            it.Set(dwis->GetPixel(index));
            ++it;
        }
        dwis=dwis2;
    }

    return dwis;
}





VISU_struct ConvertBruker::read_visu_file(std::string visu_file)
{

    VISU_struct visu_struct;

    std::ifstream infile(visu_file.c_str());
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


            if(var_name==std::string("VISUCOREDIM"))
            {
                visu_struct.dim=atoi(line.substr(temp_pos2+1).c_str());
            }
            if(var_name==std::string("VISUCORESIZE"))
            {
                int sz=atoi(line.substr(temp_pos2+2).c_str());
                for(int v=0;v<sz;v++)
                    infile>>visu_struct.size[v];
            }
            if(var_name==std::string("VISUCOREEXTENT"))
            {
                int sz=atoi(line.substr(temp_pos2+2).c_str());
                for(int v=0;v<sz;v++)
                {
                    float dum;
                    infile>>dum;
                    visu_struct.extent[v]=dum;
                }
            }
            if(var_name==std::string("VISUCOREORIENTATION"))
            {
                for(int v1=0;v1<3;v1++)
                    for(int v2=0;v2<3;v2++)
                        infile>>visu_struct.orientation(v1,v2);
            }
            if(var_name==std::string("VISUCOREPOSITION"))
            {
                infile>>visu_struct.pos[0];
                infile>>visu_struct.pos[1];
                infile>>visu_struct.pos[2];
            }

            if(var_name==std::string("VISUCOREDATAOFFS"))
            {
                std::getline(infile, line);
                if(line.at(0)=='@')
                {
                    std::string ok_temp= line.substr(line.find('(')+1);
                    visu_struct.intercept= atof(ok_temp.c_str());

                }
                else
                {
                    visu_struct.intercept= atof(line.c_str());
                }
            }
            if(var_name==std::string("VISUCOREDATASLOPE"))
            {
                std::getline(infile, line);
                if(line.at(0)=='@')
                {
                    std::string ok_temp= line.substr(line.find('(')+1);
                    visu_struct.slope= atof(ok_temp.c_str());

                }
                else
                {
                    visu_struct.slope= atof(line.c_str());
                }
            }
            if(var_name==std::string("VISUCOREWORDTYPE"))
            {
                visu_struct.word_type=line.substr(temp_pos2+1);
            }
            if(var_name==std::string("VISUCORESLICEPACKSSLICES"))
            {
                int nlines=atoi(line.substr(temp_pos2+2).c_str());
                int sm=0;
                for(int l=0;l<nlines;l++)
                {
                    std::getline(infile, line);
                    std::string ok_temp= line.substr(line.find(',')+1);
                    sm+= atoi(ok_temp.c_str());
                }
                visu_struct.nslices=sm;
            }
            if(var_name==std::string("VISUCORESLICEPACKSSLICEDIST"))
            {
                std::getline(infile, line);
                visu_struct.slice_thickness= atof(line.c_str());
            }
            if(var_name==std::string("VISUSUBJECTPOSITION"))
            {
                visu_struct.subject_position=line.substr(temp_pos2+1);
            }
        }
    }


    infile.close();


    return visu_struct;
}











METHOD_struct ConvertBruker::read_method_file(std::string method_file)
{

    METHOD_struct method_struct;
    method_struct.PVM_EPI_BlipDir=0;
    method_struct.RG=1;

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



            if(var_name==std::string("PHASEMODE"))
            {
                method_struct.phase_mode=line.substr(temp_pos2+1);
            }

            if(var_name==std::string("PVM_NAVERAGES"))
            {
                method_struct.N_averages=atoi(line.substr(temp_pos2+1).c_str());
            }
            if(var_name==std::string("PVM_NREPETITIONS"))
            {
                method_struct.N_reps=atoi(line.substr(temp_pos2+1).c_str());
            }
            if(var_name==std::string("PVM_REPETITIONTIME"))
            {
                method_struct.TR=atof(line.substr(temp_pos2+1).c_str());
            }
            if(var_name==std::string("PVM_SPACKARRREADORIENT"))
            {
                std::getline(infile, line);
                method_struct.ReadOrient=line;
            }
            if(var_name==std::string("PVM_SPACKARRSLICEORIENT"))
            {
                std::getline(infile, line);
                method_struct.SliceOrient=line;
            }
            if(var_name==std::string("PVM_DWAOIMAGES"))
            {
                method_struct.N_A0=atoi(line.substr(temp_pos2+1).c_str());
            }
            if(var_name==std::string("PVM_DWNDIFFEXPEACH"))
            {
                method_struct.NdiffExp=atoi(line.substr(temp_pos2+1).c_str());
            }
            if(var_name==std::string("PVM_DWNDIFFDIR"))
            {
                method_struct.Ndir=atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name==std::string("PVM_RGVALUE"))
            {
                method_struct.RG=atof(line.substr(temp_pos2+1).c_str());
            }
            if(var_name==std::string("PVM_DWGRADDUR"))
            {
                infile>>method_struct.small_delta;
            }
            if(var_name==std::string("PVM_DWGRADSEP"))
            {
                infile>>method_struct.BIG_DELTA;
            }
            if(var_name==std::string("PVM_DWEFFBVAL"))
            {
                method_struct.N_totalvol = atoi(line.substr(temp_pos2+2).c_str());
                method_struct.eff_bval.set_size(method_struct.N_totalvol);
                infile>>method_struct.eff_bval;
            }
            if(var_name==std::string("PVM_DWGRADPHASE"))
            {
                method_struct.N_totalvol = atoi(line.substr(temp_pos2+2).c_str());
                method_struct.grad_phase.set_size(method_struct.N_totalvol);
                infile>>method_struct.grad_phase;
            }
            if(var_name==std::string("PVM_DWGRADREAD"))
            {
                method_struct.N_totalvol = atoi(line.substr(temp_pos2+2).c_str());
                method_struct.grad_read.set_size(method_struct.N_totalvol);
                infile>>method_struct.grad_read;
            }
            if(var_name==std::string("PVM_DWGRADSLICE"))
            {
                method_struct.N_totalvol = atoi(line.substr(temp_pos2+2).c_str());
                method_struct.grad_slice.set_size(method_struct.N_totalvol);
                infile>>method_struct.grad_slice;
            }
            //if(var_name==std::string("PVM_DWBMAT"))
            if(var_name==std::string("PVM_DWBMATIMAG"))
            {
                int ndwis = atoi(line.substr(temp_pos2+2).c_str());
                vnl_matrix<double> br_bmatrix(ndwis,9);
                infile>>br_bmatrix;

                method_struct.Bmatrix.set_size(ndwis,6);
                for(int i=0;i<ndwis;i++)
                {
                    method_struct.Bmatrix(i,0)=  br_bmatrix(i,0);
                    method_struct.Bmatrix(i,1)=2*br_bmatrix(i,1);
                    method_struct.Bmatrix(i,2)=2*br_bmatrix(i,2);
                    method_struct.Bmatrix(i,3)=  br_bmatrix(i,4);
                    method_struct.Bmatrix(i,4)=2*br_bmatrix(i,5);
                    method_struct.Bmatrix(i,5)=  br_bmatrix(i,8);
                }
            }
            if(var_name=="PVM_EPI_BLIPDIR")
            {
                method_struct.PVM_EPI_BlipDir=atoi(line.substr(temp_pos2+1).c_str());
            }

            if(var_name=="PVM_SPACKARRGRADORIENT")
            {
                infile >> method_struct.grad_orient;
            }
        }
    }


    infile.close();

    return method_struct;
}


RECO_struct ConvertBruker::read_reco_header(std::string reco_file)
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


