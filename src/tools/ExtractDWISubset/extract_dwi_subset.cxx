#include "extract_dwi_subset_parser.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "../utilities/read_bmatrix_file.h"

#include "defines.h"




int main(int argc, char * argv[])
{
    Extract_DWI_Subset_PARSER *parser = new Extract_DWI_Subset_PARSER(argc,argv);

    std::string oname;
    if(parser->getOutputImageName()=="")
    {
        std::string nm=parser->getInputImageName();
        oname=nm.substr(0,nm.find(".nii")) + std::string("_DR.nii");
    }
    else
    {
        oname=parser->getOutputImageName();
    }

    std::string nii_file= parser->getInputImageName();
    std::string bmtxt_file = nii_file.substr(0,nii_file.find(".nii"))+ ".bmtxt";


    std::string bvals_list= parser->getBvalsString();
    std::string vols_list= parser->getVolIdsString();
    remove_if(bvals_list.begin(), bvals_list.end(), isspace);
    remove_if(vols_list.begin(), vols_list.end(), isspace);

    vnl_matrix<double> Bmatrix= read_bmatrix_file(bmtxt_file);
    int nvols = Bmatrix.rows();
    std::vector<bool> pickits;
    pickits.resize(nvols);

    std::vector<float> bvals;
    bvals.resize(nvols);
    for(int i=0;i<nvols;i++)
    {
        bvals[i]= Bmatrix(i,0)+ Bmatrix(i,3)+Bmatrix(i,5);
    }


    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;

    if(bvals_list !="")
    {
        while ((pos = bvals_list.find(delimiter)) != std::string::npos)
        {
            token = bvals_list.substr(0, pos);
            if(token.find('-')!=std::string::npos)
            {
                size_t npos= token.find('-');
                float firstb=  atof(token.substr(0,npos).c_str());
                float secondb= atof(token.substr(npos+1).c_str());

                for(int i=0;i<nvols;i++)
                {
                    float currb= bvals[i];

                    if(currb >= 0.95* firstb  && currb <= 1.05* secondb)
                        pickits[i]=1;
                }
            }
            else
            {
                float curr_b= atof(token.c_str());
                for(int i=0;i<nvols;i++)
                {
                    if( fabs(curr_b- bvals[i]) <= 0.1*bvals[i] )
                        pickits[i]=1;
                }
            }
            bvals_list.erase(0, pos + delimiter.length());
        }
        token= bvals_list;
        if(token.find('-')!=std::string::npos)
        {
            size_t npos= token.find('-');
            float firstb=  atof(token.substr(0,npos).c_str());
            float secondb= atof(token.substr(npos+1).c_str());

            for(int i=0;i<nvols;i++)
            {
                float currb= bvals[i];

                if(currb >= 0.95* firstb  && currb <= 1.05* secondb)
                    pickits[i]=1;
            }
        }
        else
        {
            float curr_b= atof(token.c_str());
            for(int i=0;i<nvols;i++)
            {
                if( fabs(curr_b- bvals[i]) <= 0.1*bvals[i] )
                    pickits[i]=1;
            }
        }
    }
    else
    {
        while ((pos = vols_list.find(delimiter)) != std::string::npos)
        {
            token = vols_list.substr(0, pos);
            if(token.find('-')!=std::string::npos)
            {
                size_t npos= token.find('-');
                int firstv=  atoi(token.substr(0,npos).c_str());
                int secondv= atoi(token.substr(npos+1).c_str());
                for(int i=firstv;i<=secondv;i++)
                    pickits[i]=1;
            }
            else
            {
                int v= atoi(token.c_str());
                pickits[v]=1;

            }
            vols_list.erase(0, pos + delimiter.length());
        }
        token=vols_list;
        if(token.find('-')!=std::string::npos)
        {
            size_t npos= token.find('-');
            int firstv=  atoi(token.substr(0,npos).c_str());
            int secondv= atoi(token.substr(npos+1).c_str());
            for(int i=firstv;i<=secondv;i++)
                pickits[i]=1;
        }
        else
        {
            int v= atoi(token.c_str());
            pickits[v]=1;

        }
    }


    int new_Nvolumes=0;
    for(int i=0;i<nvols;i++)
        new_Nvolumes+= (int)pickits[i];


    vnl_matrix<double> new_Bmatrix;
    new_Bmatrix.set_size(new_Nvolumes,6);
    new_Bmatrix.fill(0);


    std::string new_nii_name= oname;
    std::string new_bmtxt_name= new_nii_name.substr(0,new_nii_name.rfind(".nii")) + ".bmtxt";

    int new_index=-1;
    for(int i=0;i<nvols;i++)
    {
        if(pickits[i])
        {
            new_index++;
            ImageType3D::Pointer img= read_3D_volume_from_4D(nii_file,i);
            write_3D_image_to_4D_file<float>(img,new_nii_name,new_index,new_Nvolumes);
            //write_3D_image_to_4D_file<float>(img,new_nii_name,new_index,new_Nvolumes);

            new_Bmatrix.set_row(new_index, Bmatrix.get_row(i));
        }
    }

    std::ofstream outfile(new_bmtxt_name);
    outfile<<new_Bmatrix;
    outfile.close();



    return EXIT_SUCCESS;
}
