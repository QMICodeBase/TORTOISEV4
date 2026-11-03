

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <fstream>
using namespace std;



#include "defines.h"

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good(); // Returns true if the file was opened successfully
}


float median3(std::vector<float> &v)
{
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}


float calculate_mad( std::vector<float>& data)
{
    if (data.empty()) return 0.0;

    // Step 1: Compute the mean
    double mean = median3(data);

    // Step 2: Compute the sum of absolute deviations from the mean
    double absolute_deviation_sum = 0.0;
    for (double val : data) {
        absolute_deviation_sum += std::abs(val - mean);
    }

    // Step 3: Return the average absolute deviation
    return absolute_deviation_sum / data.size();
}

bool IsLabelNew(std::vector<int> &labels, int label)
{
    if(label==0)
        return false;

    for(int i=0;i<labels.size();i++)
        if(labels[i]==label)
            return false;

    return true;
}


int main( int argc , char * argv[] )
{
    if(argc<2)    
    {
        std::cout<< "ComputeROIStatistic main_image  ROI_image statistic_type (0: mean, 1: stdev, 2:median, 3: MAD) output_csv_name final_mathematical_operator (1: do nothing, exp, log)(optional. default:1 ) category_string(optional)"<<std::endl;
        return EXIT_FAILURE;
    }
    

    std::string iname= argv[1];
    std::string ROI_name = argv[2];


    ImageType3D::Pointer input_img = readImageD<ImageType3D>(iname);
    ImageType3D::Pointer ROI_img = readImageD<ImageType3D>(ROI_name);

    std::vector<int> labels;
    int label_id=-1;
    std::map<int,int> mapping;
    {
        itk::ImageRegionIteratorWithIndex<ImageType3D> it(ROI_img,ROI_img->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            int label = it.Get();
            bool is_new_label = IsLabelNew(labels,label);
            if(is_new_label)
            {
                labels.push_back(label);
                label_id++;
                mapping[label]=label_id;
            }
        }
    }
    std::sort(labels.begin(), labels.end());
    int Nlabels=labels.size();


    std::vector< std::vector<float> >  data;
    data.resize(Nlabels);
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(input_img,input_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        int clabel = ROI_img->GetPixel(ind3);
        if(clabel !=0)
        {
            int mp = mapping.at(clabel);
            data[mp].push_back(it.Get());
        }
    }

    int stat_type = atoi(argv[3]);

    std::string oper="";
    if(argc>=6)
        oper=argv[5];

    std::vector<float> stat_data;
    stat_data.resize(Nlabels);

    if(stat_type==0)
    {
        for(int l=0;l<Nlabels;l++)
        {
            double sm=0;
            for(int d=0;d<data[l].size();d++)
                sm+=data[l][d];
            sm/=data[l].size();
            if(oper=="exp")
                sm=exp(sm);
            if(oper=="log")
                sm=log(sm);
            stat_data[l]=sm;
        }
    }

    if(stat_type==1)
    {
        for(int l=0;l<Nlabels;l++)
        {
            double sm=0;
            for(int d=0;d<data[l].size();d++)
                sm+=data[l][d];
            sm/=data[l].size();
            double mn=sm;

            sm=0;
            for(int d=0;d<data[l].size();d++)
            {
                sm+= (data[l][d] - mn) * (data[l][d] - mn);
            }
            sm/= data[l].size() -1;
            sm=sqrt(sm);
            if(oper=="exp")
                sm=exp(sm);
            if(oper=="log")
                sm=log(sm);

            stat_data[l]=sm;

        }
    }

    if(stat_type==2)
    {
        for(int l=0;l<Nlabels;l++)
        {
            double sm =median3(data[l]);
            if(oper=="exp")
                sm=exp(sm);
            if(oper=="log")
                sm=log(sm);
            stat_data[l] = sm;
        }
    }

    if(stat_type==3)
    {
        for(int l=0;l<Nlabels;l++)
        {
            double sm=calculate_mad(data[l]);
            if(oper=="exp")
                sm=exp(sm);
            if(oper=="log")
                sm=log(sm);
            stat_data[l] = sm;
        }
    }


    std::string categ="";
    if(argc>=7)
        categ=argv[6];

    std::string oname= argv[4];

    size_t pos = iname.find_last_of("/\\");
    if (pos != std::string::npos)
    {
        iname=iname.substr(pos + 1);
    }

    if(fileExists(oname))
    {
        FILE *fp = fopen(oname.c_str(),"a");                

        fprintf(fp,"%s,",iname.c_str());
        if(categ!="")
            fprintf(fp,"%s,",categ.c_str());
        for(int l=0;l<Nlabels-1;l++)
            fprintf(fp,"%f,",stat_data[l]);
        fprintf(fp,"%f\n",stat_data[Nlabels-1]);
        fclose(fp);
    }
    else
    {
        FILE *fp = fopen(oname.c_str(),"w");
        fprintf(fp,"ImageName,");
        if(categ!="")
            fprintf(fp,"Category,");
        for(int l=0;l<Nlabels-1;l++)
            fprintf(fp,"ROI%d,",labels[l]);
        fprintf(fp,"ROI%d\n",labels[Nlabels-1]);

        fprintf(fp,"%s,",iname.c_str());
        if(categ!="")
            fprintf(fp,"%s,",categ.c_str());

        for(int l=0;l<Nlabels-1;l++)
            fprintf(fp,"%f,",stat_data[l]);
        fprintf(fp,"%f\n",stat_data[Nlabels-1]);
        fclose(fp);
    }

    
    

    
    return EXIT_SUCCESS;
}
