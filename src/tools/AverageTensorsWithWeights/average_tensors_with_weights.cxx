

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
using namespace std;

#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

//#include "defines.h"
//#include "../tools/DRTAMAS/DRTAMAS_utilities_cp.cxx"


using  InternalMatrixType=vnl_matrix_fixed< float, 3, 3 >;
using DTMatrixImageType = itk::Image<InternalMatrixType,3>;
using ImageType3D = itk::Image<float,3>;
using ImageType4D = itk::Image<float,4>;

float median2(vector<float> &v)
{
    if(v.size()==1)
    {
        return v[0];
    }
    
    if(v.size()==2)
    {
        return 0.5*(v[0]+v[1]);
    }
    
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}


float avg2 ( vector<float>& v )
{
        float return_value = 0.0;
        int n = v.size();
       
        for ( int i=0; i < n; i++)
        {
            return_value += v[i];
        }
       
        return ( return_value /n);
}


int main( int argc , char * argv[] )
{
    if(argc<5)    
    {
        std::cout<<"Usage:   AverageTensorsWithWeights   full_path_to_textfile_containing_list_of_tensors  name_of_output_image iteration total_iteration"<<std::endl;
        return EXIT_FAILURE;
    }
       
    int iter= atoi(argv[3]); 
    int total_iter= atoi(argv[4]); 
    float iter_perc= 1.0*iter/(total_iter);
    
    int N_images=0;
    
    std::string currdir;
    std::string nm(argv[1]);
    
            
    int mypos= nm.rfind("/");
    if(mypos ==-1)
        currdir= std::string("./");
    else   
       currdir= nm.substr(0,mypos+1);
    
    
    ifstream inFile(argv[1]);
    if (!inFile) 
    {
        cerr << "File " << argv[1] << " not found." << endl;
        return EXIT_FAILURE;
    }
    
    std::vector<std::string> filenames;
    string line;
    while (getline(inFile, line)) 
    {
        if (line.empty()) 
            continue;
        
        std::string file_name=line;
        FILE * fp= fopen(file_name.c_str(),"rb");
        
        if(!fp)
        {
            file_name= currdir + file_name;
            
            FILE * fp2= fopen(file_name.c_str(),"rb");
            if(!fp2)
            {            
                std::cout<< "File " << line << " does not exist. Exiting!" << std::endl;
                return EXIT_FAILURE;
            }
            else
               fclose(fp2);
        }
        else
            fclose(fp);
        
        
        filenames.push_back(file_name);
        N_images++; 
    }        
    inFile.close();
    
    
    
    
    std::vector<ImageType4D::Pointer> image_list;
    image_list.resize(N_images);    
    for(int i=0; i<N_images;i++)
    {
        using DTReaderType= itk::ImageFileReader<ImageType4D>;
        DTReaderType::Pointer reader= DTReaderType::New();
        reader->SetFileName(filenames[i]);
        reader->Update();
        ImageType4D::Pointer tensor4d= reader->GetOutput();
        image_list[i]= tensor4d;
    }
           
    ImageType4D::SizeType imsize= image_list[0]->GetLargestPossibleRegion().GetSize();
    

    ImageType4D::Pointer template_image= ImageType4D::New();
    template_image->SetRegions(image_list[0]->GetLargestPossibleRegion());
    template_image->Allocate();
    template_image->SetSpacing(image_list[0]->GetSpacing());
    template_image->SetOrigin(image_list[0]->GetOrigin());
    template_image->SetDirection(image_list[0]->GetDirection());
    template_image->FillBuffer(0);
    

    ImageType3D::SizeType sz;
    sz[0]= image_list[0]->GetLargestPossibleRegion().GetSize()[0];
    sz[1]= image_list[0]->GetLargestPossibleRegion().GetSize()[1];
    sz[2]= image_list[0]->GetLargestPossibleRegion().GetSize()[2];
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType reg(start,sz);

    ImageType3D::SpacingType spc;
    spc[0]= image_list[0]->GetSpacing()[0];
    spc[1]= image_list[0]->GetSpacing()[1];
    spc[2]= image_list[0]->GetSpacing()[2];

    ImageType3D::PointType orig;
    orig[0]= image_list[0]->GetOrigin()[0];
    orig[1]= image_list[0]->GetOrigin()[1];
    orig[2]= image_list[0]->GetOrigin()[2];

    ImageType4D::DirectionType dir4d = image_list[0]->GetDirection();
    ImageType3D::DirectionType dir;
    dir(0,0)=dir4d(0,0);dir(0,1)=dir4d(0,1);dir(0,2)=dir4d(0,2);
    dir(1,0)=dir4d(1,0);dir(1,1)=dir4d(1,1);dir(1,2)=dir4d(1,2);
    dir(2,0)=dir4d(2,0);dir(2,1)=dir4d(2,1);dir(2,2)=dir4d(2,2);


    
    ImageType3D::Pointer median_image= ImageType3D::New();
    median_image->SetRegions(reg);
    median_image->Allocate();
    median_image->SetSpacing(spc);
    median_image->SetOrigin(orig);
    median_image->SetDirection(dir);
    median_image->FillBuffer(0);
   
    #pragma omp parallel for
    for(int k=0;k<imsize[2];k++)
    {        
        ImageType3D::IndexType index3D;             
        index3D[2]=k;
        ImageType4D::IndexType ind4;
        ind4[2]=k;

        for(int j=0;j<imsize[1];j++)
        {            
            index3D[1]=j;
            ind4[1]=j;
            for(int i=0;i<imsize[0];i++)
            {                     
                index3D[0]=i;
                ind4[0]=i;
                
                std::vector<float> median_images;
                median_images.resize(N_images);
                
                for(int im=0;im<N_images;im++)
                {
                    float trace=0;
                    ind4[3]=0;
                    trace+=image_list[im]->GetPixel(ind4);
                    ind4[3]=1;
                    trace+=image_list[im]->GetPixel(ind4);
                    ind4[3]=2;

                    median_images[im]=trace;
                }
                
                float median_trace= median2(median_images);
                median_image->SetPixel(index3D,median_trace);
            }
        }
    }
    
    
    typedef itk::DiscreteGaussianImageFilter<ImageType3D, ImageType3D> ImageSmoothingFilterType;
    ImageSmoothingFilterType::Pointer SmoothingFilterf = ImageSmoothingFilterType::New();
    SmoothingFilterf->SetUseImageSpacingOn();
    SmoothingFilterf->SetVariance( (1-iter_perc)*(1-iter_perc)*0.6 );
    SmoothingFilterf->SetMaximumError( 0.01 );
    SmoothingFilterf->SetInput( median_image);
    SmoothingFilterf->Update();
    median_image=SmoothingFilterf->GetOutput();
    
    
   
    #pragma omp parallel for
    for(int k=0;k<imsize[2];k++)
    {       
        ImageType3D::IndexType index3D;                
        index3D[2]=k;
        ImageType4D::IndexType ind4;
        ind4[2]=k;

        for(int j=0;j<imsize[1];j++)
        {            
            index3D[1]=j;
            ind4[1]=j;
            for(int i=0;i<imsize[0];i++)
            {                     
                index3D[0]=i;
                ind4[0]=i;
                                
                std::vector<float> weights;
                weights.resize(N_images);
                
                if(N_images==2)
                {
                    weights[0]=0.5;
                    weights[1]=0.5;                    
                }
                else
                {
                    if(N_images==3)
                    {
                        weights[0]=0.33333333334;
                        weights[1]=0.33333333333;
                        weights[2]=0.33333333333;
                    }
                    else
                    {
                        float median_trace=median_image->GetPixel(index3D);


                        float max=-1;
                        for(int im=0;im<N_images;im++)
                        {

                            float trace=0;
                            ind4[3]=0;
                            trace+=image_list[im]->GetPixel(ind4);
                            ind4[3]=1;
                            trace+=image_list[im]->GetPixel(ind4);
                            ind4[3]=2;

                            weights[im]= fabs(trace-median_trace);
                            if(weights[im]>max)
                                max=weights[im];

                        }

                        float sm=0;
                        for(int im=0;im<N_images;im++)
                        {
                            if(max<4)
                            {
                                weights[im]=1.;
                                sm++;
                            }
                            else
                            {
                                float x=weights[im];
                                if(iter_perc <0.5)
                                {

                                    float deg= total_iter -iter*2;

                                    weights[im]= pow(-x/max +1,deg);                            
                                }
                                else
                                {
                                    float deg= std::max(2*iter-total_iter,1);
                                    weights[im]= 1- pow(x/max,deg);                            
                                }

                               // weights[im]=  exp(-alpha *weights[im]);
                                //weights[im]= -0.5/max * weights[im] +1;                        
                                sm+=weights[im];
                            }
                        }


                        for(int im=0;im<N_images;im++)
                        {
                            if(sm <0.00000001)
                                weights[im]=1./N_images;
                            else
                                weights[im]/=sm;                    
                        }
                        
                    }                    
                }


                for(int v=0;v<6;v++)
                {
                    ind4[3]=v;
                    double val=0;
                    for(int im=0;im<N_images;im++)
                    {
                        val+= weights[im]*image_list[im]->GetPixel(ind4);
                    }
                    template_image->SetPixel(ind4,val);
                }

            }
        }
    }

    std::string output_name(argv[2]);
    using WrType=itk::ImageFileWriter<ImageType4D>;
    WrType::Pointer wr = WrType::New();
    wr->SetFileName(output_name);
    wr->SetInput(template_image);
    wr->Update();
    
    return EXIT_SUCCESS;
}
