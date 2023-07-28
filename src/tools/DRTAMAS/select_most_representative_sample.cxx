

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
using namespace std;



#include "DRTAMAS_utilities_cp.h"


           
    
double ComputeImageDistances(DTMatrixImageType::Pointer image1,DTMatrixImageType::Pointer image2)
{
    ImageType3D::SizeType imsize= image1->GetLargestPossibleRegion().GetSize();

    ImageType3D::IndexType index_im;
    
    double sm=0;
    
    DTMatrixImageType::PixelType id; id.set_identity();

    for(int k=0;k<imsize[2];k++)
    {
        index_im[2]=k;
        for(int j=0;j<imsize[1];j++)
        {
            index_im[1]=j;
            for(int i=0;i<imsize[0];i++)
            {
                index_im[0]=i;


                DTMatrixImageType::PixelType mat1= image1->GetPixel(index_im);
                DTMatrixImageType::PixelType mat2= image2->GetPixel(index_im);

                double mat1_TR= (mat1(0,0)+mat1(1,1)+mat1(2,2))/3.;
                double mat2_TR= (mat2(0,0)+mat2(1,1)+mat2(2,2))/3.;

                mat1= mat1- mat1_TR * id;
                mat2= mat2- mat2_TR * id;

                auto diff= mat1- mat2;
                sm+= diff.frobenius_norm();
            }
        }
    }

    return sm;
    
}

int main( int argc , char * argv[] )
{
    if(argc<3)    
    {
        std::cout<<"Usage:   SelectMostRepresentativeSample   full_path_to_textfile_containing_list_of_tensors  full_name_of_output_image"<<std::endl;
        return 0;
    }
       
    
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
        return 0;
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
                return 0;
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
                

    std::vector<DTMatrixImageType::Pointer> image_list;
    image_list.resize(N_images);
    
    for(int i=0; i<N_images;i++)
    {
        std::cout<<filenames[i]<<std::endl;
        image_list[i]=ReadAndOrientTensor(filenames[i]);
    }
    
     
    vnl_matrix<double> pairwise_distances(N_images,N_images);
    pairwise_distances.fill(0);   
    
    #pragma omp parallel for     
    for(int i=0; i<N_images-1;i++)
    {
        for(int j=i+1;j<N_images;j++)
        {
            pairwise_distances(i,j) =  ComputeImageDistances(image_list[i],image_list[j]);                
        }
    }

    pairwise_distances= pairwise_distances+pairwise_distances.transpose();

    double min_dist=std::numeric_limits<double>::max();
    int min_id=-1;
    for(int i=0; i<N_images;i++)
    {
        double sm=0;
        for(int j=0;j<N_images;j++)
        {
            sm+=   pairwise_distances(i,j);
        }

        if(sm<min_dist)
        {
            min_dist=sm;
            min_id=i;                
        }  
    }




    DTMatrixImageType::Pointer template_image= image_list[min_id];
    std::string output_name(argv[2]);    
    OrientAndWriteTensor(template_image,output_name);

    
    std::cout<< " The chosen image for rigid template: " << filenames[min_id] << std::endl;
    
    return EXIT_SUCCESS;
}
