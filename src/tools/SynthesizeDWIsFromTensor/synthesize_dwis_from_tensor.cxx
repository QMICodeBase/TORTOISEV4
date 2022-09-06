#include "defines.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/write_3D_image_to_4D_file.h"
#include "../tools/EstimateTensor/DTIModel.h"


int main(int argc,char *argv[])
{
    if(argc<5)
    {
        std::cout<<"Usage: SynthesizeDWIsFromTensor full_path_to_tensor_image full_path_to_A0_image full_path_to_synthesization_Bmatrix output_filename"<<std::endl;
        return 0;
    }


    typedef itk::ImageFileReader<ImageType4D> TensorReaderType;
    TensorReaderType::Pointer tensor_reader=TensorReaderType::New();
    tensor_reader->SetFileName(argv[1]);
    tensor_reader->Update();
    ImageType4D::Pointer tensor_image_4D_itk= tensor_reader->GetOutput();


    ImageType3D::Pointer A0_image= readImageD<ImageType3D>(argv[2]);

    vnl_matrix<double> Bmatrix = read_bmatrix_file(argv[3]);


    DTImageType::Pointer dt_image= DTImageType::New();
    dt_image->SetRegions(A0_image->GetLargestPossibleRegion());
    dt_image->Allocate();
    dt_image->SetSpacing(A0_image->GetSpacing());
    dt_image->SetOrigin(A0_image->GetOrigin());    
    dt_image->SetDirection(A0_image->GetDirection());


    ImageType3D::SizeType imsize = A0_image->GetLargestPossibleRegion().GetSize();


    ImageType4D::IndexType ind;
    ImageType3D::IndexType index;

    for(int k=0;k<imsize[2];k++)
    {
        ind[2]=k;
        index[2]=k;
        for(int j=0;j<imsize[1];j++)
        {
            ind[1]=j;
            index[1]=j;
            for(int i=0;i<imsize[0];i++)
            {
                ind[0]=i;
                index[0]=i;

                DTType curr_tens;

                    ind[3]=0;
                    curr_tens[0]= tensor_image_4D_itk->GetPixel(ind)/1000000.;
                    ind[3]=1;
                    curr_tens[3]= tensor_image_4D_itk->GetPixel(ind)/1000000.;
                    ind[3]=2;
                    curr_tens[5]= tensor_image_4D_itk->GetPixel(ind)/1000000.;
                    ind[3]=3;
                    curr_tens[1]= tensor_image_4D_itk->GetPixel(ind)/1000000.;
                    ind[3]=4;
                    curr_tens[2]= tensor_image_4D_itk->GetPixel(ind)/1000000.;
                    ind[3]=5;
                    curr_tens[4]= tensor_image_4D_itk->GetPixel(ind)/1000000.;


                dt_image->SetPixel(index,curr_tens);
            }
        }
    }    



    DTIModel dti_estimator;
    dti_estimator.SetOutput(dt_image);
    dti_estimator.SetA0Image(A0_image);


    int Nvols= Bmatrix.rows();

    for(int i=0;i<Nvols;i++)
    {
        vnl_vector<double> bmat_vec= Bmatrix.get_row(i);

        ImageType3D::Pointer dwi= dti_estimator.SynthesizeDWI( bmat_vec );

        write_3D_image_to_4D_file<float>(dwi,argv[4],i,Nvols);
    }
}
