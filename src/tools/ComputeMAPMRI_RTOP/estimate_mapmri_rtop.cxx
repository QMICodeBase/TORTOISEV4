#include "defines.h"


#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "../tools/EstimateMAPMRI/MAPMRIModel.h"

int main(int argc, char *argv[])
{

    if(argc<2)
    {
        std::cout<<"EstimateMAPMRI_RTOP full_path_to_MAPMRI_coeff_image full_path_to_MAPMRI_uvec_image"<<std::endl;
        return 0;
    }

    std::string filename(argv[1]);

    typedef itk::VectorImage<float, 3>  VecImageType;
    typedef itk::ImageFileReader<VecImageType> VecReaderType;
    VecReaderType::Pointer reader = VecReaderType::New();
    reader->SetFileName(filename);
    reader->Update();
    VecImageType::Pointer coeff_image= reader->GetOutput();


    std::string filename2(argv[2]);
    typedef itk::ImageFileReader<MAPMRIModel::EValImageType> EvalReaderType;
    EvalReaderType::Pointer readere = EvalReaderType::New();
    readere->SetFileName(filename2);
    readere->Update();
    MAPMRIModel::EValImageType::Pointer eval_image= readere->GetOutput();


    int Ncoeffs =coeff_image->GetNumberOfComponentsPerPixel();

    int order;
    for(order=2;order<=10;order+=2)
    {
        int nc= (order/2+1)*(order/2+2)*(2*order+3)/6;
        if(nc==Ncoeffs)
            break;
    }


    ImageType3D::Pointer RTOP_image = ImageType3D::New();
    RTOP_image->SetRegions(coeff_image->GetLargestPossibleRegion());
    RTOP_image->Allocate();
    RTOP_image->SetOrigin(coeff_image->GetOrigin());
    RTOP_image->SetSpacing(coeff_image->GetSpacing());
    RTOP_image->SetDirection(coeff_image->GetDirection());
    RTOP_image->FillBuffer(0);

    ImageType3D::Pointer RTAP_image = ImageType3D::New();
    RTAP_image->SetRegions(coeff_image->GetLargestPossibleRegion());
    RTAP_image->Allocate();
    RTAP_image->SetOrigin(coeff_image->GetOrigin());
    RTAP_image->SetSpacing(coeff_image->GetSpacing());
    RTAP_image->SetDirection(coeff_image->GetDirection());
    RTAP_image->FillBuffer(0);

    ImageType3D::Pointer RTPP_image = ImageType3D::New();
    RTPP_image->SetRegions(coeff_image->GetLargestPossibleRegion());
    RTPP_image->Allocate();
    RTPP_image->SetOrigin(coeff_image->GetOrigin());
    RTPP_image->SetSpacing(coeff_image->GetSpacing());
    RTPP_image->SetDirection(coeff_image->GetDirection());
    RTPP_image->FillBuffer(0);



    int n1ap[]={0,2,0,0,1,1,0,4,0,0,3,3,1,1,0,0,2,2,0,2,1,1,6,0,0,5,5,1,1,0,0,4,4,2,2,0,0,4,1,1,3,3,0,3,3,2,2,1,1,2,8,0,0,7,7,1,
           1,0,0,6,6,2,2,0,0,6,1,1,5,5,3,3,0,0,5,5,2,2,1,1,4,4,0,4,4,3,3,1,1,4,2,2,3,3,2,10,0,0,9,9,1,1,0,0,8,8,2,2,0,0,8,1,
           1,7,7,3,3,0,0,7,7,2,2,1,1,6,6,4,4,0,0,6,6,3,3,1,1,6,2,2,5,5,0,5,5,4,4,1,1,5,5,3,3,2,2,4,4,2,4,3,3};

      int n2ap[]={0,0,2,0,1,0,1,0,4,0,1,0,3,0,3,1,2,0,2,1,2,1,0,6,0,1,0,5,0,5,1,2,0,4,0,4,2,1,4,1,3,0,3,2,1,3,1,3,2,2,0,8,0,1,0,7,0,
           7,1,2,0,6,0,6,2,1,6,1,3,0,5,0,5,3,2,1,5,1,5,2,4,0,4,3,1,4,1,4,3,2,4,2,3,2,3,0,10,0,1,0,9,0,9,1,2,0,8,0,8,2,1,8,1,3,
           0,7,0,7,3,2,1,7,1,7,2,4,0,6,0,6,4,3,1,6,1,6,3,2,6,2,5,0,5,4,1,5,1,5,4,3,2,5,2,5,3,4,2,4,3,4,3};

      int n3ap[]={0,0,0,2,0,1,1,0,0,4,0,1,0,3,1,3,0,2,2,1,1,2,0,0,6,0,1,0,5,1,5,0,2,0,4,2,4,1,1,4,0,3,3,1,2,1,3,2,3,2,0,0,8,0,1,
           0,7,1,7,0,2,0,6,2,6,1,1,6,0,3,0,5,3,5,1,2,1,5,2,5,0,4,4,1,3,1,4,3,4,2,2,4,2,3,3,0,0,10,0,1,0,9,1,9,0,2,0,8,2,8,
           1,1,8,0,3,0,7,3,7,1,2,1,7,2,7,0,4,0,6,4,6,1,3,1,6,3,6,2,2,6,0,5,5,1,4,1,5,4,5,2,3,2,5,3,5,2,4,4,3,3,4};

    std::vector<int> n1a (n1ap, n1ap + sizeof(n1ap) / sizeof(int) );
    std::vector<int> n2a (n2ap, n2ap + sizeof(n2ap) / sizeof(int) );
    std::vector<int> n3a (n3ap, n3ap + sizeof(n3ap) / sizeof(int) );

    double fac[]={1,1,2,6,24,120,720,5040,40320,362880,3628800};
    double sqrt_fac[11];
    for(int i=0;i<11;i++)
        sqrt_fac[i]=sqrt(fac[i]);

    double dfac[]={1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025,10321920,34459425,185794560,654729075,3715891200};


    std::vector<double> s0mtrx_ashore(n1a.size(),0);
    for(int i=0;i<n1a.size();i++)
    {
        if(n1a[i]%2==0 && n2a[i]%2==0 && n3a[i]%2==0)
        {
            s0mtrx_ashore[i] = (sqrt_fac[n1a[i]]/dfac[n1a[i]]) * (sqrt_fac[n2a[i]]/dfac[n2a[i]]) * (sqrt_fac[n3a[i]]/dfac[n3a[i]]);
        }
    }


    int mop[11]={1  ,    -1  ,     1   ,   -1    ,   1    ,  -1    ,   1   ,   -1   ,    1    ,  -1    ,   1};

    std::vector<double> vector_for_rtop;
    vector_for_rtop.resize(n1a.size());
    for(int i=0;i<n1a.size();i++)
    {
         vector_for_rtop[i]= mop[(n1a[i]+n2a[i]+n3a[i])/2]   *  s0mtrx_ashore[i] /sqrt(8. * pow(DPI,3));
    }

    std::vector<double> vector_for_rtap;
    vector_for_rtap.resize(n1a.size());
    for(int i=0;i<n1a.size();i++)
    {
         vector_for_rtap[i]= mop[(n2a[i]+n3a[i])/2]   *  s0mtrx_ashore[i] /sqrt(2. * DPI);
    }

    std::vector<double> vector_for_rtpp;
    vector_for_rtpp.resize(n1a.size());
    for(int i=0;i<n1a.size();i++)
    {
         vector_for_rtpp[i]= mop[n1a[i]/2]   *  s0mtrx_ashore[i] /sqrt(2. * DPI);
    }




    itk::ImageRegionIteratorWithIndex<ImageType3D> it(RTOP_image,RTOP_image->GetLargestPossibleRegion());
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType index= it.GetIndex();
        VecImageType::PixelType pix= coeff_image->GetPixel(index);
        if(pix[0]!=0)
        {
            double S0value=0;
            for(int i=0;i<Ncoeffs;i++)
            {
                S0value+= pix[i]*s0mtrx_ashore[i];
            }
            for(int i=0;i<Ncoeffs;i++)
                pix[i]/=S0value;

            MAPMRIModel::EValImageType::PixelType uu =eval_image->GetPixel(index);


            double sm=0;
            for(int i=0;i<Ncoeffs;i++)
                sm+= vector_for_rtop[i]*pix[i];
            if(uu[0]==0 ||uu[1]==0 ||uu[2]==0 )
                sm=0;
            else
                sm/= uu[0]*uu[1]*uu[2];

            double RTOP= sm;
            if(RTOP<0)
                RTOP=0;

            if(RTOP>0.4)
                RTOP=0;

            sm=0;
            for(int i=0;i<Ncoeffs;i++)
                sm+= vector_for_rtap[i]*pix[i];
            if(uu[1]==0 || uu[2]==0 )
                sm=0;
            else
                sm/= uu[1]*uu[2];

            double RTAP= sm;
            if(RTAP<0)
                RTAP=0;
            if(RTAP>0.54288)
                RTAP=0;

            sm=0;
            for(int i=0;i<Ncoeffs;i++)
                sm+= vector_for_rtpp[i]*pix[i];
            if(uu[0]==0)
                sm=0;
            else
                sm/= uu[0];

            double RTPP= sm;
            if(RTPP<0)
                RTPP=0;
            if(RTPP>0.73)
                RTPP=0.;


            RTOP_image->SetPixel(index,RTOP);
            RTAP_image->SetPixel(index,RTAP);
            RTPP_image->SetPixel(index,RTPP);

        }

        ++it;
    }


    std::string outname = filename.substr(0,filename.find(".nii")) + std::string("_RTOP.nii");
    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetFileName(outname);
    wr->SetInput(RTOP_image);
    wr->Update();

    outname = filename.substr(0,filename.find(".nii")) + std::string("_RTAP.nii");
        WrType::Pointer wr1= WrType::New();
        wr1->SetFileName(outname);
        wr1->SetInput(RTAP_image);
        wr1->Update();

        outname = filename.substr(0,filename.find(".nii")) + std::string("_RTPP.nii");
            WrType::Pointer wr2= WrType::New();
            wr2->SetFileName(outname);
            wr2->SetInput(RTPP_image);
            wr2->Update();


}








