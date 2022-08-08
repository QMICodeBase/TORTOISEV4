#include "defines.h"


#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"



int main(int argc, char *argv[])
{

    if(argc<2)
    {
        std::cout<<"EstimateMAPMRI_NG full_path_to_MAPMRI_coeff_image"<<std::endl;
        return 0;
    }

    std::string filename(argv[1]);

    typedef itk::VectorImage<float, 3>  VecImageType;
    typedef itk::ImageFileReader<VecImageType> VecReaderType;
    VecReaderType::Pointer reader = VecReaderType::New();
    reader->SetFileName(filename);
    reader->Update();
    VecImageType::Pointer coeff_image= reader->GetOutput();

    int Ncoeffs =coeff_image->GetNumberOfComponentsPerPixel();

    int order;
    for(order=2;order<=10;order+=2)
    {
        int nc= (order/2+1)*(order/2+2)*(2*order+3)/6;
        if(nc==Ncoeffs)
            break;
    }


    ImageType3D::Pointer NG_image = ImageType3D::New();
    NG_image->SetRegions(coeff_image->GetLargestPossibleRegion());
    NG_image->Allocate();
    NG_image->SetOrigin(coeff_image->GetOrigin());
    NG_image->SetSpacing(coeff_image->GetSpacing());
    NG_image->SetDirection(coeff_image->GetDirection());        
    NG_image->FillBuffer(0);



    ImageType3D::Pointer NGpar_image = ImageType3D::New();
    NGpar_image->SetRegions(coeff_image->GetLargestPossibleRegion());
    NGpar_image->Allocate();
    NGpar_image->SetOrigin(coeff_image->GetOrigin());
    NGpar_image->SetSpacing(coeff_image->GetSpacing());
    NGpar_image->SetDirection(coeff_image->GetDirection());    
    NGpar_image->FillBuffer(0);

    ImageType3D::Pointer NGperp_image = ImageType3D::New();
    NGperp_image->SetRegions(coeff_image->GetLargestPossibleRegion());
    NGperp_image->Allocate();
    NGperp_image->SetOrigin(coeff_image->GetOrigin());
    NGperp_image->SetSpacing(coeff_image->GetSpacing());
    NGperp_image->SetDirection(coeff_image->GetDirection());    
    NGperp_image->FillBuffer(0);




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

    int mop[] = {1  ,    -1    ,   1   ,   -1   ,    1    ,  -1    ,   1    ,  -1   ,    1    ,  -1     ,  1};

    double dfac[]={1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025,10321920,34459425,185794560,654729075,3715891200};


    std::vector<double> s0mtrx_ashore(n1a.size(),0);
    for(int i=0;i<n1a.size();i++)
    {
        if(n1a[i]%2==0 && n2a[i]%2==0 && n3a[i]%2==0)
        {
            s0mtrx_ashore[i] = (sqrt_fac[n1a[i]]/dfac[n1a[i]]) * (sqrt_fac[n2a[i]]/dfac[n2a[i]]) * (sqrt_fac[n3a[i]]/dfac[n3a[i]]);
        }
    }





    itk::ImageRegionIteratorWithIndex<ImageType3D> it(NG_image,NG_image->GetLargestPossibleRegion());
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

            double a2=0;
            for(int i=0;i<Ncoeffs;i++)
                a2+= pix[i]*pix[i];


            double NG=0;
            if(a2!=0)
               NG= sqrt(1- pix[0]*pix[0]/ a2);
            it.Set(NG);



            int nterms= order/2;
            vnl_vector<double> a1d(nterms+1,0);
            for(int kk=0;kk<=nterms;kk++)
            {
                for(int mm=0;mm<Ncoeffs;mm++)
                {
                    if(n1a[mm]==2*kk && n1a[mm]+n2a[mm]+n3a[mm]<=order)
                    {
                        if(n2a[mm]%2 ==0 && n3a[mm]%2 ==0)
                        {
                            a1d[kk]+=pix[mm]* mop[(n2a[mm]+n3a[mm])/2] *sqrt(fac[n2a[mm]]*fac[n3a[mm]]) /
                               (dfac[n2a[mm]]*dfac[n3a[mm]]);
                        }
                    }
                }
            }
            double tot_a1d2= 0;
            for(int kk=0;kk<=nterms;kk++)
                tot_a1d2 += a1d[kk]*a1d[kk];
            double NGpar=0;

            if(tot_a1d2!=0)
                NGpar=sqrt( 1-a1d[0]*a1d[0]/tot_a1d2 );




            NGpar_image->SetPixel(index,NGpar);

            vnl_matrix<double> a2d(order+1,order+1,0);
            for(int kk=0;kk<=order;kk++)
            {
                for(int ll=0;ll<=order;ll++)
                {
                    for(int mm=0;mm<Ncoeffs;mm++)
                    {
                        if(n2a[mm]==2*kk && n3a[mm]==ll && n1a[mm]<=order)
                        {
                            if(n1a[mm]%2 ==0)
                            {
                                a2d(kk,ll)+= pix[mm]* mop[n1a[mm]/2] * sqrt(fac[n1a[mm]]/dfac[n1a[mm]]);
                            }
                        }
                    }
                }
            }
            double tot_a2d2= 0;
            for(int kk=0;kk<=order;kk++)
                for(int ll=0;ll<=order;ll++)
                    tot_a2d2+= a2d(kk,ll)*a2d(kk,ll);

            double NGperp=0;
            if(tot_a2d2!=0)
                NGperp= sqrt(1- a2d(0,0)*a2d(0,0)/tot_a2d2);
            NGperp_image->SetPixel(index,NGperp);

        }

        ++it;
    }


    {
    std::string outname = filename.substr(0,filename.find(".nii")) + std::string("_NG.nii");
    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetFileName(outname);
    wr->SetInput(NG_image);
    wr->Update();
    }

    {
    std::string outname = filename.substr(0,filename.find(".nii")) + std::string("_NGpar.nii");
    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetFileName(outname);
    wr->SetInput(NGpar_image);
    wr->Update();
    }

    {
    std::string outname = filename.substr(0,filename.find(".nii")) + std::string("_NGperp.nii");
    typedef itk::ImageFileWriter<ImageType3D> WrType;
    WrType::Pointer wr= WrType::New();
    wr->SetFileName(outname);
    wr->SetInput(NGperp_image);
    wr->Update();
    }








}








