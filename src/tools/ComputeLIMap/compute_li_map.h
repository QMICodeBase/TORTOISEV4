#ifndef COMPUTELIMAP_H
#define COMPUTELIMAP_H

#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageDuplicator.h"


typedef float RealType;
typedef itk::Image<RealType,4> ImageType4D;
typedef itk::Image<RealType,3> ImageType3D;
typedef  vnl_matrix_fixed< double, 3, 3 > InternalMatrixType;

typedef itk::Image<InternalMatrixType,3> DTImageType;


#define SIMILAR_PAR_A 1

vnl_vector<double>  compute_similarity(ImageType3D::Pointer A0_image,ImageType3D::IndexType ind)
{
    vnl_vector<double> temp(8,0);
    ImageType3D::IndexType nind;
    nind[2]=ind[2];

    double cval = A0_image->GetPixel(ind);
    if(cval>0)
    {
        {
            int vind=0;
            nind[0]=ind[0]+1;
            nind[1]=ind[1];
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
        {
            int vind=1;
            nind[0]=ind[0]-1;
            nind[1]=ind[1];
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
        {
            int vind=2;
            nind[0]=ind[0];
            nind[1]=ind[1]+1;
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
        {
            int vind=3;
            nind[0]=ind[0];
            nind[1]=ind[1]-1;
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
        {
            int vind=4;
            nind[0]=ind[0]+1;
            nind[1]=ind[1]+1;
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
        {
            int vind=5;
            nind[0]=ind[0]+1;
            nind[1]=ind[1]-1;
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
        {
            int vind=6;
            nind[0]=ind[0]-1;
            nind[1]=ind[1]+1;
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
        {
            int vind=7;
            nind[0]=ind[0]-1;
            nind[1]=ind[1]-1;
            double nval= A0_image->GetPixel(nind);
            if(nval >0)
            {
                double val = fabs(cval-nval)/ (cval+nval);
                val=  val>0 ? val : 0;
                val=  val<1 ? val : 1;
                temp[vind]=1 -val;
            }
        }
    }

    temp[0]*=0.7071;
    temp[1]*=0.7071;
    temp[2]*=0.7071;
    temp[3]*=0.7071;

    return temp;
}

double el_mult(InternalMatrixType &m1,InternalMatrixType &m2)
{
    InternalMatrixType res;
    res(0,0)=m1(0,0)*m2(0,0);
    res(0,1)=m1(0,1)*m2(0,1);
    res(0,2)=m1(0,2)*m2(0,2);
    res(1,1)=m1(1,1)*m2(1,1);
    res(1,2)=m1(1,2)*m2(1,2);
    res(2,2)=m1(2,2)*m2(2,2);

    return res(0,0)+res(1,1)+res(2,2)+2*res(0,1)+2*res(0,2)+2*res(1,2);
}


ImageType3D::Pointer compute_li_map(ImageType4D::Pointer image4D,ImageType3D::Pointer A0_image)
{              
    ImageType3D::SizeType imsize= A0_image->GetLargestPossibleRegion().GetSize();


    ImageType3D::Pointer li_map = ImageType3D::New();
    li_map->SetRegions(A0_image->GetLargestPossibleRegion());
    li_map->Allocate();
    li_map->SetOrigin(A0_image->GetOrigin());
    li_map->SetSpacing(A0_image->GetSpacing());
    li_map->SetDirection(A0_image->GetDirection());
    li_map->FillBuffer(0.);


    DTImageType::Pointer tensor_image= DTImageType::New();
    tensor_image->SetRegions(A0_image->GetLargestPossibleRegion());
    tensor_image->Allocate();
    tensor_image->SetOrigin(A0_image->GetOrigin());
    tensor_image->SetSpacing(A0_image->GetSpacing());
    tensor_image->SetDirection(A0_image->GetDirection());


    DTImageType::Pointer dev_tensor_image= DTImageType::New();
    dev_tensor_image->SetRegions(A0_image->GetLargestPossibleRegion());
    dev_tensor_image->Allocate();
    dev_tensor_image->SetOrigin(A0_image->GetOrigin());
    dev_tensor_image->SetSpacing(A0_image->GetSpacing());
    dev_tensor_image->SetDirection(A0_image->GetDirection());


    InternalMatrixType eye;
    eye.set_identity();

    ImageType3D::IndexType ind;
    ImageType4D::IndexType index;

    for(int k=0;k<imsize[2];k++)
    {
        index[2]=k;
        ind[2]=k;
        for(int j=0;j<imsize[1];j++)
        {
            index[1]=j;
            ind[1]=j;
            for(int i=0;i<imsize[0];i++)
            {
                index[0]=i;
                ind[0]=i;

                InternalMatrixType curr_tens,dev_tens;

                index[3]=0;
                curr_tens(0,0)=image4D->GetPixel(index);
                index[3]=1;
                curr_tens(1,1)=image4D->GetPixel(index);
                index[3]=2;
                curr_tens(2,2)=image4D->GetPixel(index);
                index[3]=3;
                curr_tens(0,1)=image4D->GetPixel(index);
                curr_tens(1,0)=image4D->GetPixel(index);
                index[3]=4;
                curr_tens(0,2)=image4D->GetPixel(index);
                curr_tens(2,0)=image4D->GetPixel(index);
                index[3]=5;
                curr_tens(1,2)=image4D->GetPixel(index);
                curr_tens(2,1)=image4D->GetPixel(index);

                double MD= (curr_tens(0,0)+curr_tens(1,1)+curr_tens(2,2))/3.;
                dev_tens= curr_tens -MD*eye;

                tensor_image->SetPixel(ind,curr_tens);
                dev_tensor_image->SetPixel(ind,dev_tens);
            }
        }
    }



    for(int k=0;k<imsize[2];k++)
    {
        ind[2]=k;
        for(int j=1;j<imsize[1]-1;j++)
        {
            ind[1]=j;
            for(int i=1;i<imsize[0]-1;i++)
            {
                ind[0]=i;

                if( k==56 && i==55 && j==97)
                    int ma=0;

                if(A0_image->GetPixel(ind)>0)
                {
                    vnl_vector<double> sim= compute_similarity(A0_image,ind);


                    double total_sim = sim.sum();
                    if(total_sim > SIMILAR_PAR_A)
                    {
                        vnl_vector<double> temp(8,0), temp2(8,0);
                        ImageType3D::IndexType nind;
                        nind[2]=k;

                        InternalMatrixType curr_dev= dev_tensor_image->GetPixel(ind);
                        InternalMatrixType curr_tens= tensor_image->GetPixel(ind);

                        double organ1= curr_tens.frobenius_norm() * sqrt(2./3.);


                        int vind=0;
                        if(sim[vind]>0)
                        {
                            nind[0]=i+1;
                            nind[1]=j;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }
                        vind=1;
                        if(sim[vind]>0)
                        {
                            nind[0]=i-1;
                            nind[1]=j;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }
                        vind=2;
                        if(sim[vind]>0)
                        {
                            nind[0]=i;
                            nind[1]=j+1;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }
                        vind=3;
                        if(sim[vind]>0)
                        {
                            nind[0]=i;
                            nind[1]=j-1;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }
                        vind=4;
                        if(sim[vind]>0)
                        {
                            nind[0]=i+1;
                            nind[1]=j+1;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }
                        vind=5;
                        if(sim[vind]>0)
                        {
                            nind[0]=i+1;
                            nind[1]=j-1;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }
                        vind=6;
                        if(sim[vind]>0)
                        {
                            nind[0]=i-1;
                            nind[1]=j+1;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }
                        vind=7;
                        if(sim[vind]>0)
                        {
                            nind[0]=i-1;
                            nind[1]=j-1;
                            InternalMatrixType n_dev= dev_tensor_image->GetPixel(nind);
                            InternalMatrixType n_tens= tensor_image->GetPixel(nind);
                            double dmult = el_mult(curr_dev,n_dev);
                            double tmult = el_mult(curr_tens,n_tens);
                            double val=dmult/tmult;
                            val= val>0 ? val :0;
                            temp[vind]= sqrt(val);
                            double organ2=n_tens.frobenius_norm()* sqrt(2./3.);
                            temp2[vind]=dmult/organ1/organ2;
                        }

                        double t1=0,t3=0,t2=0;
                        for(vind=0;vind<8;vind++)
                        {
                            if(sim[vind]>0)
                            {
                                t1+=temp[vind]*sim[vind];
                                t2+=temp2[vind]*sim[vind];
                                t3+=sim[vind];
                            }
                        }

                        double org1= t1/t3 *sqrt(1.5);
                        double org2= t2/t3;

                        double li =0.5*(org1+org2);
                        li_map->SetPixel(ind,li);
                    }
                }
            }
        }
    }

    return li_map;

}





#endif
