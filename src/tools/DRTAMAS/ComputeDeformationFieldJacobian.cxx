#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "defines.h"
#include "DRTAMAS_utilities_cp.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkInvertDisplacementFieldImageFilter.h"


DisplacementFieldType::Pointer InvertDisplacementField( const DisplacementFieldType * field)
{
    typedef itk::InvertDisplacementFieldImageFilter<DisplacementFieldType> InverterType;

    InverterType::Pointer inverter = InverterType::New();
    inverter->SetInput( field );
    inverter->SetMaximumNumberOfIterations( 200 );
    inverter->SetMeanErrorToleranceThreshold( 0.0003 );
    inverter->SetMaxErrorToleranceThreshold( 0.03 );
    inverter->Update();

    DisplacementFieldType::Pointer inverseField = inverter->GetOutput();

    return inverseField;
}

int main( int argc , char * argv[] )
{
    if(argc<2)
    {
        std::cout<<"Usage:   ComputeDeformationFieldJacobian field UseLog(0/1. Default:1) "<<std::endl;
        return EXIT_FAILURE;
    }

    DisplacementFieldType::Pointer field= readImageD<DisplacementFieldType>(argv[1]);

    bool useLOG=1;
    if(argc>2)
        useLOG=atoi(argv[2]);

    ImageType3D::Pointer output_img=ImageType3D::New();
    output_img->SetRegions(field->GetLargestPossibleRegion());
    output_img->Allocate();
    output_img->SetDirection(field->GetDirection());
    output_img->SetOrigin(field->GetOrigin());
    output_img->SetSpacing(field->GetSpacing());


    using JacobianPixelType = itk::Vector<float,9>;
    using JacobianImageType= itk::Image<JacobianPixelType,3>;

    bool write_jac=false;
    JacobianImageType::Pointer jac_img = nullptr;
    JacobianPixelType zervec; zervec.Fill(0);
    if(argc>3 && atoi(argv[3])==1)
    {
        write_jac=true;
        jac_img= JacobianImageType::New();
        jac_img->SetRegions(field->GetLargestPossibleRegion());
        jac_img->Allocate();
        jac_img->SetDirection(field->GetDirection());
        jac_img->SetOrigin(field->GetOrigin());
        jac_img->SetSpacing(field->GetSpacing());
        jac_img->FillBuffer(zervec);
    }


    DisplacementFieldType::Pointer smooth_field= field;

    int Mit=0;

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(output_img,output_img->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType3D::IndexType ind3= it.GetIndex();
        vnl_matrix_fixed<double,3,3> JAC = ComputeJacobian(field,ind3);

        double det= vnl_determinant<double,3,3>(JAC);

        if(det<=0)
        {
            JAC=ComputeJacobian(smooth_field,ind3);
            det= vnl_determinant<double,3,3>(JAC);
        }

        while(det<=0 && Mit<10)
        {
            auto temp=InvertDisplacementField(smooth_field);
            smooth_field=InvertDisplacementField(temp);
            JAC = ComputeJacobian(smooth_field,ind3);
            det= vnl_determinant<double,3,3>(JAC);
            ++Mit;
        }

        if(write_jac)
        {
            JacobianPixelType vec;
            vec[0]=JAC(0,0);vec[1]=JAC(0,1);vec[2]=JAC(0,2);
            vec[3]=JAC(1,0);vec[4]=JAC(1,1);vec[5]=JAC(1,2);
            vec[6]=JAC(2,0);vec[7]=JAC(2,1);vec[8]=JAC(2,2);

            jac_img->SetPixel(ind3,vec);
        }


        if(det<=0)
            det=1E-5;
        if(useLOG)
            det=std::log(det);


        it.Set(det);
    }

    std::string iname=argv[1];
    std::string oname = iname.substr(0,iname.rfind(".nii")) + "_JAC.nii";
    writeImageD<ImageType3D>(output_img,oname);

    if(write_jac)
    {
        oname = iname.substr(0,iname.rfind(".nii")) + "_JACMAT.nii";

        using JacWriterType = itk::ImageFileWriter<JacobianImageType>;
        JacWriterType::Pointer wr=JacWriterType::New();
        wr->SetFileName(oname);
        wr->SetInput(jac_img);
        wr->Update();
    }

    return EXIT_SUCCESS;






}
