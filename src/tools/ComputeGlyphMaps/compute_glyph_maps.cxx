#include <vtkPolyDataMapper.h>
 #include <vtkActor.h>
 #include <vtkSmartPointer.h>
 #include <vtkRenderWindow.h>
 #include <vtkRenderer.h>
 #include <vtkRenderWindowInteractor.h>
 #include <vtkPolyData.h>
 #include <vtkSphereSource.h>
 #include <vtkWindowToImageFilter.h>
 #include <vtkPNGWriter.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include "vtkTensorGlyphOkan.h"
#include <vtkImageMapper.h>
#include <vtkActor2D.h>
#include <vtkImageActor.h>
#include <vtkFloatArray.h>
#include "defines.h"
#include "vtkCamera.h"
#include <vtkGraphicsFactory.h>
//#include <vtkImagingFactory.h>

#include "itkExtractImageFilter.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"

#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <vtkInteractorStyleTrackballCamera.h>


typedef double RealType;
typedef itk::DiffusionTensor3D<RealType> TensorPixelType;
typedef itk::Vector<RealType,6> VectorPixelType;

typedef itk::Image<itk::DiffusionTensor3D<RealType>,3>         TensorImageType;
typedef itk::Image<VectorPixelType,3>         VectorImageType;

typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;


int sl_numbers[3];
float FA_thr;

ImageType4D::Pointer dti_image;
ImageType4D::SizeType sz;

int slice_view=2;

std::vector< vtkSmartPointer<vtkImageActor> > image_actors[3];
std::vector< vtkSmartPointer<vtkActor> > actors[3];



void MyRender(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkRenderWindow> renderWindow )
{
    renderer->RemoveAllViewProps();

    vtkSmartPointer<vtkSphereSource> sphereSource =     vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetCenter(0, 0, 0.0);
    sphereSource->SetPhiResolution(20);
    sphereSource->SetThetaResolution(20);
    sphereSource->SetRadius(1.);
    sphereSource->Update();

     vtkSmartPointer<vtkActor> actor_x = vtkSmartPointer<vtkActor>::New();
     vtkSmartPointer<vtkActor> actor_y = vtkSmartPointer<vtkActor>::New();
     vtkSmartPointer<vtkActor> actor_z = vtkSmartPointer<vtkActor>::New();

     vtkSmartPointer<vtkImageActor> image_actor_x = vtkSmartPointer<vtkImageActor>::New();
     vtkSmartPointer<vtkImageActor> image_actor_y = vtkSmartPointer<vtkImageActor>::New();
     vtkSmartPointer<vtkImageActor> image_actor_z = vtkSmartPointer<vtkImageActor>::New();

    //axis 2
    {
        vtkSmartPointer<vtkImageData> imageData =  vtkSmartPointer<vtkImageData>::New();
        imageData->SetDimensions(sz[0],sz[1],1);
        imageData->AllocateScalars(VTK_FLOAT, 1);
        imageData->SetOrigin(0,0,sl_numbers[2]);
        imageData->SetSpacing(1,1,1);

        vtkSmartPointer<vtkPoints> points =          vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkDoubleArray> tensors = vtkSmartPointer<vtkDoubleArray>::New();
        tensors->SetNumberOfComponents(9);

        vtkSmartPointer<vtkUnsignedCharArray> colors =      vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetName("colors");
        colors->SetNumberOfComponents(3);

        int counter=0;
        ImageType4D::IndexType ind4;
        ind4[2]=sl_numbers[2];

        double max_FA=-1;
        for(int j=0;j<sz[1];j++)
        {
            ind4[1]=j;
            for(int i=0;i<sz[0];i++)
            {
                ind4[0]=i;

                InternalMatrixType mat;
                ind4[3]=0;
                mat(0,0)= dti_image->GetPixel(ind4);
                ind4[3]=1;
                mat(1,1)= dti_image->GetPixel(ind4);
                ind4[3]=2;
                mat(2,2)= dti_image->GetPixel(ind4);
                ind4[3]=3;
                mat(0,1)= dti_image->GetPixel(ind4);
                mat(1,0)= dti_image->GetPixel(ind4);
                ind4[3]=4;
                mat(0,2)= dti_image->GetPixel(ind4);
                mat(2,0)= dti_image->GetPixel(ind4);
                ind4[3]=5;
                mat(2,1)= dti_image->GetPixel(ind4);
                mat(1,2)= dti_image->GetPixel(ind4);

                vnl_symmetric_eigensystem<RealType>  eigf(mat);

                double MDf= (eigf.D(0,0)+eigf.D(1,1)+eigf.D(2,2))/3.;
                double nom= (eigf.D(0,0)-MDf)*(eigf.D(0,0)-MDf) + (eigf.D(1,1)-MDf)*(eigf.D(1,1)-MDf) + (eigf.D(2,2)-MDf)*(eigf.D(2,2)-MDf);
                double denom = eigf.D(0,0)*eigf.D(0,0) + eigf.D(1,1)*eigf.D(1,1) + eigf.D(2,2)*eigf.D(2,2) ;
                double FAf=std::min( sqrt (1.5* nom/denom),1.);


                float* pixelfR = static_cast<float*>(imageData->GetScalarPointer(i,sz[1]-j-1,0));
                pixelfR[0]= FAf  *255;

                if(FAf>max_FA)
                    max_FA=FAf;

                if(FAf>FA_thr)
                {
                    points->InsertNextPoint(i, sz[1]-j-1, sl_numbers[2]);

                    double maxval= std::max(eigf.D(0,0),std::max(eigf.D(1,1),eigf.D(2,2)));
                    eigf.D(0,0)=eigf.D(0,0)/maxval*FAf;
                    eigf.D(1,1)=eigf.D(1,1)/maxval*FAf;
                    eigf.D(2,2)=eigf.D(2,2)/maxval*FAf;

                    vnl_vector<double> evec= eigf.get_eigenvector(2);

                    InternalMatrixType vmat= eigf.recompose();
                    unsigned char s[3];

                    s[0]=(unsigned char)floor(255*fabs(evec[0]));
                    s[1]=(unsigned char)floor(255*fabs(evec[1]));
                    s[2]=(unsigned char)floor(255*fabs(evec[2]));

                    colors->InsertNextTupleValue( s);
                    tensors->InsertTuple9(counter,vmat(0,0),-vmat(0,1),vmat(0,2),-vmat(1,0),vmat(1,1),-vmat(1,2),vmat(2,0),-vmat(2,1),vmat(2,2));
                    counter++;

                }
            }
        }
        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->SetPoints(points);
        polyData->GetPointData()->SetTensors(tensors);
        polyData->GetPointData()->SetScalars(colors);


        vtkSmartPointer<vtkTensorGlyphOkan> tensorGlyph = vtkSmartPointer<vtkTensorGlyphOkan>::New();
        tensorGlyph->SetInputData(polyData);
        tensorGlyph->SetSourceConnection(sphereSource->GetOutputPort());
        tensorGlyph->ColorGlyphsOn();
        tensorGlyph->SetColorModeToScalars();
        tensorGlyph->ThreeGlyphsOff();
        tensorGlyph->ExtractEigenvaluesOn();
        tensorGlyph->ClampScalingOff();
        tensorGlyph->SetScaleFactor(1./max_FA);
        tensorGlyph->Update();

        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(tensorGlyph->GetOutput());
        mapper->SetColorMode(0);

        actor_z->SetMapper(mapper);
        image_actor_z->SetInputData(imageData);

    }  //axis



     //axis 0
     {
         vtkSmartPointer<vtkImageData> imageData =  vtkSmartPointer<vtkImageData>::New();
         imageData->SetDimensions(1,sz[1],sz[2]);
         imageData->AllocateScalars(VTK_FLOAT, 1);
         imageData->SetOrigin(sl_numbers[0],0,0);
         imageData->SetSpacing(1,1,1);

         vtkSmartPointer<vtkPoints> points =          vtkSmartPointer<vtkPoints>::New();
         vtkSmartPointer<vtkDoubleArray> tensors = vtkSmartPointer<vtkDoubleArray>::New();
         tensors->SetNumberOfComponents(9);

         vtkSmartPointer<vtkUnsignedCharArray> colors =      vtkSmartPointer<vtkUnsignedCharArray>::New();
         colors->SetName("colors");
         colors->SetNumberOfComponents(3);

         int counter=0;
         ImageType4D::IndexType ind4;
         ind4[0]=sl_numbers[0];

         double max_FA=-1;
         for(int k=0;k<sz[2];k++)
         {
             ind4[2]=k;
             for(int j=0;j<sz[1];j++)
             {
                 ind4[1]=j;

                 InternalMatrixType mat;
                 ind4[3]=0;
                 mat(0,0)= dti_image->GetPixel(ind4);
                 ind4[3]=1;
                 mat(1,1)= dti_image->GetPixel(ind4);
                 ind4[3]=2;
                 mat(2,2)= dti_image->GetPixel(ind4);
                 ind4[3]=3;
                 mat(0,1)= dti_image->GetPixel(ind4);
                 mat(1,0)= dti_image->GetPixel(ind4);
                 ind4[3]=4;
                 mat(0,2)= dti_image->GetPixel(ind4);
                 mat(2,0)= dti_image->GetPixel(ind4);
                 ind4[3]=5;
                 mat(2,1)= dti_image->GetPixel(ind4);
                 mat(1,2)= dti_image->GetPixel(ind4);

                 vnl_symmetric_eigensystem<RealType>  eigf(mat);

                 double MDf= (eigf.D(0,0)+eigf.D(1,1)+eigf.D(2,2))/3.;
                 double nom= (eigf.D(0,0)-MDf)*(eigf.D(0,0)-MDf) + (eigf.D(1,1)-MDf)*(eigf.D(1,1)-MDf) + (eigf.D(2,2)-MDf)*(eigf.D(2,2)-MDf);
                 double denom = eigf.D(0,0)*eigf.D(0,0) + eigf.D(1,1)*eigf.D(1,1) + eigf.D(2,2)*eigf.D(2,2) ;
                 double FAf=std::min( sqrt (1.5* nom/denom),1.);


                 float* pixelfR = static_cast<float*>(imageData->GetScalarPointer(0,sz[1]-j-1,k));
                 pixelfR[0]= FAf  *255;

                 if(FAf>max_FA)
                     max_FA=FAf;

                 if(FAf>FA_thr)
                 {
                     points->InsertNextPoint(sl_numbers[0], sz[1]-j-1, k);

                     double maxval= std::max(eigf.D(0,0),std::max(eigf.D(1,1),eigf.D(2,2)));
                     eigf.D(0,0)=eigf.D(0,0)/maxval*FAf;
                     eigf.D(1,1)=eigf.D(1,1)/maxval*FAf;
                     eigf.D(2,2)=eigf.D(2,2)/maxval*FAf;

                     vnl_vector<double> evec= eigf.get_eigenvector(2);

                     InternalMatrixType vmat= eigf.recompose();
                     unsigned char s[3];

                     s[0]=(unsigned char)floor(255*fabs(evec[0]));
                     s[1]=(unsigned char)floor(255*fabs(evec[1]));
                     s[2]=(unsigned char)floor(255*fabs(evec[2]));

                     colors->InsertNextTupleValue( s);
                     tensors->InsertTuple9(counter,vmat(0,0),-vmat(0,1),vmat(0,2),-vmat(1,0),vmat(1,1),-vmat(1,2),vmat(2,0),-vmat(2,1),vmat(2,2));
                     counter++;

                 }
             }
         }
         vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
         polyData->SetPoints(points);
         polyData->GetPointData()->SetTensors(tensors);
         polyData->GetPointData()->SetScalars(colors);


         vtkSmartPointer<vtkTensorGlyphOkan> tensorGlyph = vtkSmartPointer<vtkTensorGlyphOkan>::New();
         tensorGlyph->SetInputData(polyData);
         tensorGlyph->SetSourceConnection(sphereSource->GetOutputPort());
         tensorGlyph->ColorGlyphsOn();
         tensorGlyph->SetColorModeToScalars();
         tensorGlyph->ThreeGlyphsOff();
         tensorGlyph->ExtractEigenvaluesOn();
         tensorGlyph->ClampScalingOff();
         tensorGlyph->SetScaleFactor(1./max_FA);
         tensorGlyph->Update();

         vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
         mapper->SetInputData(tensorGlyph->GetOutput());
         mapper->SetColorMode(0);

         actor_x->SetMapper(mapper);
         image_actor_x->SetInputData(imageData);
         image_actor_x->SetDisplayExtent(0,0,0,sz[1]-1,0,sz[2]-1);


     }  //axis 0


     //axis 1
     {
         vtkSmartPointer<vtkImageData> imageData2 =  vtkSmartPointer<vtkImageData>::New();
         imageData2->SetDimensions(sz[0],1,sz[2]);
         imageData2->AllocateScalars(VTK_FLOAT, 1);
         imageData2->SetOrigin(0,sz[1]-1-sl_numbers[1],0);
         imageData2->SetSpacing(1,1,1);

         vtkSmartPointer<vtkPoints> points =          vtkSmartPointer<vtkPoints>::New();
         vtkSmartPointer<vtkDoubleArray> tensors = vtkSmartPointer<vtkDoubleArray>::New();
         tensors->SetNumberOfComponents(9);

         vtkSmartPointer<vtkUnsignedCharArray> colors =      vtkSmartPointer<vtkUnsignedCharArray>::New();
         colors->SetName("colors");
         colors->SetNumberOfComponents(3);

         int counter=0;
         ImageType4D::IndexType ind4;
         ind4[1]=sl_numbers[1];

         double max_FA=-1;
         for(int k=0;k<sz[2];k++)
         {
             ind4[2]=k;
             for(int i=0;i<sz[0];i++)
             {
                 ind4[0]=i;

                 InternalMatrixType mat;
                 ind4[3]=0;
                 mat(0,0)= dti_image->GetPixel(ind4);
                 ind4[3]=1;
                 mat(1,1)= dti_image->GetPixel(ind4);
                 ind4[3]=2;
                 mat(2,2)= dti_image->GetPixel(ind4);
                 ind4[3]=3;
                 mat(0,1)= dti_image->GetPixel(ind4);
                 mat(1,0)= dti_image->GetPixel(ind4);
                 ind4[3]=4;
                 mat(0,2)= dti_image->GetPixel(ind4);
                 mat(2,0)= dti_image->GetPixel(ind4);
                 ind4[3]=5;
                 mat(2,1)= dti_image->GetPixel(ind4);
                 mat(1,2)= dti_image->GetPixel(ind4);

                 vnl_symmetric_eigensystem<RealType>  eigf(mat);

                 double MDf= (eigf.D(0,0)+eigf.D(1,1)+eigf.D(2,2))/3.;
                 double nom= (eigf.D(0,0)-MDf)*(eigf.D(0,0)-MDf) + (eigf.D(1,1)-MDf)*(eigf.D(1,1)-MDf) + (eigf.D(2,2)-MDf)*(eigf.D(2,2)-MDf);
                 double denom = eigf.D(0,0)*eigf.D(0,0) + eigf.D(1,1)*eigf.D(1,1) + eigf.D(2,2)*eigf.D(2,2) ;
                 double FAf=std::min( sqrt (1.5* nom/denom),1.);


                 float* pixelfR = static_cast<float*>(imageData2->GetScalarPointer(i,0,k));
                 pixelfR[0]= FAf  *255;

                 if(FAf>max_FA)
                     max_FA=FAf;

                 if(FAf>FA_thr)
                 {
                     points->InsertNextPoint(i, sz[1]-1-sl_numbers[1], k);

                     double maxval= std::max(eigf.D(0,0),std::max(eigf.D(1,1),eigf.D(2,2)));
                     eigf.D(0,0)=eigf.D(0,0)/maxval*FAf;
                     eigf.D(1,1)=eigf.D(1,1)/maxval*FAf;
                     eigf.D(2,2)=eigf.D(2,2)/maxval*FAf;

                     vnl_vector<double> evec= eigf.get_eigenvector(2);

                     InternalMatrixType vmat= eigf.recompose();
                     unsigned char s[3];

                     s[0]=(unsigned char)floor(255*fabs(evec[0]));
                     s[1]=(unsigned char)floor(255*fabs(evec[1]));
                     s[2]=(unsigned char)floor(255*fabs(evec[2]));

                     colors->InsertNextTupleValue( s);
                     tensors->InsertTuple9(counter,vmat(0,0),-vmat(0,1),vmat(0,2),-vmat(1,0),vmat(1,1),-vmat(1,2),vmat(2,0),-vmat(2,1),vmat(2,2));
                     counter++;

                 }
             }
         }
         vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
         polyData->SetPoints(points);
         polyData->GetPointData()->SetTensors(tensors);
         polyData->GetPointData()->SetScalars(colors);


         vtkSmartPointer<vtkTensorGlyphOkan> tensorGlyph = vtkSmartPointer<vtkTensorGlyphOkan>::New();
         tensorGlyph->SetInputData(polyData);
         tensorGlyph->SetSourceConnection(sphereSource->GetOutputPort());
         tensorGlyph->ColorGlyphsOn();
         tensorGlyph->SetColorModeToScalars();
         tensorGlyph->ThreeGlyphsOff();
         tensorGlyph->ExtractEigenvaluesOn();
         tensorGlyph->ClampScalingOff();
         tensorGlyph->SetScaleFactor(1./max_FA);
         tensorGlyph->Update();

         vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
         mapper->SetInputData(tensorGlyph->GetOutput());
         mapper->SetColorMode(0);

         actor_y->SetMapper(mapper);
         image_actor_y->SetInputData(imageData2);
         image_actor_y->SetDisplayExtent(0,sz[0]-1,0,0,0,sz[2]-1);



     }  //axis 1



     renderer->AddActor2D(image_actor_z);
     renderer->AddActor2D(image_actor_y);
     renderer->AddActor2D(image_actor_x);
     renderer->AddActor(actor_x);
     renderer->AddActor(actor_y);
     renderer->AddActor(actor_z);



}



// Define interaction style
class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
public:
  static KeyPressInteractorStyle* New();
  vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);

  virtual void OnKeyPress() override
  {
    // Get the keypress
    vtkRenderWindowInteractor* rwi = this->Interactor;
    std::string key = rwi->GetKeySym();

    auto renderer=this->GetCurrentRenderer();
    auto renderWindow =renderer->GetRenderWindow();


    // Handle an arrow key
    if (key == "Up")
    {
        sl_numbers[1]++;
        if(sl_numbers[1]>sz[1]-1)
            sl_numbers[1]=sz[1]-1;

        MyRender(renderer,renderWindow);

    }
    if (key == "Down")
    {
        sl_numbers[1]--;
        if(sl_numbers[1]<0)
            sl_numbers[1]=0;
        MyRender(renderer,renderWindow);

    }
    if (key == "Left")
    {
        sl_numbers[0]--;
        if(sl_numbers[0]<0)
            sl_numbers[0]=0;
        MyRender(renderer,renderWindow);

    }
    if (key == "Right")
    {
        sl_numbers[0]++;
        if(sl_numbers[0]>sz[0]-1)
            sl_numbers[0]=sz[0]-1;
        MyRender(renderer,renderWindow);

    }
    if (key == "Prior")
    {
        sl_numbers[2]++;
        if(sl_numbers[2]>sz[2]-1)
            sl_numbers[2]=sz[2]-1;
        MyRender(renderer,renderWindow);

    }
    if (key == "Next")
    {
        sl_numbers[2]--;
        if(sl_numbers[2]<0)
            sl_numbers[2]=0;
        MyRender(renderer,renderWindow);

    }
    if (key == "space")
    {
        slice_view = (slice_view+1)%3;
        MyRender(renderer,renderWindow);

    }

    renderWindow->Render();


    // Forward events
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }
};
vtkStandardNewMacro(KeyPressInteractorStyle);




int main(int argc, char *argv[])
{
    if(argc<2)
    {
        std::cout<<"Usage: ComputeGlyphMaps  full_path_to_tensor_NIFTI_file  max_FA_to_threshold (optional.Default:0.3)"<<std::endl;
        return 0;
    }

    std::cout<<"Use the up/down/left/arrow arrow keys and PageUp/PageDown to change slices..."<<std::endl;
    std::cout<<"Rotate with Left-mouse, zooom in-out with right mouse..."<<std::endl;
    std::cout<<"Drag with Shift+Left mouse..."<<std::endl;
    std::cout<<"A screenshot will be taken when you close the VTK window..."<<std::endl;




    vtkObject::GlobalWarningDisplayOff();

    FA_thr=0.3;
    if(argc>2)
        FA_thr=atof(argv[2]);




    dti_image= readImageD<ImageType4D>(argv[1]);
    sz=dti_image->GetLargestPossibleRegion().GetSize();


    sl_numbers[0]= sz[0]/2;
    sl_numbers[1]= sz[1]/2;
    sl_numbers[2]= sz[2]/2;

    std::string oname;
    if(argc>5)
    {
        oname= std::string(argv[5]);
    }
    else
    {
        std::ostringstream convert1;
        convert1 << sl_numbers[0];
        std::ostringstream convert2;
        convert2 << sl_numbers[1];
        std::ostringstream convert3;
        convert3 << sl_numbers[2];


        std::string sl1 = convert1.str();
        std::string sl2 = convert2.str();
        std::string sl3 = convert3.str();


        oname= std::string(argv[1]);
        oname = oname.substr(0,oname.find(".nii")) + std::string("_") + sl1 + std::string("_") + sl2 +  std::string("_") + sl3 + std::string(".png");
    }
    std::cout<<oname<<std::endl;



    for(int d=0;d<3;d++)
    {
        image_actors[d].resize(sz[d]);
        actors[d].resize(sz[d]);
    }












    vtkSmartPointer<vtkRenderer> renderer =     vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow  =     vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(900, 900);
    renderWindow->SetOffScreenRendering(0);
    renderWindow->AddRenderer(renderer);

    MyRender(renderer,renderWindow);


      // An interactor
      vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =  vtkSmartPointer<vtkRenderWindowInteractor>::New();
      renderWindowInteractor->SetRenderWindow ( renderWindow );

      // Set the custom stype to use for interaction.
      vtkSmartPointer<KeyPressInteractorStyle> style =        vtkSmartPointer<KeyPressInteractorStyle>::New();
      style->SetDefaultRenderer(renderer);
      renderWindowInteractor->SetInteractorStyle( style );


    renderWindow->Render();
    renderWindowInteractor->Initialize();
    renderWindowInteractor->Start();

  //  renderWindow->Delete();
 //   renderer->Delete();



    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =     vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetMagnification(2);
    windowToImageFilter->SetInputBufferTypeToRGB();
    windowToImageFilter->Update();

    vtkSmartPointer<vtkPNGWriter> writer =     vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(oname.c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();




  //  renderWindow->Render();
   // renderer->ResetCamera();
   // renderWindow->Render();



   // renderWindowInteractor->Initialize();
   // renderWindowInteractor->Start();




   return EXIT_SUCCESS;
}
