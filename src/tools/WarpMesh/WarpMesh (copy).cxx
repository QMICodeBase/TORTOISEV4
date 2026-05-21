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
#include <vtkImageMapper.h>
#include <vtkActor2D.h>
#include <vtkImageActor.h>
#include <vtkFloatArray.h>
#include "defines.h"
#include "vtkCamera.h"
#include <vtkGraphicsFactory.h>
#include "vtkPolyLine.h"
#include "vtkImageProperty.h"

#include "vnl/algo/vnl_symmetric_eigensystem.h"

#include "defines.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <vtkInteractorStyleTrackballCamera.h>
#include "vtkLine.h"
#include "vtkNamedColors.h"
#include "vtkProperty.h"
#include "vtkLookupTable.h"
#include "vtkCellData.h"

#include "../DRTAMAS/DRTAMAS_utilities_cp.h"

typedef double RealType;
typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;


int curr_slice_z, curr_slice_y,curr_slice_x;
ImageType3D::SizeType sz;
DisplacementFieldType::Pointer field;
ImageType3D::Pointer overlay_img;

float opac;

bool ax,sag,cor;
bool display_warped;



void MyRender(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkRenderWindow> renderWindow )
{
    renderer->RemoveAllViewProps();

    using DisplacementFieldTransformType = itk::DisplacementFieldTransform<double,3>;
    DisplacementFieldTransformType::Pointer trans= DisplacementFieldTransformType::New();
    trans->SetDisplacementField(field);

    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetNumberOfColors(256);
    lut->SetNumberOfTableValues(256);

    double mn=-0.5;
    double mx=0.5;
    lut->SetTableRange(mn, mx);
    lut->Build();

    lut->SetTableValue(0,0,0,1);
    for(int c=1;c<127;c++)
        lut->SetTableValue(c,0, c/126.,1-c/126.);

    lut->SetTableValue(127,0,1,0);
    for(int c=128;c<255;c++)
        lut->SetTableValue(c, (c-127)/127., 1- (c-127)/127.,0);

    lut->SetTableValue(255,1,0,0);

    if(ax)
    {
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> lines=vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkFloatArray> scalars = vtkSmartPointer<vtkFloatArray>::New();
        scalars->SetName("Scalars");
        scalars->SetNumberOfComponents(1);

        long cnt=0;
        for(int j=0;j<sz[1];j++)
        {
            ImageType3D::IndexType ind3;
            ind3[2]=curr_slice_z;
            ind3[1]=j;

            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                ImageType3D::PointType pt,pt_trans;
                field->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(display_warped)
                    pt_trans=trans->TransformPoint(pt);

                vnl_matrix_fixed<double,3,3> jac = ComputeJacobian(field,ind3);
                float mdet= vnl_det<double>(jac);
                mdet=log(mdet);
                scalars->InsertNextValue(mdet);


                points->InsertNextPoint(pt_trans[0],pt_trans[1],pt_trans[2]);
                if(i!=sz[0]-1)
                {
                    vtkSmartPointer<vtkLine> line=vtkSmartPointer<vtkLine>::New();
                    line->GetPointIds()->SetId(0, cnt);
                    line->GetPointIds()->SetId(1, cnt+1);
                    lines->InsertNextCell(line);

                }
                cnt++;
            }
        }

        cnt=0;
        for(int j=0;j<sz[1];j++)
        {
            for(int i=0;i<sz[0];i++)
            {
                if(j!=sz[1]-1)
                {
                    vtkSmartPointer<vtkLine> line=vtkSmartPointer<vtkLine>::New();
                    line->GetPointIds()->SetId(0, cnt);
                    line->GetPointIds()->SetId(1, cnt+sz[0]);
                    lines->InsertNextCell(line);
                }
                cnt++;
            }
        }



        // Create a polydata to store everything in
        vtkSmartPointer<vtkPolyData> linesPolyData=vtkSmartPointer<vtkPolyData>::New();
        // Add the points to the dataset
        linesPolyData->SetPoints(points);
        linesPolyData->GetPointData()->SetScalars(scalars);

        // Add the lines to the dataset
        linesPolyData->SetLines(lines);
        linesPolyData->GetCellData()->SetScalars(scalars);



        vtkSmartPointer<vtkNamedColors> colors=vtkSmartPointer<vtkNamedColors>::New();

        vtkSmartPointer<vtkPolyDataMapper> mapper=vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(linesPolyData);
        mapper->ScalarVisibilityOn();
        mapper->SetLookupTable(lut);
        mapper->SetScalarModeToUsePointData();
        mapper->SetColorModeToMapScalars();
        mapper->UseLookupTableScalarRangeOn();


        vtkSmartPointer<vtkActor> actor=vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetLineWidth(1);

        renderer->AddActor(actor);


        if(overlay_img)
        {
            ImageType3D::IndexType temp_index;
            temp_index[0]=0;
            temp_index[1]=0;
            temp_index[2]=curr_slice_z;
            ImageType3D::PointType orig;
            overlay_img->TransformIndexToPhysicalPoint(temp_index,orig);

            ImageType3D::DirectionType dir = overlay_img->GetDirection();
            ImageType3D::SpacingType spc= overlay_img->GetSpacing();

            vtkSmartPointer<vtkImageData> imageData =  vtkSmartPointer<vtkImageData>::New();
            imageData->SetDimensions(sz[0],sz[1],1);
            imageData->SetOrigin(orig[0],orig[1],orig[2]);
            imageData->SetSpacing(spc[0],spc[1],spc[2]);
            imageData->SetDirectionMatrix(dir(0,0),dir(0,1),dir(0,2),dir(1,0),dir(1,1),dir(1,2),dir(2,0),dir(2,1),dir(2,2));
            imageData->AllocateScalars(VTK_FLOAT, 1);

            for(int j=0;j<sz[1];j++)
            {
                ImageType3D::IndexType ind3;
                ind3[2]=curr_slice_z;
                ind3[1]=j;

                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    float* pixel = static_cast<float*>(imageData->GetScalarPointer(i,j,0));
                    pixel[0]= overlay_img->GetPixel(ind3)  *255;
                }
            }
            vtkSmartPointer<vtkImageActor> image_actor = vtkSmartPointer<vtkImageActor>::New();
            image_actor->SetInputData(imageData);
            image_actor->GetProperty()->SetOpacity(opac);
            renderer->AddActor2D(image_actor);
        }
    }

    if(cor)
    {
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> lines=vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkFloatArray> scalars = vtkSmartPointer<vtkFloatArray>::New();
        scalars->SetName("Scalars");
        scalars->SetNumberOfComponents(1);

        long cnt=0;
        for(int k=0;k<sz[2];k++)
        {
            ImageType3D::IndexType ind3;
            ind3[1]=curr_slice_y;
            ind3[2]=k;

            for(int i=0;i<sz[0];i++)
            {
                ind3[0]=i;
                ImageType3D::PointType pt,pt_trans;
                field->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(display_warped)
                    pt_trans=trans->TransformPoint(pt);

                vnl_matrix_fixed<double,3,3> jac = ComputeJacobian(field,ind3);
                float mdet= vnl_det<double>(jac);
                mdet=log(mdet);
                scalars->InsertNextValue(mdet);


                points->InsertNextPoint(pt_trans[0],pt_trans[1],pt_trans[2]);
                if(i!=sz[0]-1)
                {
                    vtkSmartPointer<vtkLine> line=vtkSmartPointer<vtkLine>::New();
                    line->GetPointIds()->SetId(0, cnt);
                    line->GetPointIds()->SetId(1, cnt+1);
                    lines->InsertNextCell(line);

                }
                cnt++;
            }
        }

        cnt=0;
        for(int k=0;k<sz[2];k++)
        {
            for(int i=0;i<sz[0];i++)
            {
                if(k!=sz[2]-1)
                {
                    vtkSmartPointer<vtkLine> line=vtkSmartPointer<vtkLine>::New();
                    line->GetPointIds()->SetId(0, cnt);
                    line->GetPointIds()->SetId(1, cnt+sz[0]);
                    lines->InsertNextCell(line);
                }
                cnt++;
            }
        }



        // Create a polydata to store everything in
        vtkSmartPointer<vtkPolyData> linesPolyData=vtkSmartPointer<vtkPolyData>::New();
        // Add the points to the dataset
        linesPolyData->SetPoints(points);
        linesPolyData->GetPointData()->SetScalars(scalars);

        // Add the lines to the dataset
        linesPolyData->SetLines(lines);
        linesPolyData->GetCellData()->SetScalars(scalars);



        vtkSmartPointer<vtkNamedColors> colors=vtkSmartPointer<vtkNamedColors>::New();

        vtkSmartPointer<vtkPolyDataMapper> mapper=vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(linesPolyData);
        mapper->ScalarVisibilityOn();
        mapper->SetLookupTable(lut);
        mapper->SetScalarModeToUsePointData();
        mapper->SetColorModeToMapScalars();
        mapper->UseLookupTableScalarRangeOn();


        vtkSmartPointer<vtkActor> actor=vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetLineWidth(1);

        renderer->AddActor(actor);


        if(overlay_img)
        {
            ImageType3D::IndexType temp_index;
            temp_index[0]=0;
            temp_index[1]=curr_slice_y;
            temp_index[2]=0;
            ImageType3D::PointType orig;
            overlay_img->TransformIndexToPhysicalPoint(temp_index,orig);

            ImageType3D::DirectionType dir = overlay_img->GetDirection();
            ImageType3D::SpacingType spc= overlay_img->GetSpacing();

            vtkSmartPointer<vtkImageData> imageData =  vtkSmartPointer<vtkImageData>::New();
            imageData->SetDimensions(sz[0],1,sz[2]);
            imageData->SetOrigin(orig[0],orig[1],orig[2]);
            imageData->SetSpacing(spc[0],spc[1],spc[2]);

            imageData->SetDirectionMatrix(dir(0,0),dir(0,1),dir(0,2),dir(1,0),dir(1,1),dir(1,2),dir(2,0),dir(2,1),dir(2,2));
            imageData->AllocateScalars(VTK_FLOAT, 1);

            for(int k=0;k<sz[2];k++)
            {
                ImageType3D::IndexType ind3;
                ind3[1]=curr_slice_y;
                ind3[2]=k;

                for(int i=0;i<sz[0];i++)
                {
                    ind3[0]=i;
                    float* pixel = static_cast<float*>(imageData->GetScalarPointer(i,0,k));
                    pixel[0]= overlay_img->GetPixel(ind3)  *255;
                }
            }
            vtkSmartPointer<vtkImageActor> image_actor = vtkSmartPointer<vtkImageActor>::New();
            image_actor->SetInputData(imageData);
            image_actor->GetProperty()->SetOpacity(opac);
            image_actor->SetDisplayExtent(0,sz[0]-1,0,0,0,sz[2]-1);
            renderer->AddActor2D(image_actor);
        }
    }


    if(sag)
    {
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> lines=vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkFloatArray> scalars = vtkSmartPointer<vtkFloatArray>::New();
        scalars->SetName("Scalars");
        scalars->SetNumberOfComponents(1);

        long cnt=0;
        for(int k=0;k<sz[2];k++)
        {
            ImageType3D::IndexType ind3;
            ind3[0]=curr_slice_x;
            ind3[2]=k;

            for(int j=0;j<sz[1];j++)
            {
                ind3[1]=j;
                ImageType3D::PointType pt,pt_trans;
                field->TransformIndexToPhysicalPoint(ind3,pt);
                pt_trans=pt;
                if(display_warped)
                    pt_trans=trans->TransformPoint(pt);

                vnl_matrix_fixed<double,3,3> jac = ComputeJacobian(field,ind3);
                float mdet= vnl_det<double>(jac);
                mdet=log(mdet);
                scalars->InsertNextValue(mdet);


                points->InsertNextPoint(pt_trans[0],pt_trans[1],pt_trans[2]);
                if(j!=sz[1]-1)
                {
                    vtkSmartPointer<vtkLine> line=vtkSmartPointer<vtkLine>::New();
                    line->GetPointIds()->SetId(0, cnt);
                    line->GetPointIds()->SetId(1, cnt+1);
                    lines->InsertNextCell(line);

                }
                cnt++;
            }
        }

        cnt=0;
        for(int k=0;k<sz[2];k++)
        {
            for(int j=0;j<sz[1];j++)
            {
                if(k!=sz[2]-1)
                {
                    vtkSmartPointer<vtkLine> line=vtkSmartPointer<vtkLine>::New();
                    line->GetPointIds()->SetId(0, cnt);
                    line->GetPointIds()->SetId(1, cnt+sz[1]);
                    lines->InsertNextCell(line);
                }
                cnt++;
            }
        }



        // Create a polydata to store everything in
        vtkSmartPointer<vtkPolyData> linesPolyData=vtkSmartPointer<vtkPolyData>::New();
        // Add the points to the dataset
        linesPolyData->SetPoints(points);
        linesPolyData->GetPointData()->SetScalars(scalars);

        // Add the lines to the dataset
        linesPolyData->SetLines(lines);
        linesPolyData->GetCellData()->SetScalars(scalars);



        vtkSmartPointer<vtkNamedColors> colors=vtkSmartPointer<vtkNamedColors>::New();

        vtkSmartPointer<vtkPolyDataMapper> mapper=vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(linesPolyData);
        mapper->ScalarVisibilityOn();
        mapper->SetLookupTable(lut);
        mapper->SetScalarModeToUsePointData();
        mapper->SetColorModeToMapScalars();
        mapper->UseLookupTableScalarRangeOn();


        vtkSmartPointer<vtkActor> actor=vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetLineWidth(1);

        renderer->AddActor(actor);

        if(overlay_img)
        {
            ImageType3D::IndexType temp_index;
            temp_index[0]=curr_slice_x;
            temp_index[1]=0;
            temp_index[2]=0;
            ImageType3D::PointType orig;
            overlay_img->TransformIndexToPhysicalPoint(temp_index,orig);

            ImageType3D::DirectionType dir = overlay_img->GetDirection();
            ImageType3D::SpacingType spc= overlay_img->GetSpacing();

            vtkSmartPointer<vtkImageData> imageData =  vtkSmartPointer<vtkImageData>::New();
            imageData->SetDimensions(1,sz[1],sz[2]);
            imageData->SetOrigin(orig[0],orig[1],orig[2]);
            imageData->SetSpacing(spc[0],spc[1],spc[2]);
            imageData->SetDirectionMatrix(dir(0,0),dir(0,1),dir(0,2),dir(1,0),dir(1,1),dir(1,2),dir(2,0),dir(2,1),dir(2,2));
            imageData->AllocateScalars(VTK_FLOAT, 1);

            for(int k=0;k<sz[2];k++)
            {
                ImageType3D::IndexType ind3;
                ind3[0]=curr_slice_x;
                ind3[2]=k;

                for(int j=0;j<sz[1];j++)
                {
                    ind3[1]=j;
                    float* pixel = static_cast<float*>(imageData->GetScalarPointer(0,j,k));
                    pixel[0]= overlay_img->GetPixel(ind3)  *255;
                }
            }
            vtkSmartPointer<vtkImageActor> image_actor = vtkSmartPointer<vtkImageActor>::New();
            image_actor->SetInputData(imageData);
            image_actor->GetProperty()->SetOpacity(opac);
            image_actor->SetDisplayExtent(0,0,0,sz[1]-1,0,sz[2]-1);
            renderer->AddActor2D(image_actor);
        }


    }







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
        std::cout<<key<<std::endl;

        auto renderer=this->GetCurrentRenderer();
        auto renderWindow =renderer->GetRenderWindow();

        if (key == "a")
        {
            ax=!ax;
            MyRender(renderer,renderWindow);
        }
        if (key == "c")
        {
            cor=!cor;
            MyRender(renderer,renderWindow);
        }
        if (key == "s")
        {
            sag=!sag;
            MyRender(renderer,renderWindow);
        }

        if (key == "Up")
        {
            curr_slice_y++;
            if(curr_slice_y>sz[1]-1)
                curr_slice_y=sz[1]-1;

            MyRender(renderer,renderWindow);

        }
        if (key == "Down")
        {
            curr_slice_y--;
            if(curr_slice_y<0)
                curr_slice_y=0;
            MyRender(renderer,renderWindow);

        }
        if (key == "Left")
        {
            curr_slice_x--;
            if(curr_slice_x<0)
                curr_slice_x=0;
            MyRender(renderer,renderWindow);

        }
        if (key == "Right")
        {
            curr_slice_x++;
            if(curr_slice_x>sz[0]-1)
                curr_slice_x=sz[0]-1;
            MyRender(renderer,renderWindow);

        }

        if (key == "Prior")
        {
            curr_slice_z++;
            if(curr_slice_z>sz[2]-1)
                curr_slice_z=sz[2]-1;

            MyRender(renderer,renderWindow);
        }
        if (key == "Next")
        {
            curr_slice_z--;
            if(curr_slice_z<0)
                curr_slice_z=0;

            MyRender(renderer,renderWindow);
        }

        if (key == "KP_Add")
        {
            opac+=0.05;
            if(opac>1)
                opac=1;

            MyRender(renderer,renderWindow);
        }
        if (key == "KP_Subtract")
        {
            opac-=0.05;
            if(opac<0)
                opac=0;

            MyRender(renderer,renderWindow);
        }

        if (key == "space")
        {
            display_warped=!display_warped;
            MyRender(renderer,renderWindow);

            //renderer->GetActors()->GetNextActor()->GetMapper()->SetScalarVisibility(0);
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
        std::cout<<"Usage: WarpMesh  full_path_to_displacementfield.nii overlay_image.nii (optional)"<<std::endl;
        return EXIT_FAILURE;
    }

    vtkObject::GlobalWarningDisplayOff();
    ax=1;
    sag=1;
    cor=1;
    display_warped=1;
    opac=1;

    field=readImageD<DisplacementFieldType>(argv[1]);
    overlay_img=nullptr;
    if(argc>2)
        overlay_img= readImageD<ImageType3D>(argv[2]);

    sz= field->GetLargestPossibleRegion().GetSize();
    curr_slice_z = sz[2]/2;
    curr_slice_y = sz[1]/2;
    curr_slice_x = sz[0]/2;


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



    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =     vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    // windowToImageFilter->SetMagnification(2);
    windowToImageFilter->SetScale(2);
    windowToImageFilter->SetInputBufferTypeToRGB();
    windowToImageFilter->Update();

    std::ostringstream convert1;
    convert1 << curr_slice_x;
    std::ostringstream convert2;
    convert2 << curr_slice_y;
    std::ostringstream convert3;
    convert3 << curr_slice_z;


    std::string sl1 = convert1.str();
    std::string sl2 = convert2.str();
    std::string sl3 = convert3.str();


    std::string oname= std::string(argv[1]);
    oname = oname.substr(0,oname.find(".nii")) + std::string("_") + sl1 + std::string("_") + sl2 +  std::string("_") + sl3 + std::string(".png");


    vtkSmartPointer<vtkPNGWriter> writer =     vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(oname.c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();



    return EXIT_SUCCESS;

}

