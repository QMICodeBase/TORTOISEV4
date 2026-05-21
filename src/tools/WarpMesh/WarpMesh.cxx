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
#include "WarpMesh_parser.h"

#include <ctime>

typedef double RealType;
typedef  vnl_matrix_fixed< RealType, 3, 3 > InternalMatrixType;


int curr_slice_z, curr_slice_y,curr_slice_x;
ImageType3D::SizeType sz;

std::array<bool,9> running;
float opac;
bool display_warped;

bool display_wire;
bool display_anat;

std::vector<vtkSmartPointer<vtkRenderer> > renderers;
std::vector<vtkSmartPointer<vtkRenderWindow> > renWins;

std::vector<DisplacementFieldType::Pointer> fields;
std::vector<ImageType3D::Pointer> anatomicals;

int curr_anatomical;


void MyRender(int i, vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkRenderWindow> renderWindow );


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

        int i=0;
        for(int mi=0; mi<renderers.size();mi++)
        {
            if(renderer== renderers[mi])
            {
                i=mi;
                break;
            }
        }

        if (key == "Up")
        {
            curr_slice_y++;
            if(curr_slice_y>sz[1]-1)
                curr_slice_y=sz[1]-1;

            for(int mi=i;mi<renderers.size();mi+=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
            for(int mi=i-3;mi>=0;mi-=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }
        if (key == "Down")
        {
            curr_slice_y--;
            if(curr_slice_y<0)
                curr_slice_y=0;

            for(int mi=i;mi<renderers.size();mi+=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
            for(int mi=i-3;mi>=0;mi-=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }
        if (key == "Left")
        {
            curr_slice_x--;
            if(curr_slice_x<0)
                curr_slice_x=0;

            for(int mi=i;mi<renderers.size();mi+=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
            for(int mi=i-3;mi>=0;mi-=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }
        if (key == "Right")
        {
            curr_slice_x++;
            if(curr_slice_x>sz[0]-1)
                curr_slice_x=sz[0]-1;

            for(int mi=i;mi<renderers.size();mi+=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
            for(int mi=i-3;mi>=0;mi-=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }

        if (key == "Prior")
        {
            curr_slice_z++;
            if(curr_slice_z>sz[2]-1)
                curr_slice_z=sz[2]-1;

            for(int mi=i;mi<renderers.size();mi+=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
            for(int mi=i-3;mi>=0;mi-=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }
        if (key == "Next")
        {
            curr_slice_z--;
            if(curr_slice_z<0)
                curr_slice_z=0;


            for(int mi=i;mi<renderers.size();mi+=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
            for(int mi=i-3;mi>=0;mi-=3)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }

        }

        if (key == "KP_Add")
        {
            opac+=0.2;
            if(opac>1)
                opac=1;

            for(int mi=0;mi<renderers.size();mi++)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }

        }
        if (key == "KP_Subtract")
        {
            opac-=0.2;
            if(opac<0)
                opac=0;

            for(int mi=0;mi<renderers.size();mi++)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }

        }

        if (key == "space")
        {
            display_warped=!display_warped;
            //renderer->GetActors()->GetNextActor()->GetMapper()->SetScalarVisibility(0);

            for(int mi=0;mi<renderers.size();mi++)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }

        if (key == "e" || key == "q" || key == "E" || key == "Q")
        {
            exit(EXIT_SUCCESS);
        }

        if(key=="1" ||key=="2" ||key=="3" ||key=="4" ||key=="5" ||key=="6" ||key=="7" ||key=="8" ||key=="9"  )
        {
            const char *aa= key.c_str();
            curr_anatomical= aa[0]-'0' -1;
            if(curr_anatomical>anatomicals.size()-1)
                curr_anatomical=anatomicals.size()-1;

            for(int mi=0;mi<renderers.size();mi++)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }

        if(key=="w"  )
        {
            display_wire=!display_wire;

            for(int mi=0;mi<renderers.size();mi++)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }
        if(key=="a"  )
        {
            display_anat=!display_anat;

            for(int mi=0;mi<renderers.size();mi++)
            {
                if(running[mi])
                {
                    MyRender(mi,renderers[mi],renWins[mi]);
                    renWins[mi]->Render();
                }
            }
        }



        std::cout<<key<<std::endl;



        // Forward events
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }
public:
    bool*        status=nullptr;
};
vtkStandardNewMacro(KeyPressInteractorStyle);

class MyExitCommand : public vtkCommand
{
public:
    static MyExitCommand* New() { return new MyExitCommand; }
    void Execute(vtkObject* caller, unsigned long eventId, void* callData) override
    {
        if (eventId == vtkCommand::ExitEvent)
        {
            std::cout << "Window is closing!" << std::endl;
            // Perform any cleanup or actions needed before exit
            vtkRenderWindowInteractor* iren = static_cast<vtkRenderWindowInteractor*>(caller);
            auto is =iren->GetInteractorStyle();
            KeyPressInteractorStyle *my_style = static_cast<KeyPressInteractorStyle*>(is);
            *(my_style->status)=false;
            iren->GetRenderWindow()->Finalize(); // Clean up OpenGL context
            iren->TerminateApp(); // Stop the interactor loop
        }
    }
};


void MyRender(int i, vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkRenderWindow> renderWindow )
{
    renderer->RemoveAllViewProps();

    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetNumberOfColors(256);
    lut->SetNumberOfTableValues(256);

    double mn=-0.5;
    double mx=0.5;
    int N_half= 204800;
    lut->SetTableRange(mn, mx);
 //   lut->SetNumberOfTableValues(2*N_half+1);
    lut->Build();

    /*
    lut->SetTableValue(0,0,0,1);
    for(int c=1;c<127;c++)
        lut->SetTableValue(c,0, c/126.,1-c/126.);

    lut->SetTableValue(127,0,1,0);
    for(int c=128;c<255;c++)
        lut->SetTableValue(c, (c-127)/127., 1- (c-127)/127.,0);
    lut->SetTableValue(255,1,0,0);
*/



    lut->SetTableValue(0,0,0,1);
    for(int c=1;c<127;c++)
        lut->SetTableValue(c,c/126., c/126.,1.);

    lut->SetTableValue(127,1,1,1);
    for(int c=128;c<255;c++)
        lut->SetTableValue(c, 1, -c/127. +2, 1, -c/127. +2);
    lut->SetTableValue(255,1,0,0);


    /*
    for(int c=0;c<N_half;c++)
        lut->SetTableValue(c,0, 0.,1- 0.9*c/(N_half-1)+1);
    lut->SetTableValue(N_half,1,1,1);
    for(int c=N_half+1;c<=2*N_half;c++)
        lut->SetTableValue(c,0.9*c/(N_half-1)-0.9*(N_half+1)/(N_half-1)+0.1, 0.,0);
*/



    int field_id= i/3;
    int view_id= i%3;



    DisplacementFieldType::Pointer field = fields[field_id];

    using DisplacementFieldTransformType = itk::DisplacementFieldTransform<double,3>;
    DisplacementFieldTransformType::Pointer trans= DisplacementFieldTransformType::New();
    trans->SetDisplacementField(field);

    if(view_id==0)
    {
        if(display_wire)
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
                    if(mdet<=0)
                    {
                        double rnd=1E-5 * (1.*rand() /RAND_MAX -0.5);
                        mdet=1+ rnd;
                    }
                    if(fabs(mdet-1)<1E-2)
                    {
                        DisplacementFieldType::PixelType vec= field->GetPixel(ind3);
                        double nrm = vec.GetNorm();
                        if(nrm!=0)
                        {
                            double rnd=1E-2 * (1.*rand() /RAND_MAX -0.5);
                            mdet=1+ rnd;
                        }
                    }

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
        }


        if(anatomicals.size() && display_anat)
        {
            ImageType3D::Pointer overlay_img= anatomicals[curr_anatomical];

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

    if(view_id==1)
    {
        if(display_wire)
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
                    if(mdet<=0)
                    {
                        double rnd=1E-5 * (1.*rand() /RAND_MAX -0.5);
                        mdet=1+ rnd;
                    }
                    if(fabs(mdet-1)<1E-2)
                    {
                        DisplacementFieldType::PixelType vec= field->GetPixel(ind3);
                        double nrm = vec.GetNorm();
                        if(nrm!=0)
                        {
                            double rnd=1E-2 * (1.*rand() /RAND_MAX -0.5);
                            mdet=1+ rnd;
                        }
                    }

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
        }


        if(anatomicals.size() && display_anat)
        {
            ImageType3D::Pointer overlay_img= anatomicals[curr_anatomical];

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


    if(view_id==2)
    {
        if(display_wire)
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
                    if(mdet<=0)
                    {
                        double rnd=1E-5 * (1.*rand() /RAND_MAX -0.5);
                        mdet=1+ rnd;
                    }
                    if(fabs(mdet-1)<1E-2)
                    {
                        DisplacementFieldType::PixelType vec= field->GetPixel(ind3);
                        double nrm = vec.GetNorm();
                        if(nrm!=0)
                        {
                            double rnd=1E-2 * (1.*rand() /RAND_MAX -0.5);
                            mdet=1+ rnd;
                        }
                    }

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
        }

        if(anatomicals.size() && display_anat)
        {
            ImageType3D::Pointer overlay_img= anatomicals[curr_anatomical];
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




int main(int argc, char *argv[])
{
    WarpMesh_PARSER *parser= new WarpMesh_PARSER(argc,argv);

      srand(time(0));

    int Nfields= parser->getNFields();
    int Nanatomicals = parser->getNAnatomicals();

    for(int n=0;n<Nfields;n++)
        fields.push_back(readImageD<DisplacementFieldType>(parser->getNthField(n)));
    for(int n=0;n<Nanatomicals;n++)
        anatomicals.push_back(readImageD<ImageType3D>(parser->getNthAnatomical(n)));

    for(int n=0;n<Nanatomicals;n++)
    {
        double mx =-1E10;
        itk::ImageRegionIteratorWithIndex<ImageType3D> it(anatomicals[n],anatomicals[n]->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            if(it.Get()>mx)
                mx=it.Get();
        }
        if(mx >0)
        {
            for(it.GoToBegin();!it.IsAtEnd();++it)
            {
                it.Set(it.Get()/mx);
            }
        }
    }


    vtkObject::GlobalWarningDisplayOff();    
    display_warped=1;
    opac=1;
    display_wire=1;
    display_anat=1;



    sz= fields[0]->GetLargestPossibleRegion().GetSize();
    curr_slice_z = sz[2]/2;
    curr_slice_y = sz[1]/2;
    curr_slice_x = sz[0]/2;

    curr_anatomical=0;

    int Ntotalwindow = Nfields *3;

    std::vector<vtkSmartPointer<vtkRenderWindowInteractor>> interactors;
    std::vector<vtkSmartPointer<KeyPressInteractorStyle>> styles;

    int width= 800;
    int height=800;
    auto dx = 20;
    auto dy = 40;
    auto w = width + dx;
    auto h = height + dy;


    renderers.resize(Ntotalwindow);
    renWins.resize(Ntotalwindow);

    for(int i=0;i<Ntotalwindow;i++)
    {
        vtkNew<vtkRenderWindow> renderWindow;
        renderWindow->SetSize(width, height);

        vtkNew<vtkRenderer> renderer;
        vtkCamera* camera = renderer->GetActiveCamera();

        renderers[i]=renderer;
        renWins[i]=renderWindow;

        if(i/3==0)
        {
            if(i%3 ==0)   //axial
            {
                ImageType3D::IndexType ind3;
                ind3[0]=sz[0]/2;
                ind3[1]=sz[1]/2;
                ind3[2]=sz[2]*3;
                ImageType3D::PointType cam_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,cam_pt);

                ind3[0]=sz[0]/2;
                ind3[1]=sz[1]/2;
                ind3[2]=sz[2]/2;
                ImageType3D::PointType slice_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,slice_pt);

                ind3[0]=sz[0]/2;
                ind3[1]=0;
                ind3[2]=sz[2]/2;
                ImageType3D::PointType top_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,top_pt);
                vnl_vector<double> vec(3);
                vec[0]=top_pt[0]-slice_pt[0];
                vec[1]=top_pt[1]-slice_pt[1];
                vec[2]=top_pt[2]-slice_pt[2];
                vec=vec.normalize();

                camera->SetPosition(cam_pt[0],cam_pt[1],cam_pt[2]);
                camera->SetFocalPoint(slice_pt[0],slice_pt[1], slice_pt[2]);
                camera->SetViewUp(vec[0],vec[1],vec[2]);
                camera->SetViewAngle(30); // e.g., camera->SetViewAngle(30.0);
                //camera->SetClippingRange(9.39932, 34.1725);

            }
            if(i%3 ==1)  //coronal
            {
                ImageType3D::IndexType ind3;
                ind3[0]=sz[0]/2;
                ind3[1]=sz[1]*3;
                ind3[2]=sz[2]/2;
                ImageType3D::PointType cam_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,cam_pt);

                ind3[0]=sz[0]/2;
                ind3[1]=sz[1]/2;
                ind3[2]=sz[2]/2;
                ImageType3D::PointType slice_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,slice_pt);

                ind3[0]=sz[0]/2;
                ind3[1]=sz[1]/2;
                ind3[2]=sz[2]-1;
                ImageType3D::PointType top_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,top_pt);
                vnl_vector<double> vec(3);
                vec[0]=top_pt[0]-slice_pt[0];
                vec[1]=top_pt[1]-slice_pt[1];
                vec[2]=top_pt[2]-slice_pt[2];
                vec=vec.normalize();

                camera->SetPosition(cam_pt[0],cam_pt[1],cam_pt[2]);
                camera->SetFocalPoint(slice_pt[0],slice_pt[1], slice_pt[2]);
                camera->SetViewUp(vec[0],vec[1],vec[2]);
                camera->SetViewAngle(30); // e.g., camera->SetViewAngle(30.0);
                //camera->SetClippingRange(9.39932, 34.1725);

            }
            if(i%3 ==2) //sagittal
            {
                ImageType3D::IndexType ind3;
                ind3[0]=sz[0]*3;
                ind3[1]=sz[1]/2;
                ind3[2]=sz[2]/2;
                ImageType3D::PointType cam_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,cam_pt);

                ind3[0]=sz[0]/2;
                ind3[1]=sz[1]/2;
                ind3[2]=sz[2]/2;
                ImageType3D::PointType slice_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,slice_pt);

                ind3[0]=sz[0]/2;
                ind3[1]=sz[1]/2;
                ind3[2]=sz[2]-1;
                ImageType3D::PointType top_pt;
                fields[0]->TransformIndexToPhysicalPoint(ind3,top_pt);
                vnl_vector<double> vec(3);
                vec[0]=top_pt[0]-slice_pt[0];
                vec[1]=top_pt[1]-slice_pt[1];
                vec[2]=top_pt[2]-slice_pt[2];
                vec=vec.normalize();

                camera->SetPosition(cam_pt[0],cam_pt[1],cam_pt[2]);
                camera->SetFocalPoint(slice_pt[0],slice_pt[1], slice_pt[2]);
                camera->SetViewUp(vec[0],vec[1],vec[2]);
                camera->SetViewAngle(30); // e.g., camera->SetViewAngle(30.0);
                //camera->SetClippingRange(9.39932, 34.1725);

            }
        }
        else
        {
            renderer->SetActiveCamera(renderers[i%3]->GetActiveCamera());
        }

        renderWindow->AddRenderer(renderer);
        std::vector<std::string> nms = {"axial", "coronal", "sagittal"};

        std::stringstream ss;
        ss << "Field " << i/3 << " " << nms[i%3];
        renderWindow->SetWindowName(ss.str().c_str());

        vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
        interactors.push_back(renderWindowInteractor);

        renderWindowInteractor->SetRenderWindow(renderWindow);
        renderWindow->Render();
        renderWindow->SetPosition((i % 3) * w,  (i / Nfields) * h);


        MyRender(i,renderer,renderWindow);

        running[i] = true;
        vtkNew<KeyPressInteractorStyle> style;
        styles.push_back(style);
        styles[i]->status = &running[i];
        interactors[i]->SetInteractorStyle(styles[i]);
        styles[i]->SetCurrentRenderer(renderer);

        vtkNew<MyExitCommand> exitCommand;
        interactors[i]->AddObserver(vtkCommand::ExitEvent, exitCommand);

    }


    // Changes in any window will be simultaneously reflected in the other
    // windows.
    interactors[0]->Initialize();
    // If all are running then process the commands.
    while(1)
    //while (std::all_of(running.begin(), running.end(),  [](bool i) { return i == true; }))
    {
        for (unsigned int i = 0; i < Ntotalwindow; i++)
        {
            if (running[i])
            {
                interactors[i]->ProcessEvents();
                interactors[i]->Render();
            }
            else
            {
                interactors[i]->TerminateApp();
                std::cout << "Window " << i << " has stopped running." << std::endl;
            }
        }
    }


    delete parser;
    return EXIT_SUCCESS;

}

