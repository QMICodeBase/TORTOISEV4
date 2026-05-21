
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
#include "vtkCamera.h"
#include <vtkGraphicsFactory.h>
#include "vtkPolyLine.h"
#include "vtkImageProperty.h"
#include "vtkLight.h"

#include "vnl/algo/vnl_symmetric_eigensystem.h"

#include "itkImageRegionIteratorWithIndex.h"
#include <vtkInteractorStyleTrackballCamera.h>
#include "vtkLine.h"
#include "vtkNamedColors.h"
#include "vtkProperty.h"
#include "vtkLookupTable.h"
#include "vtkCellData.h"


#include <vtkCylinderSource.h>
#include <vtkMapper.h>
#include <vtkNamedColors.h>
#include <vtkPolyDataMapper.h>

#include <ctime>

void GetCurrentCameraProperties(vtkRenderer* renderer) {
    // Get the current active camera
    vtkCamera* camera = renderer->GetActiveCamera();

    if (camera) {
        // Get camera position (world coordinates)
        double position[3];
        camera->GetPosition(position);

        // Get camera focal point (world coordinates)
        double focalPoint[3];
        camera->GetFocalPoint(focalPoint);

        // Get camera view up vector
        double viewUp[3];
        camera->GetViewUp(viewUp);

        // Get view angle (in degrees, for perspective projection)
        double viewAngle = camera->GetViewAngle();

        // Get clipping range (near and far planes)
        double clippingRange[2];
        camera->GetClippingRange(clippingRange);

        // Print the properties
        std::cout << "Camera Properties:" << std::endl;
        std::cout << "  Position:    (" << position[0] << ", " << position[1] << ", " << position[2] << ")" << std::endl;
        std::cout << "  Focal Point: (" << focalPoint[0] << ", " << focalPoint[1] << ", " << focalPoint[2] << ")" << std::endl;
        std::cout << "  View Up:     (" << viewUp[0] << ", " << viewUp[1] << ", " << viewUp[2] << ")" << std::endl;
        std::cout << "  View Angle:  " << viewAngle << " degrees" << std::endl;
        std::cout << "  Clipping Range: [" << clippingRange[0] << ", " << clippingRange[1] << "]" << std::endl;
    } else {
        std::cerr << "Error: No active camera found in the renderer." << std::endl;
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
        }
        if (key == "c")
        {
        }
        if (key == "s")
        {
        }

        if (key == "Up")
        {

        }
        if (key == "Down")
        {

        }
        if (key == "Left")
        {

        }
        if (key == "Right")
        {

        }

        if (key == "Prior")
        {
        }
        if (key == "Next")
        {
        }

        if (key == "KP_Add")
        {
        }
        if (key == "KP_Subtract")
        {
        }

        if (key == "space")
        {
            GetCurrentCameraProperties(renderer);
        }

        renderWindow->Render();


        // Forward events
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }
};
vtkStandardNewMacro(KeyPressInteractorStyle);




int main(int argc, char *argv[])
{
    if(argc<3)
    {
        std::cout<<"Usage: Cylinders rad  Nlines"<<std::endl;
        return EXIT_FAILURE;
    }

    vtkObject::GlobalWarningDisplayOff();

    int Nlines= atoi(argv[2]);
    float rad= atof(argv[1]);


    vtkSmartPointer<vtkRenderer> renderer =     vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow  =     vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(900, 900);
    renderWindow->SetOffScreenRendering(0);
    renderWindow->AddRenderer(renderer);


    float height=10.;



    vtkNew<vtkCylinderSource> cylinderSource;
    cylinderSource->SetRadius(rad);
    cylinderSource->SetHeight(height);
    cylinderSource->SetResolution(50); // Number of facets around the cylinder

    vtkNew<vtkPolyDataMapper> cylinderMapper;
    cylinderMapper->SetInputConnection(cylinderSource->GetOutputPort());


    vtkNew<vtkActor> cylinderActor;
    cylinderActor->SetMapper(cylinderMapper);

    vtkNew<vtkNamedColors> colors;
    cylinderActor->GetProperty()->SetColor(colors->GetColor3d("Red").GetData());
    cylinderActor->GetProperty()->SetOpacity(0.9);
    cylinderActor->GetProperty()->SetSpecular(0.8); // Specular highlights
    cylinderActor->GetProperty()->SetSpecularPower(10.0);
    cylinderActor->GetProperty()->SetDiffuse(0.8);
    cylinderActor->GetProperty()->SetAmbient(0.2);

    auto light = vtkSmartPointer<vtkLight>::New();
    light->SetPosition(0, -1*height*2, 0); // Position the light source
    light->SetColor(colors->GetColor3d("White").GetData()); // White light
    light->SetIntensity(1.0); // Light intensity
    renderer->AddLight(light);

    auto light2 = vtkSmartPointer<vtkLight>::New();
    light2->SetPosition(1, 0, 1); // Position the light source
    light2->SetColor(colors->GetColor3d("White").GetData()); // White light
    light2->SetIntensity(0.7); // Light intensity
    renderer->AddLight(light2);

    srand(time(0));

    for(int n=0;n<Nlines;n++)
    {
        float r = rad * sqrt(1.*rand()/RAND_MAX);
       // float r = rad * (1.*rand()/RAND_MAX);
        float theta = 1.*rand()/RAND_MAX * 2 * 3.141592;

        float x = 0 + r * cos(theta);
        float z = 0 + r * sin(theta);

        float ml=0.6;
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->InsertNextPoint(x,  -1.* height*ml,z); // Point 1
        points->InsertNextPoint(x,  height*ml,z); // Point 2

        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, 0); // Connects to the first point in vtkPoints
        line->GetPointIds()->SetId(1, 1); // Connects to the second point in vtkPoints

        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->SetPoints(points);
        vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
        lines->InsertNextCell(line);
        polyData->SetLines(lines);

        vtkSmartPointer<vtkPolyDataMapper> lmapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        lmapper->SetInputData(polyData);

        vtkSmartPointer<vtkActor> lactor = vtkSmartPointer<vtkActor>::New();
        lactor->SetMapper(lmapper);
        lactor->GetProperty()->SetLineWidth(4.0);

        lactor->GetProperty()->SetColor(colors->GetColor3d("Blue").GetData());
        renderer->AddActor(lactor);


    }





    renderer->AddActor(cylinderActor);
    renderer->SetBackground(colors->GetColor3d("White").GetData());

    // An interactor
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =  vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow ( renderWindow );

    // Set the custom stype to use for interaction.
    vtkSmartPointer<KeyPressInteractorStyle> style =        vtkSmartPointer<KeyPressInteractorStyle>::New();
    style->SetDefaultRenderer(renderer);
    renderWindowInteractor->SetInteractorStyle( style );

    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    camera->SetPosition(10.2961, -15.8318, -7.63797);
    camera->SetFocalPoint(-0.526641, -0.890972, 0.2768);
    camera->SetViewUp(0.837029, 0.421505, 0.348877);
    camera->SetViewAngle(30); // e.g., camera->SetViewAngle(30.0);
    camera->SetClippingRange(9.39932, 34.1725); // e.g., camera->SetClippingRange(0.1, 1000.0);

        renderer->SetActiveCamera(camera);


    renderWindow->Render();
    renderWindowInteractor->Initialize();
    renderWindowInteractor->Start();



    return EXIT_SUCCESS;

}
