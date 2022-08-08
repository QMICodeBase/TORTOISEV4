

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;


#include "defines.h"


int main(int argc, char*argv[])
{
    if(argc<3)    
    {
        std::cout<<"Usage: Set4DImageDirection full_path_to_4d_image full_path_to_output_name    16_directions (first row- then secon row... If not provided, identity is put)"<<std::endl;
        return 0;
    }

    ImageType4D::Pointer image4D= readImageD<ImageType4D>(argv[1]);
    

    ImageType4D::DirectionType dir;
    dir.SetIdentity();

    if(argc>3)
    {
        dir(0,0) = atof(argv[3]);
        dir(0,1) = atof(argv[4]);
        dir(0,2) = atof(argv[5]);
        dir(0,3) = atof(argv[6]);
        dir(1,0) = atof(argv[7]);
        dir(1,1) = atof(argv[8]);
        dir(1,2) = atof(argv[9]);
        dir(1,3) = atof(argv[10]);
        dir(2,0) = atof(argv[11]);
        dir(2,1) = atof(argv[12]);
        dir(2,2) = atof(argv[13]);
        dir(2,3) = atof(argv[14]);
        dir(3,0) = atof(argv[15]);
        dir(3,1) = atof(argv[16]);
        dir(3,2) = atof(argv[17]);
        dir(3,3) = atof(argv[18]);
    }

    image4D->SetDirection(dir);                   

    writeImageD<ImageType4D>(image4D,argv[2]);
    
    return EXIT_SUCCESS;

}
