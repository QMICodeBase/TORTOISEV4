#include "../utilities/read_3Dvolume_from_4D.h"
#include "extract_image_parser.h"
#include "defines.h"

int main(int argc, char * argv[])
{
    Extract_Image_PARSER *parser = new Extract_Image_PARSER(argc,argv);

    std::string oname;
    if(parser->getOutputImageName()=="")
    {
        std::string nm=parser->getInputImageName();
        char buf[2000];
        sprintf(buf,"_V%.3d.nii",parser->getVolId());
        oname=nm.substr(0,nm.find(".nii")) + std::string(buf);
    }
    else
    {
        oname=parser->getOutputImageName();
    }

    ImageType3D::Pointer img = read_3D_volume_from_4D(parser->getInputImageName(),parser->getVolId());

    writeImageD<ImageType3D>(img,oname);

    return EXIT_SUCCESS;
}
