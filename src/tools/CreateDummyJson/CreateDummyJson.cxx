#include "defines.h"
#include "CreateDummyJson_parser.h"


#include "../utilities/read_3Dvolume_from_4D.h"


int main(int argc, char *argv[])
{

    CreateDummyJson_PARSER *parser= new CreateDummyJson_PARSER(argc,argv);


    json my_json;

    int MBf= parser->getMBFactor();
    std::string phase= parser->getPhaseEncoding();
    float PF= parser->getPF();


    my_json["PartialFourier"] = PF;
    my_json["PhaseEncodingDirection"]= phase;
    my_json["MultibandAccelerationFactor"]=MBf;
    if(parser->getBigDelta()!=-1)
        my_json["BigDelta"]=parser->getBigDelta();
    if(parser->getSmallDelta()!=-1)
        my_json["SmallDelta"]=parser->getSmallDelta();


    ImageType3D::Pointer first_vol = read_3D_volume_from_4D(parser->getInputImageName(),0);
    ImageType3D::SizeType sz= first_vol->GetLargestPossibleRegion().GetSize();

    std::vector<float> times;
    times.resize(sz[2]);

    int Nexc= sz[2]/MBf;

    for(int k=0;k<Nexc;k++)
    {
        for(int m=0;m<MBf;m++)
        {
            int id = Nexc*m + k;
            times[id]=k;
        }
    }

    my_json["SliceTiming"]= times;

    std::string name= parser->getInputImageName();
    std::string json_name= name.substr(0,name.find(".nii")) + ".json";
    std::ofstream out_json(json_name);
    out_json << std::setw(4) << my_json << std::endl;
    out_json.close();

}
