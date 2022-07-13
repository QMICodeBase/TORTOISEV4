#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>


#include "defines.h"


int mainokan(int argc, char *argv[]);

ImageType3D::Pointer betokan(ImageType3D::Pointer filtered_img)
{
    float *filtered_mask= new float[ filtered_img->GetLargestPossibleRegion().GetSize()[0]*filtered_img->GetLargestPossibleRegion().GetSize()[1]*filtered_img->GetLargestPossibleRegion().GetSize()[2]];


    int sizes[3];
    sizes[0]=filtered_img->GetLargestPossibleRegion().GetSize()[0];
    sizes[1]=filtered_img->GetLargestPossibleRegion().GetSize()[1];
    sizes[2]=filtered_img->GetLargestPossibleRegion().GetSize()[2];

    float res[3];
    res[0]= filtered_img->GetSpacing()[0];
    res[1]= filtered_img->GetSpacing()[1];
    res[2]= filtered_img->GetSpacing()[2];


    int argc=7;
    char *argv[7]={0};
    argv[0]= (char *) sizes;
    argv[1]= (char *) res;
    argv[2]= (char *) (filtered_img->GetBufferPointer());
    argv[3]= (char *) filtered_mask;

    char p1[3]="-f";
    float p2=0.3;


    argv[4]= p1;
    argv[5]= (char*) &p2;

    char p3[3]="-v";

    argv[6]=p3;
    mainokan(argc,argv);


    ImageType3D::Pointer out_img= ImageType3D::New();
    out_img->SetRegions(filtered_img->GetLargestPossibleRegion());
    out_img->SetSpacing(filtered_img->GetSpacing());        
    out_img->SetDirection(filtered_img->GetDirection());
    out_img->SetOrigin(filtered_img->GetOrigin());
    out_img->GetPixelContainer()->SetImportPointer(filtered_mask,filtered_img->GetLargestPossibleRegion().GetSize()[0]*filtered_img->GetLargestPossibleRegion().GetSize()[1]*filtered_img->GetLargestPossibleRegion().GetSize()[2],false);


    return out_img;


}
