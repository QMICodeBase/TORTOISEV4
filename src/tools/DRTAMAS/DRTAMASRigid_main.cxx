#include "defines.h"


#include "DRTAMAS_parser.h"
#include "DRTAMASRigid.h"

int main(int argc,char *argv[])
{
    DRTAMAS_PARSER *parser= new DRTAMAS_PARSER(argc,argv);


    for(int i=0;i<argc;i++)
         std::cout << argv[i]<< " ";
     std::cout<<std::endl;
     std::cout<<std::endl;


     try
     {
         DRTAMASRigid *my_program= new DRTAMASRigid();
         my_program->SetParser(parser);
         my_program->Process2();
     }
     catch( itk::ExceptionObject & err )
     {
         std::cerr << "Exception Object caught: " << std::endl;
         std::cerr << err << std::endl;
         return EXIT_FAILURE;
     }



     return EXIT_SUCCESS;

}
