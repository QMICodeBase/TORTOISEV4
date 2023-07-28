#include "defines.h"


#include "DRTAMAS_parser.h"
#include "DRTAMAS.h"

int main(int argc,char *argv[])
{
    DRTAMAS_PARSER *parser= new DRTAMAS_PARSER(argc,argv);


    for(int i=0;i<argc;i++)
         std::cout << argv[i]<< " ";
     std::cout<<std::endl;
     std::cout<<std::endl;


     try
     {
         DRTAMAS *my_program= new DRTAMAS();
         my_program->SetParser(parser);
         my_program->Process();
     }
     catch( itk::ExceptionObject & err )
     {
         std::cerr << "Exception Object caught: " << std::endl;
         std::cerr << err << std::endl;
         return EXIT_FAILURE;
     }



     return EXIT_SUCCESS;

}
