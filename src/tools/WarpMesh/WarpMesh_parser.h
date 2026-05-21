#ifndef _WarpMesh_PARSER_h
#define _WarpMesh_PARSER_h


#include <iostream>
#include <vector>
#include <string>

#include "antsCommandLineParser.h"


class WarpMesh_PARSER : public itk::ants::CommandLineParser
{
public:
    WarpMesh_PARSER( int argc , char * argv[]);
    ~WarpMesh_PARSER();
    
    
     int getNFields();
     int getNAnatomicals();
     std::string getNthField(int n);
     std::string getNthAnatomical(int n);     
 


protected:
    void InitializeCommandLineOptions();

     
private:
    virtual void CreateParserandFillText(int argc , char * argv[] );
    
};





#endif
