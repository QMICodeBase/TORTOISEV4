#include "FlipSwapBMatrix_parser.h"
#include "defines.h"
#include "../utilities/read_bmatrix_file.h"


 int sgn(int val)
{
     if(val ==0)
         return 1;
     else
        return (int(0) < val) - (val < int(0));
}

int main(int argc, char * argv[])
{

    FlipSwapBMatrix_PARSER *parser= new FlipSwapBMatrix_PARSER(argc,argv);

    std::string bmtxtfile = parser->getInputBMatrix();


    vnl_matrix<double> Bmatrix = read_bmatrix_file(bmtxtfile);
    int Nvols=Bmatrix.rows();


    vnl_matrix<double> new_Bmatrix=Bmatrix;
    new_Bmatrix.fill(0);


    std::string nx= parser->getNewX();
    std::string ny= parser->getNewY();
    std::string nz= parser->getNewZ();


    int mappings[6];

    {
        if(nx.find("x")!=std::string::npos)
        {
            mappings[0]=0;
        }
        if(nx.find("y")!=std::string::npos)
        {
            mappings[0]=3;
        }

        if(nx.find("z")!=std::string::npos)
        {
            mappings[0]=5;
        }
    }
    {
        if(ny.find("x")!=std::string::npos)
        {
            mappings[3]=0;
        }
        if(ny.find("y")!=std::string::npos)
        {
            mappings[3]=3;
        }

        if(ny.find("z")!=std::string::npos)
        {
            mappings[3]=5;
        }
    }
    {
        if(nz.find("x")!=std::string::npos)
        {
            mappings[5]=0;
        }
        if(nz.find("y")!=std::string::npos)
        {
            mappings[5]=3;
        }

        if(nz.find("z")!=std::string::npos)
        {
            mappings[5]=5;
        }
    }


    {
        if( (nx.find("x")!=std::string::npos) && (ny.find("y")!=std::string::npos))
        {
            mappings[1]=1;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            mappings[1]*=mult;
        }
        if( (nx.find("x")!=std::string::npos) && (ny.find("z")!=std::string::npos))
        {
            mappings[1]=2;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            mappings[1]*=mult;
        }


        if( (nx.find("y")!=std::string::npos) && (ny.find("x")!=std::string::npos))
        {
            mappings[1]=1;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            mappings[1]*=mult;
        }
        if( (nx.find("y")!=std::string::npos) && (ny.find("z")!=std::string::npos))
        {
            mappings[1]=4;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            mappings[1]*=mult;
        }


        if( (nx.find("z")!=std::string::npos) && (ny.find("x")!=std::string::npos))
        {
            mappings[1]=2;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            mappings[1]*=mult;
        }
        if( (nx.find("z")!=std::string::npos) && (ny.find("y")!=std::string::npos))
        {
            mappings[1]=4;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            mappings[1]*=mult;
        }
    }










    {
        if( (nx.find("x")!=std::string::npos) && (nz.find("y")!=std::string::npos))
        {
            mappings[2]=1;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[2]*=mult;
        }
        if( (nx.find("x")!=std::string::npos) && (nz.find("z")!=std::string::npos))
        {
            mappings[2]=2;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[2]*=mult;
        }


        if( (nx.find("y")!=std::string::npos) && (nz.find("x")!=std::string::npos))
        {
            mappings[2]=1;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[2]*=mult;
        }
        if( (nx.find("y")!=std::string::npos) && (nz.find("z")!=std::string::npos))
        {
            mappings[2]=4;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[2]*=mult;
        }


        if( (nx.find("z")!=std::string::npos) && (nz.find("x")!=std::string::npos))
        {
            mappings[2]=2;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[2]*=mult;
        }
        if( (nx.find("z")!=std::string::npos) && (nz.find("y")!=std::string::npos))
        {
            mappings[2]=4;
            int mult=1;
            if(nx.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[2]*=mult;
        }
    }



    {
        if( (ny.find("x")!=std::string::npos) && (nz.find("y")!=std::string::npos))
        {
            mappings[4]=1;
            int mult=1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[4]*=mult;
        }
        if( (ny.find("x")!=std::string::npos) && (nz.find("z")!=std::string::npos))
        {
            mappings[4]=2;
            int mult=1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[4]*=mult;
        }


        if( (ny.find("y")!=std::string::npos) && (nz.find("x")!=std::string::npos))
        {
            mappings[4]=1;
            int mult=1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[4]*=mult;
        }
        if( (ny.find("y")!=std::string::npos) && (nz.find("z")!=std::string::npos))
        {
            mappings[4]=4;
            int mult=1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[4]*=mult;
        }


        if( (ny.find("z")!=std::string::npos) && (nz.find("x")!=std::string::npos))
        {
            mappings[4]=2;
            int mult=1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[4]*=mult;
        }
        if( (ny.find("z")!=std::string::npos) && (nz.find("y")!=std::string::npos))
        {
            mappings[4]=4;
            int mult=1;
            if(ny.find("-")!=std::string::npos)
                mult*=-1;
            if(nz.find("-")!=std::string::npos)
                mult*=-1;
            mappings[4]*=mult;
        }
    }


   new_Bmatrix.set_column(0, Bmatrix.get_column(abs(mappings[0])) * sgn(mappings[0])    );
   new_Bmatrix.set_column(1, Bmatrix.get_column(abs(mappings[1])) * sgn(mappings[1])    );
   new_Bmatrix.set_column(2, Bmatrix.get_column(abs(mappings[2])) * sgn(mappings[2])    );
   new_Bmatrix.set_column(3, Bmatrix.get_column(abs(mappings[3])) * sgn(mappings[3])    );
   new_Bmatrix.set_column(4, Bmatrix.get_column(abs(mappings[4])) * sgn(mappings[4])    );
   new_Bmatrix.set_column(5, Bmatrix.get_column(abs(mappings[5])) * sgn(mappings[5])    );

   std::string out_name = parser->getOutputBMatrix();
   if(out_name=="")
   {
       out_name =     bmtxtfile.substr(0,bmtxtfile.find(".bmtxt")) + std::string("_rot.bmtxt");
   }


   std::ofstream of(out_name);
   of<<new_Bmatrix;
   of.close();




}
