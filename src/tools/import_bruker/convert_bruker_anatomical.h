#ifndef _CONVERT_BRUKER_ANATOMICAL_h
#define _CONVERT_BRUKER_ANATOMICAL_h

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "bruker_anatomical_parser.h"
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>

#include "itkImageFileWriter.hxx"
#include "defines.h"
#include "convert_bruker.h"

namespace fs = boost::filesystem;










class ConvertBrukerAnatomical
{
public:
    ConvertBrukerAnatomical ( int argc , char * argv[] );
    ~ConvertBrukerAnatomical();


private:
    bool convert_bruker_anatomical(std::string input_folder, std::string output_filename);

    RECO_struct read_reco_header(std::string reco_file);
    ACQP_struct read_acqp_header(std::string acqp_file, RECO_struct reco_struct);



    void get_grad_orient(std::string method_file);
    void get_dimension_order(RECO_struct reco_struct,ACQP_struct acqp_struct,int dimension_order[]);

    ImageType4DITK::Pointer get_bruker_image_data(std::string twodseq_file,RECO_struct reco_struct,ACQP_struct acqp_struct);




private:
    BrukerAnatomical_PARSER* parser;
    std::string slice_orientation;
    std::string read_orientation;    
    vnl_matrix_fixed<double,3,3> grad_orient;
    int NTE;
    double RG;
};





#endif
