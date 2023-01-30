#ifndef _DEFINES_H
#define _DEFINES_H


#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include <ostream>


#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include "itkDisplacementFieldTransform.h"
namespace fs = boost::filesystem;

#include "../external_src/json_nlohmann/json.hpp"
using json = nlohmann::json;



#define DPI 3.1415926535897931

using CoordType= double;
using PixelDataType = float;

using EigenVecType = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using ImageType2D=itk::Image<PixelDataType,2>;
using ImageType3D=itk::Image<PixelDataType,3>;
using ImageType4D=itk::Image<PixelDataType,4>;
using ImageType3DBool=itk::Image<char,3>;
using ImageType4DBool=itk::Image<char,4>;


using DTType=itk::Vector<PixelDataType,6> ;
using DTImageType=itk::Image<DTType,3>;

using MAPType = itk::VariableLengthVector<float>;
using MAPImageType=itk::VectorImage<PixelDataType,3>;


using DisplacementFieldType = itk::DisplacementFieldTransform<CoordType,3>::DisplacementFieldType;

using  InternalMatrixType=vnl_matrix_fixed< double, 3, 3 >;



template<typename ImageType>
typename ImageType::Pointer readImageD(std::string filename);

template<typename ImageType>
void writeImageD(typename ImageType::Pointer img, std::string filename);



namespace Color {
    enum Code {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };
}







#endif
