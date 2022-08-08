#ifndef _DIFFPREP_ANONYMIZER_h
#define _DIFFPREP_ANONYMIZER_h

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>


#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include "itkGDCMImageIOOkan.h"


#include "defines.h"



typedef itk::GDCMImageIOOkan       ImageIOType;
typedef itk::Image<unsigned short,3> SliceType;




#endif
