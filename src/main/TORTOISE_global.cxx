#ifndef _TORTOISEGLOBAL_CXX
#define _TORTOISEGLOBAL_CXX



#include "TORTOISE.h"

TORTOISE::TeeStream* TORTOISE::stream=nullptr;
std::string TORTOISE::executable_folder="";
std::vector<uint> OMPTHREADBASE::Nthreads_per_OMP_thread;
std::atomic_uint OMPTHREADBASE::NAvailableCores={0};

#ifdef USECUDA
std::atomic_bool OMPTHREADBASE::gpu_available={true};
#endif

#endif
