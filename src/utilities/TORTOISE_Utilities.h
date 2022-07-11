#ifndef _TORTOISEUTILITIES_H
#define _TORTOISEUTILITIES_H

#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#include "boost/system/error_code.hpp"
#include "boost/filesystem.hpp"

int is_big_endian(void);


int getNCores();

std::string GetTORTOISEVersion();

std::string executable_path_fallback(const char *argv0);

std::string executable_path(const char *argv0);





#endif
