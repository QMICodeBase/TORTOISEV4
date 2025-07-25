#ifndef _TORTOISEUTILITIES_CXX
#define _TORTOISEUTILITIES_CXX


#include "boost/system/error_code.hpp"
#include "boost/filesystem.hpp"
#include "boost/dll/runtime_symbol_info.hpp"

#ifdef _WIN32
    #include <windows.h>
#elif MACOS
    #include <sys/param.h>
    #include <sys/sysctl.h>
#else
    #include <unistd.h>
#endif

int is_big_endian(void)
{
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}


std::string executable_path(const char *argv0)
{
   return boost::dll::program_location().string();

    //boost::filesystem::path program_location();
}

int getNCores() {
#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#elif MACOS
    int nm[2];
    size_t len = 4;
    uint32_t count;

    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);

    if(count < 1) {
        nm[1] = HW_NCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        if(count < 1) { count = 1; }
    }
    return count;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

/*
std::string GetTORTOISEVersion()
{
    return std::string("TORTOISE V4.0.0");
}
*/



/*


#include "boost/system/error_code.hpp"
#include "boost/filesystem.hpp"





std::string executable_path_fallback(const char *argv0)
{
    if (argv0 == NULL || argv0[0] == 0)
    {
        return "";
    }
    boost::system::error_code ec;
    boost::filesystem::path p(
        boost::filesystem::canonical(
            argv0, boost::filesystem::current_path(), ec));
    return p.make_preferred().string();
}


#if (BOOST_OS_CYGWIN || BOOST_OS_WINDOWS) // {

#  include <Windows.h>

std::string executable_path(const char *argv0)
{
	char buf[1024] = {0};
	DWORD ret = GetModuleFileNameA(NULL, buf, sizeof(buf));
	if (ret == 0 || ret == sizeof(buf))
	{
		return executable_path_fallback(argv0);
	}
	return buf;
}
#elif (BOOST_OS_MACOS) // } {

#  include <mach-o/dyld.h>

std::string executable_path(const char *argv0)
{
    char buf[1024] = {0};
    uint32_t size = sizeof(buf);
    int ret = _NSGetExecutablePath(buf, &size);
    if (0 != ret)
    {
        return executable_path_fallback(argv0);
    }
    boost::system::error_code ec;
    boost::filesystem::path p(
        boost::filesystem::canonical(buf, boost::filesystem::current_path(), ec));
    return p.make_preferred().string();
}

#elif (__linux__)
std::string executable_path(const char *argv0)
{
    char buf[1024] = {0};
    int aa=sizeof(buf);
    ssize_t size = readlink("/proc/self/exe", buf, sizeof(buf));
    if (size == 0 || size == sizeof(buf) || size==-1)
    {
        return executable_path_fallback(argv0);
    }
    else
    {
        buf[size] = '\0';
    }
    std::string mpath(buf);
    boost::system::error_code ec;
    boost::filesystem::path p2(mpath);

    boost::filesystem::path p(
        boost::filesystem::canonical(
            p2, boost::filesystem::current_path(), ec));
    return p.make_preferred().string();
}
#endif

*/


#endif
