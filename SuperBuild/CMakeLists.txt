cmake_minimum_required(VERSION 3.20)
project(TORTOISESuperBuild LANGUAGES C CXX)
include(ExternalProject)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "TORTOISE installation" FORCE)
endif()
set(TORTOISE_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads" CACHE PATH "Reusable source archive cache")
set(TORTOISE_DEPENDENCY_PREFIX "${CMAKE_BINARY_DIR}/dependencies" CACHE PATH "Private dependency installation")
set(TORTOISE_BUILD_JOBS 4 CACHE STRING "Maximum jobs per external build")
option(TORTOISE_BUILD_TESTING "Build and run TORTOISE adapter tests" ON)
option(USECUDA "Build CUDA executables (requires installed CUDA)" OFF)
option(USE_VTK "Build VTK tools (requires installed VTK)" OFF)
if(APPLE AND USECUDA)
    message(FATAL_ERROR "CUDA is not supported on macOS; configure with USECUDA=OFF")
endif()
if(NOT TORTOISE_BUILD_JOBS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "TORTOISE_BUILD_JOBS must be a positive integer")
endif()
set(VTK_DIR "" CACHE PATH "Optional existing VTK package")
set(common
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_STANDARD:STRING=17
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY:BOOL=OFF
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY:BOOL=OFF
    -DCMAKE_INSTALL_PREFIX:PATH=${TORTOISE_DEPENDENCY_PREFIX}
    -DCMAKE_PREFIX_PATH:PATH=${TORTOISE_DEPENDENCY_PREFIX}
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DBUILD_TESTING:BOOL=OFF
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5)
foreach(variable CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_OSX_SYSROOT CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_TOOLCHAIN_FILE OpenMP_ROOT)
    if(DEFINED ${variable} AND NOT "${${variable}}" STREQUAL "")
        string(REPLACE ";" "|" forwarded "${${variable}}")
        list(APPEND common "-D${variable}:STRING=${forwarded}")
    endif()
endforeach()
function(dependency name url hash)
    # A supplied cache archive is still checked against the pinned digest.
    if(EXISTS "${TORTOISE_DOWNLOAD_DIR}/${name}.tar.gz")
        set(url "${TORTOISE_DOWNLOAD_DIR}/${name}.tar.gz")
    endif()
    ExternalProject_Add(${name}
        TLS_VERIFY TRUE
        URL "${url}" URL_HASH "SHA256=${hash}"
        DOWNLOAD_DIR "${TORTOISE_DOWNLOAD_DIR}" DOWNLOAD_NAME "${name}.tar.gz"
        LIST_SEPARATOR |
        CMAKE_ARGS ${common} ${ARGN}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${TORTOISE_BUILD_JOBS}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
        LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)
endfunction()

dependency(zlib
    "https://zlib.net/fossils/zlib-1.3.1.tar.gz"
    9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
    -DZLIB_BUILD_EXAMPLES:BOOL=OFF)
dependency(eigen
    "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
    8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    -DEIGEN_BUILD_DOC:BOOL=OFF -DEIGEN_BUILD_TESTING:BOOL=OFF)
dependency(boost
    "https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.gz"
    c62ce6e64d34414864fef946363db91cea89c1b90360eabed0515f0eda74c75c
    "-DBOOST_INCLUDE_LIBRARIES:STRING=filesystem|iostreams|system|algorithm|lexical_cast|dll|endian|math"
    -DBOOST_IOSTREAMS_ENABLE_ZLIB:BOOL=OFF
    -DBOOST_IOSTREAMS_ENABLE_BZIP2:BOOL=OFF
    -DBOOST_IOSTREAMS_ENABLE_LZMA:BOOL=OFF
    -DBOOST_IOSTREAMS_ENABLE_ZSTD:BOOL=OFF)
dependency(fftw
    "https://www.fftw.org/fftw-3.3.10.tar.gz"
    56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    -DENABLE_FLOAT:BOOL=OFF -DENABLE_LONG_DOUBLE:BOOL=OFF -DBUILD_TESTS:BOOL=OFF)
dependency(itk
    "https://github.com/InsightSoftwareConsortium/ITK/releases/download/v6.0b02/InsightToolkit-6.0b02.tar.gz"
    7faf23f4792c438e4295ab2f43fb2c08aad266e0616972dd9c23c88db575f49d
    -DITK_USE_SYSTEM_EIGEN:BOOL=ON -DITK_USE_SYSTEM_ZLIB:BOOL=ON
    -DZLIB_USE_STATIC_LIBS:BOOL=ON
    -DZLIB_LIBRARY_RELEASE:FILEPATH=${TORTOISE_DEPENDENCY_PREFIX}/lib/libz.a -DITK_BUILD_DEFAULT_MODULES:BOOL=ON -DBUILD_EXAMPLES:BOOL=OFF -DITK_WRAP_PYTHON:BOOL=OFF)
add_dependencies(itk eigen zlib)
ExternalProject_Add(tortoise
    DEPENDS itk boost fftw
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../TORTOISEV4"
    BINARY_DIR "${CMAKE_BINARY_DIR}/tortoise-build"
    DOWNLOAD_COMMAND ""
    LIST_SEPARATOR |
    CMAKE_ARGS ${common}
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
        -DUSECUDA:BOOL=${USECUDA} -DUSE_VTK:BOOL=${USE_VTK}
        -DVTK_DIR:PATH=${VTK_DIR}
        -DBUILD_TESTING:BOOL=${TORTOISE_BUILD_TESTING}
        -DZLIB_LIBRARY_RELEASE:FILEPATH=${TORTOISE_DEPENDENCY_PREFIX}/lib/libz.a
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${TORTOISE_BUILD_JOBS}
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
    TEST_COMMAND ${CMAKE_CTEST_COMMAND} --test-dir <BINARY_DIR> --output-on-failure
    TEST_BEFORE_INSTALL ON
    BUILD_ALWAYS ON
    LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)
file(WRITE "${CMAKE_BINARY_DIR}/build-manifest.txt"
    "C compiler: ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_VERSION}\nC++ compiler: ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}\nSystem: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}\nSDK: ${CMAKE_OSX_SYSROOT}\nArchitectures: ${CMAKE_OSX_ARCHITECTURES}\nDeployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}\nBuild type: ${CMAKE_BUILD_TYPE}\nDependencies: Boost 1.86.0, Eigen 3.4.0, FFTW 3.3.10, zlib 1.3.1, ITK 6.0b02\n")
install(FILES "${CMAKE_BINARY_DIR}/build-manifest.txt" DESTINATION .)
