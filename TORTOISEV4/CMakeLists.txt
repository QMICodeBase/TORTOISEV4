cmake_minimum_required(VERSION 3.20)

PROJECT(TORTOISEV4  )

SET(USEOPENMP 1)
#   SET(CMAKE_BUILD_TYPE Debug)
#   SET(ISDEBUG 1)

if ( NOT DEFINED USECUDA)
    SET(USECUDA  0)
endif()
if ( NOT DEFINED USE_VTK)
    SET(USE_VTK 0)
endif()
if ( NOT DEFINED USEWEBGPU)
    SET(USEWEBGPU 0)
endif()
if ( NOT DEFINED USEMETAL)
    SET(USEMETAL 0)
endif()
if ( USECUDA AND USEWEBGPU )
    message(FATAL_ERROR "USECUDA and USEWEBGPU are mutually exclusive; configure one backend per build tree")
endif()
if ( USEMETAL AND (USECUDA OR USEWEBGPU) )
    message(FATAL_ERROR "USEMETAL is mutually exclusive with USECUDA and USEWEBGPU; configure one backend per build tree")
endif()
if ( USEMETAL AND NOT APPLE )
    message(FATAL_ERROR "USEMETAL requires macOS")
endif()

# Validation-only. Routes every DWI volume through the GPU registration path
# instead of splitting it with the (non-reproducible) ITK CPU path, making a GPU
# run deterministic so another backend can be diffed against it meaningfully.
# Costs registration performance. NOTE it DOES change computed values: its
# OmpThreadBase.h half fixes ITK's work-unit count, which alters metric values (that
# is what makes them reproducible). See DIFFPREP.cxx and
# benchmark/scripts/demo_diffprep_nondeterminism.sh. Never enable for production.
if ( NOT DEFINED DETERMINISTIC_GPU)
    SET(DETERMINISTIC_GPU 0)
endif()
if ( DETERMINISTIC_GPU )
    add_definitions(-DTORTOISE_DETERMINISTIC_GPU)
    message(WARNING "DETERMINISTIC_GPU=1: all volumes routed to the GPU registration path. "
                    "Reproducible but slower; validation builds only.")
endif()

# Variant builds MUST NOT collide with the reference binaries. All build trees share
# one bin/ (CMAKE_RUNTIME_OUTPUT_DIRECTORY below), and until this was added,
# build_cuda_nofma and build_cuda_det both linked to bin/TORTOISEProcess_cuda -
# and build_cuda_nofma also relinked bin/gpu_replay, the CUDA self-replay
# bit-exactness anchor. A single `cmake --build build_cuda_det` would silently
# replace the CUDA *reference* with a validation build, and every downstream gate
# would then be measuring the wrong binary while reporting success.
set(TORTOISE_BIN_SUFFIX "")
if ( DETERMINISTIC_GPU )
    set(TORTOISE_BIN_SUFFIX "${TORTOISE_BIN_SUFFIX}_det")
endif()
if ( CMAKE_CUDA_FLAGS MATCHES "fmad=false" )
    set(TORTOISE_BIN_SUFFIX "${TORTOISE_BIN_SUFFIX}_nofma")
endif()
if ( NOT TORTOISE_BIN_SUFFIX STREQUAL "" )
    message(STATUS "Variant build: executables get the suffix '${TORTOISE_BIN_SUFFIX}'")
endif()

if ( NOT DEFINED QSIPREP)
    SET(QSIPREP 0)
else()
    message("Doing smaller build for QSIPrep")
endif()

if ( DEFINED CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        SET(ISDEBUG 0)
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            SET(ISDEBUG 1)
        else()
            SET(CMAKE_BUILD_TYPE Release)
            SET(ISDEBUG 0)
        endif()
    endif()
else()
    SET(CMAKE_BUILD_TYPE Release)
    SET(ISDEBUG 0)
endif()


message( "${CMAKE_CXX_COMPILER}")
message( "${CMAKE_C_COMPILER}")
message( "${CMAKE_BUILD_TYPE}")


message(${CMAKE_BINARY_DIR})
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/lib/ )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/lib/ )


# Vectorisation floor: x86-64-v3 = Haswell (2013) / Excavator, giving AVX2 + FMA.
# Without it the build targets baseline x86-64, i.e. SSE2 only, so -O3's autovectoriser
# is capped at 128-bit vectors and NO FMA instruction is emitted anywhere on the host.
# Deliberately NOT -march=native: that would bake in this Zen 4's AVX-512 and SIGILL on
# any older machine, and TORTOISE ships binaries. -DTORTOISE_ARCH=none for a generic build.
#
# NOTE: this changes host floating-point results (FMA contraction, wider reduction
# vectors). It does not reach the CUDA kernels - CUDA_FLAGS are separate and there is no
# -Xcompiler forwarding - but src/cuda_src/*.cxx host wrappers ARE compiled with these
# flags, so `gpu_replay --all` must be re-checked after any change here.
IF(NOT DEFINED TORTOISE_ARCH)
    # arm64 has no x86-64-v3. NOTE this means macOS host floating-point results
    # differ from the Linux reference before any GPU kernel runs - record it in the run artefact.
    IF(APPLE)
        SET(TORTOISE_ARCH "none")
    ELSE()
        SET(TORTOISE_ARCH "x86-64-v3")
    ENDIF()
ENDIF()
IF(TORTOISE_ARCH STREQUAL "none")
    SET(TORTOISE_ARCH_FLAG "")
ELSE()
    SET(TORTOISE_ARCH_FLAG "-march=${TORTOISE_ARCH}")
ENDIF()
message(STATUS "TORTOISE host arch flag: ${TORTOISE_ARCH_FLAG}")

IF(APPLE AND USEOPENMP)
    # Apple clang has no bundled OpenMP runtime; Homebrew libomp supplies it.
    # std::bind2nd was deprecated in C++11 and REMOVED in C++17, but ITK 5.4+ requires
    # C++17 on Apple LLVM, so the two constraints collide. libc++ keeps the removed
    # binders behind this macro. Three upstream call sites use it
    # (DTIModel.cxx:2031,2841 and estimate_tensor_wlls.cxx:274) and are left untouched -
    # rewriting upstream source to a lambda would be a change this port has no remit for.
    # ...and the WebGPU targets build at C++20 (Dawn's headers require it), where
    # std::plus's first_argument_type/result_type typedefs bind2nd needs were removed
    # as well. The Metal targets are C++17, where they are merely deprecated.
    add_definitions(-D_LIBCPP_ENABLE_CXX17_REMOVED_BINDERS
                    -D_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
    # Size: -ffunction-sections/-fdata-sections put each function and object in its own
    # section so ld64's -dead_strip can drop them individually rather than per .o.
    # NOTE this only affects OUR translation units - ITK's 111 static archives (68 MB)
    # were built without them, so the linker's granularity there stays per-object.
    add_compile_options(-ffunction-sections -fdata-sections)
    add_link_options(-Wl,-dead_strip)

    # The vendored betokan archive is Linux x86-64 with no source in the tree, so it
    # cannot be linked on arm64. create_mask.cxx's __APPLE__ branch shells out to the
    # user's FSL bet2 instead (CLAUDE.md 6.4), so there is nothing to link here.
    set(BETOKAN_LIB "")
    set(EXCLUDE_ZLIB_FLAG "")   # -Wl,--exclude-libs is GNU ld only; ld64 rejects it
    # GCC-only; Apple clang rejects them outright.
    set(STATIC_CXX_LIBS "")

    # FFTW is a Homebrew keg, not on the default search path.
    if(DEFINED FFTW_ROOT)
        include_directories(${FFTW_ROOT}/include)
        link_directories(${FFTW_ROOT}/lib)
    endif()
    SET(OMP_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    # ITK 5.4's itkMacro.h hard-errors below C++17 on Apple LLVM. Measured: the tree's
    # headers compile clean at 17 against Homebrew ITK 5.4.5, so no ITK 6 source build.
    SET(CMAKE_CXX_FLAGS "${TORTOISE_ARCH_FLAG} ${OMP_FLAGS} -std=c++17 -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    SET(CMAKE_C_FLAGS   "${TORTOISE_ARCH_FLAG} ${OMP_FLAGS} -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    link_directories(/opt/homebrew/opt/libomp/lib)
    link_libraries(omp)
    add_definitions(-DUSE_OPENMP)
ELSEIF(USEOPENMP)
    SET(CMAKE_CXX_FLAGS "${TORTOISE_ARCH_FLAG} -fopenmp -std=c++14 -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    SET(CMAKE_C_FLAGS "${TORTOISE_ARCH_FLAG} -fopenmp -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    add_definitions(-DUSE_OPENMP)
ELSE(USEOPENMP)
    SET(CMAKE_CXX_FLAGS "${TORTOISE_ARCH_FLAG} -fPIC -std=c++14 -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    SET(CMAKE_C_FLAGS "${TORTOISE_ARCH_FLAG} -fPIC -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
ENDIF(USEOPENMP)



set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    ON)
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_NO_BOOST_CMAKE ON)
set(BOOST_ROOT /usr/local/boost186)
set(Boost_LIBRARY_DIRS /usr/local/boost186/lib)
set(Boost_INCLUDE_DIRS /usr/local/boost186/include)
# Boost.System has been header-only since 1.69 and Homebrew's 1.90 ships no stub
# library, so requesting it as a component fails on macOS. It contributes no symbols.
if(APPLE)
    find_package(Boost COMPONENTS iostreams filesystem REQUIRED)
else()
    find_package(Boost COMPONENTS iostreams filesystem system REQUIRED)
endif()

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  MESSAGE("Boost Found")
else(Boost_FOUND)
   MESSAGE("BOOST NOT FOUND!")
endif()


message(${Boost_LIBRARIES})

IF(USECUDA)
    include(CheckLanguage)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda")
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/include/")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    include_directories("/usr/local/cuda/targets/x86_64-linux/include/")

    find_package(CUDAToolkit)

    check_language(CUDA)
    message(${CMAKE_CUDA_COMPILER})
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      add_definitions(-DUSECUDA)
      add_definitions(-DUSEGPU)          # see the USEWEBGPU branch for why
      message("CUDA Enabled..")
    else()
      message(STATUS "No CUDA SUPPORT..")
    endif()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11") # Or higher standard

ELSE(USECUDA)
    set(ZLIB_USE_STATIC_LIBS ON)
ENDIF(USECUDA)



IF(ISDEBUG)
     if ( NOT DEFINED ITK_DIR)         
         SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-6.0b01-debug)      
    endif()
    FIND_PACKAGE(ITK)
    IF(ITK_FOUND)
        INCLUDE(${ITK_USE_FILE})
        MESSAGE("ITK ${ITK_VERSION_MAJOR}${ITK_VERSION_MINOR}${ITK_VERSION_PATCH} Debug version used!")
    ELSE(ITK_FOUND)
        MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
    ENDIF(ITK_FOUND)
ELSE(ISDEBUG)
    if ( NOT DEFINED ITK_DIR)
        SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-6.0b01-release)        
       #SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-6.0b01-QMIL09_release)        
    endif()
    FIND_PACKAGE(ITK)
    IF(ITK_FOUND)
       INCLUDE(${ITK_USE_FILE})
       MESSAGE("ITK ${ITK_VERSION_MAJOR}${ITK_VERSION_MINOR}${ITK_VERSION_PATCH} Release version used!")
       MESSAGE("${ITK_DIR}")
    ELSE(ITK_FOUND)
        MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
    ENDIF(ITK_FOUND)
ENDIF(ISDEBUG)



find_package (Eigen3 3.4 REQUIRED NO_MODULE)
if(Eigen3_FOUND)    
    message("Eigen found.")
    message(${EIGEN3_INCLUDE_DIR})    
    include_directories(${EIGEN3_INCLUDE_DIR})
ELSE(Eigen3_FOUND)
    message("NO Eigen.")
endif()


set(SOURCES ../src/main/defines.cxx  ../src/main/TORTOISE_global.cxx ../src/main/antsCommandLineParser.cxx ../src/main/antsCommandLineOption.cxx  ../src/utilities/TORTOISE_Utilities.cxx ../src/utilities/extract_3Dvolume_from_4D.cxx  ../src/utilities/read_3Dvolume_from_4D.cxx ../src/utilities/read_bmatrix_file.cxx ../src/utilities/math_utilities.cxx ../src/external_src/cmpfit-1.3a/mpfit.c)
include_directories(../src/main)

if(NOT DEFINED EXCLUDE_ZLIB_FLAG)
    set(EXCLUDE_ZLIB_FLAG -Wl,--exclude-libs,libitkzlib-5.4.a)
endif()
if(NOT DEFINED STATIC_CXX_LIBS)
    set(STATIC_CXX_LIBS -static-libgcc -static-libstdc++)
endif()
if(NOT DEFINED BETOKAN_LIB)
    set(BETOKAN_LIB ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a)
endif()

IF(USEMETAL)
    set(METAL_DEFS USEMETAL USEGPU)
    # metal_src holds one header per interface and must shadow cuda_src, whose
    # directory is still needed for the backend-neutral sources living there.
    set(METAL_INCS ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu_src
                   ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda_src)
    find_library(METAL_FW Metal)
    find_library(FOUNDATION_FW Foundation)
    set(METAL_FRAMEWORKS ${METAL_FW} ${FOUNDATION_FW})
    message("Metal backend enabled")

    # .metal shaders -> generated headers holding raw string literals, by the same
    # script the WGSL sources use.
    file(GLOB MSL_SOURCES CONFIGURE_DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/../src/metal_src/shaders/*.metal)
    # metric_common.metal is a prelude prepended to the metric shaders, not a
    # standalone module - and it must stay in every DEPENDS or editing it leaves the
    # generated headers stale while the build reports success (CLAUDE.md 6.3).
    set(MSL_PRELUDE ${CMAKE_CURRENT_SOURCE_DIR}/../src/metal_src/shaders/metric_common.metal)
    list(REMOVE_ITEM MSL_SOURCES ${MSL_PRELUDE})
    set(MSL_HEADERS)
    foreach(_msl ${MSL_SOURCES})
        get_filename_component(_stem ${_msl} NAME_WE)
        set(_hdr ${CMAKE_CURRENT_BINARY_DIR}/gen/msl/${_stem}.metal.h)
        add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${CMAKE_COMMAND} -DIN=${_msl} -DOUT=${_hdr} -DVAR=k${_stem}MSL
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake
            DEPENDS ${_msl} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake ${MSL_PRELUDE}
            COMMENT "Embedding ${_stem}.metal")
        list(APPEND MSL_HEADERS ${_hdr})
    endforeach()
    add_custom_target(msl_headers DEPENDS ${MSL_HEADERS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/gen/msl)

    # Sources of the Metal backend as it is ported, milestone by milestone.
    SET(METAL_SOURCES
        ../src/metal_src/metal_context.mm
        ../src/metal_src/gpu_image.mm
        ../src/metal_src/image_utilities.mm
        ../src/metal_src/reductions.mm
        ../src/metal_src/compose_fields.mm
        ../src/metal_src/invert_field.mm
        ../src/metal_src/compute_entropy.mm
        ../src/metal_src/resample_image.mm
        ../src/metal_src/warp_image.mm
        ../src/metal_src/quadratic_transform_image.mm
        ../src/metal_src/gaussian_smooth_image.mm
        ../src/metal_src/compute_metric.mm)

    # M1 smoke test: device selection, allocation at validation scale, a kernel.
    add_executable(metal_probe ../src/metal_src/metal_probe_main.mm
                               ../src/metal_src/metal_context.mm)
    add_dependencies(metal_probe msl_headers)
    target_link_libraries(metal_probe ${METAL_FRAMEWORKS})
    target_compile_definitions(metal_probe PRIVATE ${METAL_DEFS})
    target_include_directories(metal_probe BEFORE PRIVATE ${METAL_INCS})
    set_target_properties(metal_probe PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    # M6: the full pipeline on Metal. Same main/ sources as the CUDA and WebGPU
    # builds, with the Metal backend swapped in behind the gpu_src shims.
    # bet comes from the user's FSL install at runtime (see create_mask.cxx), so
    # unlike the Linux targets there is no libbetokan to link.
    add_executable(TORTOISEProcess_metal ../src/main/main.cxx ../src/main/register_dwi_to_b0.cxx
        ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx
        ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx
        ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx ../src/tools/RotateBMatrix/rotate_bmatrix.cxx
        ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx
        ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx ../src/main/EPIREG.cxx
        ../src/main/TORTOISE_parser.cxx ../src/main/run_drbuddi_stage.cxx
        ../src/main/DRBUDDI_Diffeo.cxx ../src/main/drbuddi_image_utilities.cxx
        ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx
        ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx
        ../src/main/create_mask.cxx ../src/cuda_src/register_dwi_to_b0_cuda.cxx
        ${METAL_SOURCES} ${SOURCES})
    add_dependencies(TORTOISEProcess_metal msl_headers)
    target_link_libraries(TORTOISEProcess_metal
        ${ITK_LIBRARIES} Eigen3::Eigen ${Boost_LIBRARIES} ${METAL_FRAMEWORKS}
        -lz fftw3)
    target_compile_definitions(TORTOISEProcess_metal PRIVATE ${METAL_DEFS})
    target_include_directories(TORTOISEProcess_metal BEFORE PRIVATE ${METAL_INCS})
    set_target_properties(TORTOISEProcess_metal PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    add_executable(DRBUDDI_metal ../src/tools/DRBUDDI/DRBUDDI_main.cxx
        ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx
        ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx ../src/main/run_drbuddi_stage.cxx
        ../src/main/DRBUDDI_Diffeo.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx
        ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx
        ../src/main/create_mask.cxx ../src/main/drbuddi_image_utilities.cxx
        ${METAL_SOURCES} ${SOURCES})
    add_dependencies(DRBUDDI_metal msl_headers)
    target_link_libraries(DRBUDDI_metal
        ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${METAL_FRAMEWORKS} -lz fftw3)
    target_compile_definitions(DRBUDDI_metal PUBLIC -DDRBUDDIALONE)
    target_compile_definitions(DRBUDDI_metal PRIVATE ${METAL_DEFS})
    target_include_directories(DRBUDDI_metal BEFORE PRIVATE ${METAL_INCS})
    set_target_properties(DRBUDDI_metal PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    # M2+: the same replay source as webgpu_replay, bound to the Metal backend by
    # src/gpu_src/gpu_backend.h. One harness, two executables.
    add_executable(metal_replay ../src/tools/GPUReplay/webgpu_replay_main.cxx ${METAL_SOURCES})
    add_dependencies(metal_replay msl_headers)
    target_link_libraries(metal_replay ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${METAL_FRAMEWORKS})
    target_compile_definitions(metal_replay PRIVATE ${METAL_DEFS})
    target_include_directories(metal_replay BEFORE PRIVATE ${METAL_INCS})
    set_target_properties(metal_replay PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
ENDIF(USEMETAL)

IF(USEWEBGPU)
    # Dawn (WebGPU). Built separately by tortoise_libraries/build_dawn.sh, which
    # pins the revision in dawn.pin. Vulkan backend only - the GL backends are
    # compiled out so no build-time path to the integrated GPU exists.
    # DAWN_DIR must be supplied by the builder: -DDAWN_DIR=/path/to/dawn, or the
    # DAWN_DIR environment variable. There is deliberately no default - a default
    # pointing at one developer's home directory is not portable and hides the
    # requirement until the link step.
    if(NOT DEFINED DAWN_DIR AND DEFINED ENV{DAWN_DIR})
        set(DAWN_DIR $ENV{DAWN_DIR})
    endif()
    if(NOT DEFINED DAWN_DIR)
        message(FATAL_ERROR
            "USEWEBGPU=1 requires DAWN_DIR. Pass -DDAWN_DIR=<path to the dawn checkout> "
            "(the directory containing build/src/dawn/native/libwebgpu_dawn.a), or set "
            "the DAWN_DIR environment variable. See tortoise_libraries/build_dawn.sh.")
    endif()
    set(DAWN_LIB ${DAWN_DIR}/build/src/dawn/native/libwebgpu_dawn.a)
    if(NOT EXISTS ${DAWN_LIB})
        message(FATAL_ERROR "Dawn not built: ${DAWN_LIB} missing. Run tortoise_libraries/build_dawn.sh")
    endif()
    # USEWEBGPU/USEGPU and the shim include path are applied PER TARGET, not
    # directory-wide: the ELSE(USECUDA) branch below still defines the plain CPU
    # targets, and giving those -DUSEGPU would compile GPU code paths into
    # binaries that link neither the WebGPU sources nor Dawn.
    # USEGPU is set on the command line rather than via defines.h because several
    # files use #ifdef USEGPU without including it; an invisible macro would
    # silently compile the GPU branch out and fall back to CPU.
    # Apple ld64 has neither --start-group nor libdl; CMAKE_DL_LIBS is empty there.
    if(APPLE)
        set(GRP_START "")
        set(GRP_END "")
        find_library(METAL_FW Metal)
        find_library(QUARTZCORE_FW QuartzCore)
        find_library(IOSURFACE_FW IOSurface)
        find_library(FOUNDATION_FW Foundation)
        find_library(IOKIT_FW IOKit)
        find_library(COREFOUNDATION_FW CoreFoundation)
        set(DAWN_PLATFORM_LIBS ${METAL_FW} ${QUARTZCORE_FW} ${IOSURFACE_FW}
                               ${FOUNDATION_FW} ${IOKIT_FW} ${COREFOUNDATION_FW})
    else()
        set(GRP_START "-Wl,--start-group")
        set(GRP_END "-Wl,--end-group")
        set(DAWN_PLATFORM_LIBS "")
    endif()
    set(WEBGPU_DEFS USEWEBGPU USEGPU)
    # gpu_src holds one shim per backend interface and must shadow cuda_src, whose
    # directory is still needed for the backend-neutral sources that live there.
    set(WEBGPU_INCS ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu_src
                    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda_src)
    include_directories(${DAWN_DIR}/include ${DAWN_DIR}/build/gen/include)
    message("WebGPU (Dawn) enabled: ${DAWN_LIB}")

    # WGSL shaders -> generated headers holding raw string literals.
    # CONFIGURE_DEPENDS so adding a shader to an already-configured tree creates its
    # generation rule on the next build, rather than being silently omitted.
    file(GLOB WGSL_SOURCES CONFIGURE_DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/../src/webgpu_src/shaders/*.wgsl)
    # metric_common.wgsl is a prelude prepended to the metric shaders, not a
    # standalone module.
    list(REMOVE_ITEM WGSL_SOURCES
         ${CMAKE_CURRENT_SOURCE_DIR}/../src/webgpu_src/shaders/metric_common.wgsl)
    set(WGSL_HEADERS)
    foreach(_wgsl ${WGSL_SOURCES})
        get_filename_component(_stem ${_wgsl} NAME_WE)
        set(_hdr ${CMAKE_CURRENT_BINARY_DIR}/gen/wgsl/${_stem}.wgsl.h)
        add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${CMAKE_COMMAND} -DIN=${_wgsl} -DOUT=${_hdr} -DVAR=k${_stem}WGSL
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake
            DEPENDS ${_wgsl} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake
                    ${CMAKE_CURRENT_SOURCE_DIR}/../src/webgpu_src/shaders/metric_common.wgsl
            COMMENT "Embedding ${_stem}.wgsl")
        list(APPEND WGSL_HEADERS ${_hdr})
    endforeach()
    add_custom_target(wgsl_headers DEPENDS ${WGSL_HEADERS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/gen/wgsl)

    # Sources of the WebGPU backend as it is ported, milestone by milestone.
    SET(WEBGPU_SOURCES
        ../src/webgpu_src/webgpu_context.cxx
        ../src/webgpu_src/gpu_image.cxx
        ../src/webgpu_src/resample_image.cxx
        ../src/webgpu_src/warp_image.cxx
        ../src/webgpu_src/quadratic_transform_image.cxx
        ../src/webgpu_src/gaussian_smooth_image.cxx
        ../src/webgpu_src/image_utilities.cxx
        ../src/webgpu_src/reductions.cxx
        ../src/webgpu_src/compose_fields.cxx
        ../src/webgpu_src/invert_field.cxx
        ../src/webgpu_src/compute_entropy.cxx
        ../src/webgpu_src/compute_metric.cxx)

    # M2 smoke test: adapter selection, allocation at validation scale, a kernel.
    add_executable(webgpu_probe ../src/webgpu_src/webgpu_probe_main.cxx
                                ../src/webgpu_src/webgpu_context.cxx)
    add_dependencies(webgpu_probe wgsl_headers)
    target_link_libraries(webgpu_probe ${DAWN_LIB} ${DAWN_PLATFORM_LIBS} ${CMAKE_DL_LIBS} pthread)
    # Dawn's C++ headers require C++20; the rest of the project builds at 14/17.
    target_compile_definitions(webgpu_probe PRIVATE ${WEBGPU_DEFS})
    target_include_directories(webgpu_probe BEFORE PRIVATE ${WEBGPU_INCS})
    set_target_properties(webgpu_probe PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

    # The full pipeline on WebGPU: same main/ sources as the CUDA build, with the
    # WebGPU backend swapped in behind the gpu_src shims.
    add_executable(TORTOISEProcess_webgpu ../src/main/main.cxx ../src/main/register_dwi_to_b0.cxx
        ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx
        ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx
        ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx ../src/tools/RotateBMatrix/rotate_bmatrix.cxx
        ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx
        ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx ../src/main/EPIREG.cxx
        ../src/main/TORTOISE_parser.cxx ../src/main/run_drbuddi_stage.cxx
        ../src/main/DRBUDDI_Diffeo.cxx ../src/main/drbuddi_image_utilities.cxx
        ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx
        ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx
        ../src/main/create_mask.cxx ../src/cuda_src/register_dwi_to_b0_cuda.cxx
        ${WEBGPU_SOURCES} ${SOURCES})
    add_dependencies(TORTOISEProcess_webgpu wgsl_headers)
    target_link_libraries(TORTOISEProcess_webgpu
        ${GRP_START} ${ITK_LIBRARIES} ${GRP_END}
        Eigen3::Eigen ${Boost_LIBRARIES} ${DAWN_LIB}
        ${BETOKAN_LIB}
        -lz fftw3 ${DAWN_PLATFORM_LIBS} ${CMAKE_DL_LIBS} pthread)
    target_compile_definitions(TORTOISEProcess_webgpu PRIVATE ${WEBGPU_DEFS})
    target_include_directories(TORTOISEProcess_webgpu BEFORE PRIVATE ${WEBGPU_INCS})
    set_target_properties(TORTOISEProcess_webgpu PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

    # DRBUDDI standalone on WebGPU, mirroring the DRBUDDI_cuda target.
    add_executable(DRBUDDI_webgpu ../src/tools/DRBUDDI/DRBUDDI_main.cxx
        ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx
        ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx ../src/main/run_drbuddi_stage.cxx
        ../src/main/DRBUDDI_Diffeo.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx
        ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx
        ../src/main/create_mask.cxx ../src/main/drbuddi_image_utilities.cxx
        ${WEBGPU_SOURCES} ${SOURCES})
    add_dependencies(DRBUDDI_webgpu wgsl_headers)
    target_link_libraries(DRBUDDI_webgpu
        ${GRP_START} ${ITK_LIBRARIES} ${GRP_END}
        Eigen3::Eigen ${Boost_LIBRARIES} ${DAWN_LIB}
        ${BETOKAN_LIB}
        -lz fftw3 ${DAWN_PLATFORM_LIBS} ${CMAKE_DL_LIBS} pthread)
    target_compile_definitions(DRBUDDI_webgpu PUBLIC -DDRBUDDIALONE)
    target_compile_definitions(DRBUDDI_webgpu PRIVATE ${WEBGPU_DEFS})
    target_include_directories(DRBUDDI_webgpu BEFORE PRIVATE ${WEBGPU_INCS})
    set_target_properties(DRBUDDI_webgpu PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

    # M3+: replay the captured CUDA golden vectors against the WebGPU kernels.
    add_executable(webgpu_replay ../src/tools/GPUReplay/webgpu_replay_main.cxx ${WEBGPU_SOURCES})
    add_dependencies(webgpu_replay wgsl_headers)
    # ITK's IO factory registration pulls symbols across several static libs, so
    # let the linker resolve them as a group rather than in one pass.
    target_link_libraries(webgpu_replay
        ${GRP_START} ${ITK_LIBRARIES} ${GRP_END}
        ${Boost_LIBRARIES} ${DAWN_LIB} ${DAWN_PLATFORM_LIBS} ${CMAKE_DL_LIBS} pthread)
    target_compile_definitions(webgpu_replay PRIVATE ${WEBGPU_DEFS})
    target_include_directories(webgpu_replay BEFORE PRIVATE ${WEBGPU_INCS})
    set_target_properties(webgpu_replay PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
ENDIF(USEWEBGPU)



IF(USECUDA)
     include_directories(../src/cuda_src )
     add_executable(TORTOISEProcess_cuda ../src/main/main.cxx ../src/main/register_dwi_to_b0.cxx    ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx  ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx  ../src/tools/RotateBMatrix/rotate_bmatrix.cxx  ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx  ../src/main/EPIREG.cxx ../src/main/TORTOISE_parser.cxx ../src/cuda_src/cuda_image.cxx  ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/main/run_drbuddi_stage.cxx  ../src/main/run_drbuddi_stage_TVVF.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx  ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/gpu_capture.cxx ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/main/DRBUDDI_Diffeo.cxx  ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx ../src/cuda_src/register_dwi_to_b0_cuda.cxx ../src/cuda_src/quadratic_transform_image.cxx ../src/cuda_src/quadratic_transform_image.cu ../src/cuda_src/compute_entropy.cxx ../src/cuda_src/compute_entropy.cu  ${SOURCES} )
     target_include_directories(TORTOISEProcess_cuda   PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(TORTOISEProcess_cuda  Eigen3::Eigen ${ITK_LIBRARIES} ${Boost_LIBRARIES}   fftw3 ${BETOKAN_LIB}   ${STATIC_CXX_LIBS} )

     add_executable(gpu_replay ../src/tools/GPUReplay/gpu_replay_main.cxx ../src/cuda_src/quadratic_transform_image.cu ../src/cuda_src/quadratic_transform_image.cxx ../src/cuda_src/cuda_image.cxx ../src/cuda_src/gpu_capture.cxx ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/cuda_src/compute_entropy.cu ../src/cuda_src/compute_entropy.cxx ${SOURCES})
     target_include_directories(gpu_replay PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(gpu_replay Eigen3::Eigen ${ITK_LIBRARIES} ${Boost_LIBRARIES} fftw3 ${STATIC_CXX_LIBS})

     add_executable(DRBUDDI_cuda ../src/tools/DRBUDDI/DRBUDDI_main.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx   ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx   ../src/cuda_src/cuda_image.cxx  ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/main/run_drbuddi_stage.cxx ../src/main/run_drbuddi_stage_TVVF.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx  ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/gpu_capture.cxx ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/main/DRBUDDI_Diffeo.cxx   ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx    ${SOURCES} )
     target_include_directories(DRBUDDI_cuda  PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(DRBUDDI_cuda  ${ITK_LIBRARIES} ${Boost_LIBRARIES}    ${BETOKAN_LIB} -lz    ${STATIC_CXX_LIBS})
     target_compile_definitions(DRBUDDI_cuda PUBLIC -DDRBUDDIALONE)

     add_executable(DRTAMAS_cuda  ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/drbuddi_image_utilities.cxx ../src/tools/DRTAMAS/compute_metric_dev.cu ../src/tools/DRTAMAS/compute_metric_dev.cxx   ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx ../src/tools/DRTAMAS/run_drtamas_stage.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities.cu   ../src/tools/DRTAMAS/DRTAMAS_Diffeo.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/gpu_capture.cxx ../src/cuda_src/cuda_image.cxx ../src/main/rigid_register_images.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/DRTAMAS/DRTAMAS_main.cxx ../src/tools/DRTAMAS/DRTAMAS.cxx ../src/tools/DRTAMAS/DRTAMAS_parser.cxx   ${SOURCES} )
     target_include_directories(DRTAMAS_cuda   PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(DRTAMAS_cuda  Eigen3::Eigen  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  ${STATIC_CXX_LIBS} )
     set_target_properties(DRTAMAS_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")


     add_custom_command(
         TARGET DRTAMAS_cuda POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 "../src/tools/DRTAMAS/DRTAMAS_batch_register_to_target_main.bash"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS/DRTAMAS_batch_register_to_target_main.bash"
     )
    add_custom_command(
        TARGET DRTAMAS_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
             "../src/tools/DRTAMAS/DRTAMAS_batch_register_to_target_nonsmoothlast.bash"
             "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS/DRTAMAS_batch_register_to_target_nonsmoothlast.bash"
    )
    add_custom_command(
        TARGET DRTAMAS_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
             "../src/tools/DRTAMAS/DRTAMAS_create_template_main.bash"
             "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS/DRTAMAS_create_template_main.bash"
    )
    add_custom_command(
        TARGET DRTAMAS_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
             "../src/tools/DRTAMAS/DRTAMAS_create_template_nonsmoothlast.bash"
             "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS/DRTAMAS_create_template_nonsmoothlast.bash"
    )
    add_custom_command(
        TARGET DRTAMAS_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
             "../src/tools/DRTAMAS/DRTAMAS_external_registration_with_constraint.bash"
             "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS/DRTAMAS_external_registration_with_constraint.bash"
    )


ELSE(USECUDA)
    # WebGPU cannot be fully static: Dawn dlopen()s the Vulkan loader at runtime,
    # and a -static binary segfaults on the first dlopen.
    if(NOT QSIPREP AND NOT USEWEBGPU AND NOT APPLE)
        SET(CMAKE_EXE_LINKER_FLAGS "-static ")
    endif()
    # Only the pure-CPU tree may build these. build_cpu and build_webgpu are both
    # USECUDA=0, so both used to link TORTOISEProcess and DRBUDDI into the SAME
    # shared bin/ paths - and revalidate.sh builds build_webgpu last, so the WebGPU
    # tree silently won. `file bin/TORTOISEProcess` showed a dynamically linked
    # binary, i.e. the WebGPU one, while finalize_run.sh attributed CPU provenance
    # to build_cpu. The WebGPU tree builds TORTOISEProcess_webgpu / DRBUDDI_webgpu
    # for its own use, so the plain duplicates were unused as well as harmful.
    if(NOT USEWEBGPU AND NOT USEMETAL)
    add_executable(TORTOISEProcess ../src/main/main.cxx  ../src/main/register_dwi_to_b0.cxx  ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx  ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx   ../src/tools/RotateBMatrix/rotate_bmatrix.cxx ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx ../src/main/EPIREG.cxx ../src/main/DRBUDDI_Diffeo.cxx ../src/main/run_drbuddi_stage.cxx ../src/main/drbuddi_image_utilities.cxx   ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/main/TORTOISE_parser.cxx  ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ${SOURCES})
    target_link_libraries(TORTOISEProcess PRIVATE Eigen3::Eigen ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${BETOKAN_LIB} -lz fftw3 ${STATIC_CXX_LIBS})

    add_executable(DRBUDDI ../src/tools/DRBUDDI/DRBUDDI_main.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx   ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx   ../src/main/run_drbuddi_stage.cxx ../src/main/DRBUDDI_Diffeo.cxx   ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx  ../src/main/drbuddi_image_utilities.cxx  ${SOURCES} )
    target_link_libraries(DRBUDDI  PRIVATE ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${BETOKAN_LIB} -lz fftw3 ${STATIC_CXX_LIBS})
    target_link_libraries(DRBUDDI PRIVATE ${EXCLUDE_ZLIB_FLAG})


     add_executable(Vol2SlRegistration   ../src/tools/Vol2SlRegistration/vol2sl_registration_main.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ${SOURCES})
     target_include_directories(Vol2SlRegistration   PRIVATE   )
     target_link_libraries(Vol2SlRegistration ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${STATIC_CXX_LIBS})
    endif()


ENDIF(USECUDA)

add_executable(CreateMask ../src/tools/CreateMask/create_mask_main.cxx ../src/main/create_mask.cxx ../src/main/defines.cxx)
target_link_libraries(CreateMask PRIVATE ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${BETOKAN_LIB} -lz )


add_executable(EstimateTensor   ../src/tools/EstimateTensor/estimate_tensor_wlls_main.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls_parser.cxx  ${SOURCES})
target_link_libraries(EstimateTensor ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${LIBS})
target_compile_definitions(EstimateTensor PUBLIC -DNOTORTOISE)


add_executable(EstimateMAPMRI   ../src/tools/EstimateMAPMRI/estimate_mapmri_main.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/tools/EstimateMAPMRI/estimate_mapmri_parser.cxx ${SOURCES})
target_link_libraries(EstimateMAPMRI ${ITK_LIBRARIES} ${Boost_LIBRARIES} )
target_compile_definitions(EstimateMAPMRI PUBLIC -DNOTORTOISE)

add_executable(RotateBmatrix   ../src/tools/RotateBMatrix/rotate_bmatrix_main.cxx  ../src/tools/RotateBMatrix/rotate_bmatrix.cxx ${SOURCES})
target_link_libraries(RotateBmatrix ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(TORTOISEBmatrixToFSLBVecs   ../src/tools/TORTOISEBmatrixToFSLBVecs/tortoise_bmatrix_to_fsl_bvecs.cxx  ../src/utilities/read_bmatrix_file.cxx )
target_link_libraries(TORTOISEBmatrixToFSLBVecs ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(TORTOISEBMatrixToBT   ../src/tools/TORTOISEBMatrixToBT/tortoise_bmatrix_to_bt.cxx  ../src/utilities/read_bmatrix_file.cxx )
target_link_libraries(TORTOISEBMatrixToBT ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(FSLBVecsToTORTOISEBmatrix   ../src/tools/TORTOISEBmatrixToFSLBVecs/fsl_bvecs_tortoise_bmatrix.cxx  )
target_link_libraries(FSLBVecsToTORTOISEBmatrix ${ITK_LIBRARIES} ${Boost_LIBRARIES} )


add_executable(CreateDummyJson   ../src/tools/CreateDummyJson/CreateDummyJson.cxx ../src/tools/CreateDummyJson/CreateDummyJson_parser.cxx ${SOURCES})
target_link_libraries(CreateDummyJson ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(CreateNonlinearityDisplacementMap ../src/tools/gradnonlin/mk_displacementMaps.cxx ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ${SOURCES})
target_link_libraries(CreateNonlinearityDisplacementMap ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(SelectBestB0 ../src/tools/SelectBestB0/SelectBestB0_main.cxx ../src/main/rigid_register_images.cxx ${SOURCES})
target_link_libraries(SelectBestB0 ${ITK_LIBRARIES} ${Boost_LIBRARIES} )


add_executable(Gibbs ../src/tools/UnRing/gibbs_main.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ${SOURCES})
target_link_libraries(Gibbs ${ITK_LIBRARIES} ${Boost_LIBRARIES} fftw3 )

add_executable(CombineDWIs ../src/tools/CombineDWIs/combine_dwis_main.cxx  ${SOURCES})
target_link_libraries(CombineDWIs ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(CombineDWIsWithBmatrix ../src/tools/CombineDWIs/combine_dwis_with_bmatrix_main.cxx  ${SOURCES})
target_link_libraries(CombineDWIsWithBmatrix ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(Combine4DImages ../src/tools/CombineDWIs/combine_4dimages.cxx  ${SOURCES})
target_link_libraries(Combine4DImages ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ExtractDWISubset ../src/tools/ExtractDWISubset/extract_dwi_subset.cxx   ../src/tools/ExtractDWISubset/extract_dwi_subset_parser.cxx ${SOURCES})
target_link_libraries(ExtractDWISubset ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(IsDirectionPositiveDet ../src/tools/IsDirectionPositiveDet/IsDirectionPositiveDet.cxx  ${SOURCES})
target_link_libraries(IsDirectionPositiveDet ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ExtractImage ../src/tools/ExtractImage/extract_image.cxx  ../src/tools/ExtractImage/extract_image_parser.cxx ${SOURCES} )
target_link_libraries(ExtractImage ${ITK_LIBRARIES} ${Boost_LIBRARIES})


add_executable(ExtractComponentFromDisplacementField ../src/tools/ExtractImage/extract_component_from_itk_vector_image.cxx  ../src/tools/ExtractImage/extract_image_parser.cxx ${SOURCES} )
target_link_libraries(ExtractComponentFromDisplacementField  ${ITK_LIBRARIES} ${Boost_LIBRARIES})


add_executable(SynthesizeDWIsFromTensor ../src/tools/SynthesizeDWIsFromTensor/synthesize_dwis_from_tensor.cxx ../src/tools/EstimateTensor/DTIModel.cxx ${SOURCES} )
target_link_libraries(SynthesizeDWIsFromTensor ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(SynthesizeDWIsFromMAPMRI ../src/tools/SynthesizeDWIFromMAPMRI/synthesize_dwis_from_mapmri.cxx  ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx  ../src/tools/EstimateTensor/DTIModel.cxx ${SOURCES} )
target_link_libraries(SynthesizeDWIsFromMAPMRI ${ITK_LIBRARIES} ${Boost_LIBRARIES})


add_executable(DWIDenoise ../src/tools/DWIDenoise/dwi_denoise.cxx ${SOURCES} )
target_link_libraries(DWIDenoise ${ITK_LIBRARIES} ${Boost_LIBRARIES} Eigen3::Eigen)
target_compile_definitions(DWIDenoise PUBLIC -DNOTORTOISE)


add_executable(ExtractPNGFromNIFTI  ../src/tools/ExtractPNGFromNIFTI/extract_png_from_nifti.cxx)
target_link_libraries(ExtractPNGFromNIFTI ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ExtractAllPNGsFromNIFTI  ../src/tools/ExtractAllPNGsFromNIFTI/extract_all_pngs_from_nifti.cxx)
target_link_libraries(ExtractAllPNGsFromNIFTI ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(Set4DImageDirection  ../src/tools/Set4DImageDirection/set_4dimage_direction.cxx ${SOURCES})
target_link_libraries(Set4DImageDirection ${ITK_LIBRARIES}  ${Boost_LIBRARIES} )

add_executable(Set4DImageOrigin  ../src/tools/Set4DImageOrigin/set_4dimage_origin.cxx ${SOURCES})
target_link_libraries(Set4DImageOrigin ${ITK_LIBRARIES}  ${Boost_LIBRARIES} )

add_executable(Set4DImageResolution  ../src/tools/Set4DImageResolution/Set4DImageResolution.cxx ${SOURCES})
target_link_libraries(Set4DImageResolution ${ITK_LIBRARIES}  ${Boost_LIBRARIES} )


add_executable(ReorientImage ../src/tools/ReorientImage/reorient_image.cxx  ../src/tools/ReorientImage/reorient_image_parser.cxx  ${SOURCES})
target_link_libraries(ReorientImage ${ITK_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(ReorientImage3D ../src/tools/ReorientImage3D/reorient_image3d.cxx  ../src/tools/ReorientImage3D/reorient_image_parser.cxx  ${SOURCES})
target_link_libraries(ReorientImage3D ${ITK_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(EstimateExperimentalDeviationScalar ../src/tools/EstimateExperimentalDeviationScalar/estimate_experimental_deviation_scalar.cxx ../src/tools/EstimateExperimentalDeviationScalar/estimate_experimental_deviation_scalar_parser.cxx  ../src/external_src/cmpfit-1.3a/mpfit.c ../src/tools/EstimateTensor/DTIModel.cxx ${SOURCES})
target_link_libraries(EstimateExperimentalDeviationScalar ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(DIFFPREPAnonymizer ../src/tools/DIFFPREPAnonymizer/diffprep_anonymizer.cxx ../src/tools/DIFFPREPAnonymizer/itkGDCMImageIOOkan.cxx ${SOURCES})
target_link_libraries(DIFFPREPAnonymizer ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ComputeMAPMRI_NG ../src/tools/ComputeMAPMRI_NG/estimate_mapmri_ng.cxx)
target_link_libraries(ComputeMAPMRI_NG ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeMAPMRI_RTOP ../src/tools/ComputeMAPMRI_RTOP/estimate_mapmri_rtop.cxx)
target_link_libraries(ComputeMAPMRI_RTOP ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeMAPMRI_PA ../src/tools/ComputeMAPMRI_PA/estimate_mapmri_pa.cxx)
target_link_libraries(ComputeMAPMRI_PA ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeWLMap  ../src/tools/ComputeWLMap/compute_wl_map.cxx)
target_link_libraries(ComputeWLMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ComputeWPMap  ../src/tools/ComputeWPMap/compute_wp_map.cxx)
target_link_libraries(ComputeWPMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ComputeWSMap  ../src/tools/ComputeWSMap/compute_ws_map.cxx)
target_link_libraries(ComputeWSMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeADMap  ../src/tools/ComputeADMap/compute_ad_map.cxx)
target_link_libraries(ComputeADMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeSKMap  ../src/tools/ComputeSKMap/compute_sk_map.cxx)
target_link_libraries(ComputeSKMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeRDMap  ../src/tools/ComputeRDMap/compute_rd_map.cxx)
target_link_libraries(ComputeRDMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeMODEMap  ../src/tools/ComputeMODEMap/compute_mode_map.cxx)
target_link_libraries(ComputeMODEMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeFAMap  ../src/tools/ComputeFAMap/compute_fa_map.cxx)
target_link_libraries(ComputeFAMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeTRMap  ../src/tools/ComputeTRMap/compute_tr_map.cxx)
target_link_libraries(ComputeTRMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeEigVecImage  ../src/tools/ComputeEigVecImage/compute_eigvec_image.cxx)
target_link_libraries(ComputeEigVecImage ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeLIMap  ../src/tools/ComputeLIMap/compute_li_map.cxx)
target_link_libraries(ComputeLIMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ComputeDECMap  ../src/tools/ComputeDECMap/compute_dec_map.cxx ../src/tools/ComputeDECMap/compute_dec_map_parser.cxx ${SOURCES})
target_link_libraries(ComputeDECMap ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ComputeWLDECMap  ../src/tools/ComputeDECMap/compute_wldec_map.cxx   ../src/tools/ComputeDECMap/compute_dec_map_parser.cxx    ${SOURCES})
target_link_libraries(ComputeWLDECMap ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(ComputeWPDECMap  ../src/tools/ComputeDECMap/compute_wpdec_map.cxx   ../src/tools/ComputeDECMap/compute_dec_map_parser.cxx     ${SOURCES})
target_link_libraries(ComputeWPDECMap ${ITK_LIBRARIES} ${Boost_LIBRARIES} )


add_executable(ComputeNSDECMap  ../src/tools/ComputeNSDECMap/compute_ns_dec_map.cxx   ../src/tools/ComputeNSDECMap/compute_ns_dec_map_parser.cxx     ${SOURCES})
target_link_libraries(ComputeNSDECMap ${ITK_LIBRARIES} ${Boost_LIBRARIES} )


add_executable(ComputeRTOPDTIMaps  ../src/tools/ComputeRTOPDTIMaps/compute_rtop_dti_maps.cxx)
target_link_libraries(ComputeRTOPDTIMaps ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(FlipImage3D  ../src/tools/FlipImage3D/flip_image_3d.cxx)
target_link_libraries(FlipImage3D ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(FlipImage4D  ../src/tools/FlipImage4D/flip_image_4d.cxx)
target_link_libraries(FlipImage4D ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(StdevImages3D  ../src/tools/StdevImages3D/stdev_images_3d.cxx)
target_link_libraries(StdevImages3D ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(MaskTensorWith3D  ../src/tools/MaskTensorWith3D/MaskTensorWith3D.cxx)
target_link_libraries(MaskTensorWith3D ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(Pad4DImage  ../src/tools/Pad4DImage/Pad4DImage.cxx)
target_link_libraries(Pad4DImage ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(Pad3DImage  ../src/tools/Pad4DImage/Pad3DImage.cxx)
target_link_libraries(Pad3DImage ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(FlipSwapBMatrix ../src/tools/FlipSwapBMatrix/FlipSwapBMatrix.cxx ../src/tools/FlipSwapBMatrix/FlipSwapBMatrix_parser.cxx ../src/tools/FlipSwapBMatrix/antsCommandLineParser2.cxx  ${SOURCES})
target_link_libraries(FlipSwapBMatrix ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(Convert4DImageToDisplacementField  ../src/tools/Convert4DImageToDisplacementField/Convert4DImageToDisplacementField.cxx)
target_link_libraries(Convert4DImageToDisplacementField ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(PrintHeaderT ../src/tools/PrintHeaderT/PrintHeader.cxx)
target_link_libraries(PrintHeaderT ${ITK_LIBRARIES}  )

add_executable(FitNIFTIIso
    ../src/tools/FitNIFTIIso/main.cxx
    ../src/tools/FitNIFTIIso/Fit_nifti_iso.cxx
    ../src/tools/FitNIFTIIso/init_iso_gw.cxx
    ../src/tools/FitNIFTIIso/iso_gw.cxx
    ../src/tools/FitNIFTIIso/erodeMask.cxx
    ../src/tools/FitNIFTIIso/fit_nifti_iso_parser.cxx
    ../src/tools/FitNIFTIIso/gradcal.cxx
    ../src/external_src/cmpfit-1.3a/mpfit.c
    ${SOURCES})
target_link_libraries(FitNIFTIIso ${ITK_LIBRARIES} ${Boost_LIBRARIES} Eigen3::Eigen)

add_executable(CreateGradientNonlinearityBMatrix ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx   ../src/tools/CreateGradientNonlinearityBMatrix/CreateGradientNonlinearityBMatrix.cxx  ../src/tools/CreateGradientNonlinearityBMatrix/CreateGradientNonlinearityBMatrix_parser.cxx ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ../src/main/rigid_register_images.cxx ${SOURCES})
target_link_libraries(CreateGradientNonlinearityBMatrix  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(InvertTransformation ../src/tools/InvertTransformation/invert_transformation.cxx)
target_link_libraries(InvertTransformation ${ITK_LIBRARIES}  )


add_executable(ZeroSlices3D ../src/tools/ZeroSlices3D/ZeroSlices3D.cxx ${SOURCES})
target_link_libraries(ZeroSlices3D ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(UnObliqueImage ../src/tools/UnObliqueImage/UnObliqueImage.cxx   ${SOURCES})
target_link_libraries(UnObliqueImage  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ConvertGEGradientsToBMatrix ../src/tools/ConvertGEGradientsToBMatrix/ConvertGEGradientsToBMatrix.cxx   ${SOURCES})
target_link_libraries(ConvertGEGradientsToBMatrix  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(Combine3DImagesTo4D ../src/tools/Combine3DImagesTo4D/Combine3DImagesTo4D.cxx   ${SOURCES})
target_link_libraries(Combine3DImagesTo4D  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ImportBruker ../src/tools/import_bruker/ImportBruker.cxx ../src/tools/import_bruker/bruker_parser.cxx   ../src/utilities/write_bmatrix_file.cxx  ${SOURCES})
target_link_libraries(ImportBruker  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(SmoothTransitionHCPStructural  ../src/tools/SmoothTransitionHCPStructural/SmoothTransitionHCPStructural.cxx)
target_link_libraries(SmoothTransitionHCPStructural ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(MultiplyFieldwithScalar   ../src/tools/MultiplyFieldwithScalar/MultiplyFieldwithScalar.cxx ${SOURCES})
target_link_libraries(MultiplyFieldwithScalar ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ComputeLineField   ../src/tools/ComputeLineField/ComputeLineField.cxx  ${SOURCES})
target_link_libraries(ComputeLineField Eigen3::Eigen ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(CombineTransformations   ../src/tools/CombineTransformations/combine_transformations.cxx  ${SOURCES})
target_link_libraries(CombineTransformations  ${ITK_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(CombineTransformations PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(CombineTransformationsWithOutputName   ../src/tools/CombineTransformations/combine_transformations_with_output_name.cxx ${SOURCES})
target_link_libraries(CombineTransformationsWithOutputName  ${ITK_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(CombineTransformationsWithOutputName PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(CombineTransformationsWithReference   ../src/tools/CombineTransformations/combine_transformations_with_reference.cxx ${SOURCES})
target_link_libraries(CombineTransformationsWithReference  ${ITK_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(CombineTransformationsWithReference PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(DisplacementFieldOperations   ../src/tools/DisplacementFieldOperations/DisplacementFieldOperations.cxx ${SOURCES})
target_link_libraries(DisplacementFieldOperations  ${ITK_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(DisplacementFieldOperations PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(ConstrainDefFields   ../src/tools/DRTAMAS/ConstrainDefFields.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ${SOURCES})
target_link_libraries(ConstrainDefFields  ${ITK_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(ConstrainDefFields PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(ComputeDeformationFieldJacobian   ../src/tools/DRTAMAS/ComputeDeformationFieldJacobian.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ${SOURCES})
target_link_libraries(ComputeDeformationFieldJacobian  ${ITK_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(ComputeDeformationFieldJacobian PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(ConvertRigidHDF2Text   ../src/tools/ConvertRigidHDF2Text/ConvertRigidHDF2Text.cxx  ${SOURCES})
target_link_libraries(ConvertRigidHDF2Text  ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(CombineTORTOISEProcessTransformations   ../src/tools/CombineTORTOISEProcessTransformations/CombineALLTransformations.cxx ../src/utilities/read_3Dvolume_from_4D.cxx ../src/tools/CombineTORTOISEProcessTransformations/CombineALLTransformations_parser.cxx ${SOURCES})
target_link_libraries(CombineTORTOISEProcessTransformations  ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(ComputeRotationMatrixFromTORTOISETransformations   ../src/tools/ComputeRotationMatrixFromTORTOISETransformations/ComputeRotationMatrixFromTORTOISETransformations.cxx ../src/utilities/read_3Dvolume_from_4D.cxx ../src/tools/ComputeRotationMatrixFromTORTOISETransformations/ComputeRotationMatrixFromTORTOISETransformations_parser.cxx ${SOURCES})
target_link_libraries(ComputeRotationMatrixFromTORTOISETransformations  ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(Rigid2Euler   ../src/tools/AffineToRigid/Rigid2Euler.cxx ${SOURCES})
target_link_libraries(Rigid2Euler ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(Affine2Rigid   ../src/tools/AffineToRigid/affine_to_rigid.cxx ${SOURCES})
target_link_libraries(Affine2Rigid ${ITK_LIBRARIES} ${Boost_LIBRARIES})


add_executable(ApplyTransformationToTensor   ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/DRTAMAS/ApplyTransformationToTensorDRT.cxx    ${SOURCES} )
target_link_libraries(ApplyTransformationToTensor   ${ITK_LIBRARIES} ${Boost_LIBRARIES}    ${STATIC_CXX_LIBS})
set_target_properties(ApplyTransformationToTensor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(SelectMostRepresentativeSample   ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx  ../src/tools/DRTAMAS/select_most_representative_sample.cxx    ${SOURCES} )
target_link_libraries(SelectMostRepresentativeSample   ${ITK_LIBRARIES} ${Boost_LIBRARIES}    ${STATIC_CXX_LIBS})
set_target_properties(SelectMostRepresentativeSample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(ApplyTransformationToScalar2 ../src/tools/ApplyTransformationToScalar2/ApplyTransformationToScalar2.cxx ${SOURCES})
target_link_libraries(ApplyTransformationToScalar2 ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(ApplyTransformationToScalar2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(AverageAffineTransformations2 ../src/tools/AverageAffineTransformations2/average_affine_transformations2.cxx ${SOURCES})
target_link_libraries(AverageAffineTransformations2 ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(AverageAffineTransformations2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(AverageDeformationFields ../src/tools/AverageDeformationFields/average_deformation_fields.cxx ${SOURCES})
target_link_libraries(AverageDeformationFields ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(AverageDeformationFields PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(AverageScalars ../src/tools/AverageScalars/average_scalars.cxx ${SOURCES})
target_link_libraries(AverageScalars ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(AverageScalars PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(AverageTensors ../src/tools/AverageTensors/average_tensors.cxx ${SOURCES})
target_link_libraries(AverageTensors ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(AverageTensors PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(AverageTensorsWithWeights ../src/tools/AverageTensorsWithWeights/average_tensors_with_weights.cxx ${SOURCES})
target_link_libraries(AverageTensorsWithWeights ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(AverageTensorsWithWeights PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")


add_executable(GaussianSmoothTensorImage ../src/tools/DRTAMAS/gaussian_smooth_tensor_image.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ${SOURCES})
target_link_libraries(GaussianSmoothTensorImage ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(GaussianSmoothTensorImage PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")

add_executable(ResampleTensor ../src/tools/DRTAMAS/resample_tensor.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ${SOURCES})
target_link_libraries(ResampleTensor ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
set_target_properties(ResampleTensor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")



#add_executable(Test ../src/tools/Test/Test.cxx ${SOURCES})
#target_link_libraries(Test ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

#add_executable(ReorientImageDEC ../src/tools/ReorientImage3D/reorient_imageDEC.cxx  ../src/tools/ReorientImage3D/reorient_image_parser.cxx  ${SOURCES})
#target_link_libraries(ReorientImageDEC ${ITK_LIBRARIES}  ${Boost_LIBRARIES})

#add_executable(AffineTransformationDeterminant  ../src/tools/AffineTransformationDeterminant/AffineTransformationDeterminant.cxx  ${SOURCES})
#target_include_directories(AffineTransformationDeterminant   PRIVATE   )
#target_link_libraries(AffineTransformationDeterminant ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

#add_executable(BrukerBmatToTORTOISEBMTXT   ../src/tools/BrukerBmatToTORTOISEBMTXT/BrukerBmatToTORTOISEBMTXT.cxx ${SOURCES})
#target_link_libraries(BrukerBmatToTORTOISEBMTXT ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${STATIC_CXX_LIBS} )

#add_executable(JoanahUpDownB0Transformations  ../src/tools/JoanahUpDownB0Transformations/JoanahUpDownB0Transformations.cxx ../src/tools/JoanahUpDownB0Transformations/JoanahUpDownB0Transformations_parser.cxx ${SOURCES})
#target_link_libraries(JoanahUpDownB0Transformations ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${STATIC_CXX_LIBS} )

#add_executable(RigidPhaseScaleRegister  ../src/tools/RigidPhaseScaleRegister/RigidPhaseScaleRegister.cxx ../src/main/rigid_register_images.cxx  ${SOURCES})
#target_link_libraries(RigidPhaseScaleRegister ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${STATIC_CXX_LIBS} )


#add_executable(JoanahRegisterAlongAxes  ../src/tools/JoanahUpDownB0Transformations/JoanahRegisterAlongAxes.cxx ../src/main/register_dwi_to_b0.cxx   ../src/tools/ResampleDWIs/resample_dwis.cxx ${SOURCES})
#target_link_libraries(JoanahRegisterAlongAxes ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${STATIC_CXX_LIBS} )


IF(USE_VTK)
    add_definitions(-DUSE_VTK)
    if ( NOT DEFINED VTK_DIR)
        SET(VTK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/VTK-9.4.2_build)
    endif()

    FIND_PACKAGE(VTK)
    IF(VTK_FOUND)
        INCLUDE(${VTK_USE_FILE})
    ELSE(VTK_FOUND)
      MESSAGE(FATAL_ERROR "VTK not found. Please set VTK_DIR.")
    ENDIF(VTK_FOUND)

    string(REPLACE ";" " " _static_flags "${STATIC_CXX_LIBS}")
    set(CMAKE_EXE_LINKER_FLAGS "${_static_flags}")

    add_executable(ComputeGlyphMaps ../src/tools/ComputeGlyphMaps/compute_glyph_maps.cxx  ../src/main/defines.cxx ../src/tools/ComputeGlyphMaps/vtkTensorGlyphOkan.cxx)
    target_link_libraries(ComputeGlyphMaps ${ITK_LIBRARIES} ${VTK_LIBRARIES} -lboost_system  )

    # vtk_module_autoinit is needed
    vtk_module_autoinit(
      TARGETS ComputeGlyphMaps
      MODULES ${VTK_LIBRARIES}
    )

    add_executable(WarpMesh ../src/tools/WarpMesh/WarpMesh_parser.cxx ../src/utilities/read_3Dvolume_from_4D.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/WarpMesh/WarpMesh.cxx  ../src/main/defines.cxx ../src/main/antsCommandLineParser.cxx ../src/main/antsCommandLineOption.cxx)
    target_link_libraries(WarpMesh ${ITK_LIBRARIES} ${VTK_LIBRARIES}  -lboost_system  )
    set_target_properties(WarpMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DRTAMAS")
    # vtk_module_autoinit is needed
    vtk_module_autoinit(
      TARGETS WarpMesh
      MODULES ${VTK_LIBRARIES}
 cd    )
endif(USE_VTK)

# ---------------------------------------------------------------------------
# Variant builds must not overwrite the reference binaries in the shared bin/.
#
# This is applied to EVERY executable in this directory, not to a hand-listed few.
# An earlier version named only TORTOISEProcess_cuda and gpu_replay, which left a
# variant tree free to clobber DRBUDDI_cuda, DRTAMAS_cuda, every WebGPU/CPU pipeline
# target and the ~100 standalone tools - all of which every tree builds. The harness
# resolves binaries by path and detects substitution by content hash, so a silently
# replaced tool shows up as an unexplained staleness failure at best.
#
# Empty suffix for a reference build, so normal builds are unaffected.
if ( NOT TORTOISE_BIN_SUFFIX STREQUAL "" )
    get_property(_tortoise_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_t IN LISTS _tortoise_targets)
        get_target_property(_type ${_t} TYPE)
        if ( _type STREQUAL "EXECUTABLE" )
            get_target_property(_on ${_t} OUTPUT_NAME)
            if ( NOT _on )
                set(_on ${_t})
            endif()
            set_target_properties(${_t} PROPERTIES OUTPUT_NAME "${_on}${TORTOISE_BIN_SUFFIX}")
        endif()
    endforeach()
    list(LENGTH _tortoise_targets _n)
    message(STATUS "Variant build: suffixed executables among ${_n} targets with '${TORTOISE_BIN_SUFFIX}'")
endif()
