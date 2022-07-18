cmake_minimum_required(VERSION 3.20)

PROJECT(TORTOISEV4  )

SET(ISDEBUG 0)
SET(USEOPENMP 1)

if ( NOT DEFINED USECUDA)
    SET(USECUDA 1)
endif()
if ( NOT DEFINED USE_VTK)
    SET(USE_VTK 0)
endif()



SET(CMAKE_C_COMPILER /usr/bin/gcc-9)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-9)

message(${CMAKE_BINARY_DIR})
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/lib/ )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/lib/ )


IF(USEOPENMP)
    SET(CMAKE_CXX_FLAGS "-fopenmp -std=c++14 ")
    SET(CMAKE_C_FLAGS "-fopenmp ")
    add_definitions(-DUSE_OPENMP)
ELSE(USEOPENMP)
    SET(CMAKE_CXX_FLAGS "-fPIC ")
    SET(CMAKE_C_FLAGS "-fPIC ")
ENDIF(USEOPENMP)



set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    ON)
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_NO_BOOST_CMAKE ON)
set(BOOST_ROOT /usr/local/boost176)
set(Boost_LIBRARY_DIRS /usr/local/boost176/lib)
set(Boost_INCLUDE_DIRS /usr/local/boost176/include)
find_package(Boost COMPONENTS iostreams filesystem system REQUIRED)

if(Boost_FOUND)  
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  MESSAGE("Boost Found")
else(Boost_FOUND)
   MESSAGE("BOOST NOT FOUND!")
endif()


IF(USECUDA)
    include(CheckLanguage)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda")
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/include/")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    find_package(CUDAToolkit)

    check_language(CUDA)
    message(${CMAKE_CUDA_COMPILER})
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      add_definitions(-DUSECUDA)
      message("CUDA Enabled..")
    else()
      message(STATUS "No CUDA SUPPORT..")
    endif()
ENDIF(USECUDA)


IF(ISDEBUG)
     if ( NOT DEFINED ITK_DIR)
        SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-5.0.1-debug)
    endif()
    FIND_PACKAGE(ITK) 
    IF(ITK_FOUND) 
        INCLUDE(${ITK_USE_FILE})
        MESSAGE("ITK 5.01 Debug version used!")
    ELSE(ITK_FOUND) 
        MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
    ENDIF(ITK_FOUND)
    SET(CMAKE_BUILD_TYPE Debug)
ELSE(ISDEBUG)    
    if ( NOT DEFINED ITK_DIR)
        SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-5.0.1-release)
    endif()
    FIND_PACKAGE(ITK) 
    IF(ITK_FOUND) 
       INCLUDE(${ITK_USE_FILE})
       MESSAGE("ITK 5.01 Release version used!")
    ELSE(ITK_FOUND) 
        MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
    ENDIF(ITK_FOUND)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF(ISDEBUG)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    message("Eigen found.")
message(${EIGEN3_INCLUDE_DIR})
endif()


set(SOURCES ../src/main/defines.cxx  ../src/main/TORTOISE_global.cxx ../src/main/antsCommandLineParser.cxx ../src/main/antsCommandLineOption.cxx  ../src/utilities/TORTOISE_Utilities.cxx ../src/utilities/extract_3Dvolume_from_4D.cxx ../src/utilities/itkNiftiImageIOHeader.cxx ../src/utilities/itkNiftiImageIOHeaderFactory.cxx ../src/utilities/read_3Dvolume_from_4D.cxx ../src/utilities/read_bmatrix_file.cxx ../src/utilities/math_utilities.cxx ../src/external_src/cmpfit-1.3a/mpfit.c)
include_directories(../src/main)

IF(USECUDA)
     include_directories(../src/cuda_src )
     add_executable(TORTOISEProcess_cuda ../src/main/main.cxx  ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx  ../src/tools/RotateBMatrix/rotate_bmatrix.cxx  ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx  ../src/main/EPIREG.cxx ../src/main/TORTOISE_parser.cxx ../src/cuda_src/cuda_image.cxx  ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/main/run_drbuddi_stage.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx  ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/main/DRBUDDI_Diffeo.cxx  ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx ../src/cuda_src/register_dwi_to_b0_cuda.cxx ../src/cuda_src/quadratic_transform_image.cxx ../src/cuda_src/quadratic_transform_image.cu ../src/cuda_src/compute_entropy.cxx ../src/cuda_src/compute_entropy.cu  ${SOURCES} )
     target_include_directories(TORTOISEProcess_cuda   PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(TORTOISEProcess_cuda  Eigen3::Eigen ${ITK_LIBRARIES} ${Boost_LIBRARIES}   fftw3 ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a  -llapack -lblas  -static-libgcc -static-libstdc++)

     add_executable(Vol2SlRegistration_cuda   ../src/tools/Vol2SlRegistration/vol2sl_registration_main.cxx ../src/cuda_src/quadratic_transform_image.cxx ../src/cuda_src/quadratic_transform_image.cu  ../src/cuda_src/cuda_image.cxx ../src/cuda_src/compute_entropy.cxx ../src/cuda_src/compute_entropy.cu ${SOURCES})
     target_include_directories(Vol2SlRegistration_cuda   PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(Vol2SlRegistration_cuda ${ITK_LIBRARIES} ${Boost_LIBRARIES} )



     add_executable(DRBUDDI_cuda ../src/tools/DRBUDDI/DRBUDDI_main.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx   ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx   ../src/cuda_src/cuda_image.cxx  ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/main/run_drbuddi_stage.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx  ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/main/DRBUDDI_Diffeo.cxx   ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx    ${SOURCES} )
     target_include_directories(DRBUDDI_cuda  PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(DRBUDDI_cuda  ${ITK_LIBRARIES} ${Boost_LIBRARIES}    ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a  -llapack -lblas  -static-libgcc -static-libstdc++)
     target_compile_definitions(DRBUDDI_cuda PUBLIC -DDRBUDDIALONE)


ELSE(USECUDA)
    SET(CMAKE_EXE_LINKER_FLAGS "-static ")
    add_executable(TORTOISEProcess ../src/main/main.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx   ../src/tools/RotateBMatrix/rotate_bmatrix.cxx ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx ../src/main/EPIREG.cxx ../src/main/DRBUDDI_Diffeo.cxx ../src/main/run_drbuddi_stage.cxx ../src/main/drbuddi_image_utilities.cxx   ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/main/TORTOISE_parser.cxx  ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ${SOURCES})
    target_link_libraries(TORTOISEProcess Eigen3::Eigen ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan_orig.a fftw3 )

     add_executable(Vol2SlRegistration   ../src/tools/Vol2SlRegistration/vol2sl_registration_main.cxx  ${SOURCES})
     target_include_directories(Vol2SlRegistration   PRIVATE   )
     target_link_libraries(Vol2SlRegistration ${ITK_LIBRARIES} ${Boost_LIBRARIES} )


     add_executable(DRBUDDI ../src/tools/DRBUDDI/DRBUDDI_main.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx   ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx   ../src/main/run_drbuddi_stage.cxx ../src/main/DRBUDDI_Diffeo.cxx   ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx  ../src/main/drbuddi_image_utilities.cxx  ${SOURCES} )
     target_link_libraries(DRBUDDI  ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan_orig.a fftw3 )
     target_compile_definitions(DRBUDDI PUBLIC -DDRBUDDIALONE)


ENDIF(USECUDA)


#    add_executable(CreateMask ../src/tools/CreateMask/create_mask_main.cxx ../src/main/create_mask.cxx ../src/main/defines.cxx)
 #   target_link_libraries(CreateMask ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a -llapack -lblas)


add_executable(EstimateTensor   ../src/tools/EstimateTensor/estimate_tensor_wlls_main.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls_parser.cxx  ${SOURCES})
target_link_libraries(EstimateTensor ${ITK_LIBRARIES} ${Boost_LIBRARIES} )
target_compile_definitions(EstimateTensor PUBLIC -DNOTORTOISE)


add_executable(EstimateMAPMRI   ../src/tools/EstimateMAPMRI/estimate_mapmri_main.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/tools/EstimateMAPMRI/estimate_mapmri_parser.cxx ${SOURCES})
target_link_libraries(EstimateMAPMRI ${ITK_LIBRARIES} ${Boost_LIBRARIES} )
target_compile_definitions(EstimateMAPMRI PUBLIC -DNOTORTOISE)


add_executable(RotateBmatrix   ../src/tools/RotateBMatrix/rotate_bmatrix_main.cxx  ../src/tools/RotateBMatrix/rotate_bmatrix.cxx ${SOURCES})
target_link_libraries(RotateBmatrix ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(TORTOISEBmatrixToFSLBVecs   ../src/tools/TORTOISEBmatrixToFSLBVecs/tortoise_bmatrix_to_fsl_bvecs.cxx  ../src/utilities/read_bmatrix_file.cxx )
target_link_libraries(TORTOISEBmatrixToFSLBVecs ${ITK_LIBRARIES} ${Boost_LIBRARIES} )

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


add_executable(ExtractDWISubset ../src/tools/ExtractDWISubset/extract_dwi_subset.cxx   ../src/tools/ExtractDWISubset/extract_dwi_subset_parser.cxx ${SOURCES})
target_link_libraries(ExtractDWISubset ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(IsDirectionPositiveDet ../src/tools/IsDirectionPositiveDet/IsDirectionPositiveDet.cxx  ${SOURCES})
target_link_libraries(IsDirectionPositiveDet ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


IF(USE_VTK)
    add_definitions(-DUSE_VTK)
    if ( NOT DEFINED VTK_DIR)
        SET(VTK_DIR /qmi_home/irfanogo/Desktop/codes/other_codes/ANTs_09_05_2013/Release/VTK_801_Linux_static)
    endif()

    FIND_PACKAGE(VTK)
    IF(VTK_FOUND)
        INCLUDE(${VTK_USE_FILE})
    ELSE(VTK_FOUND)
      MESSAGE(FATAL_ERROR "VTK not found. Please set VTK_DIR.")
    ENDIF(VTK_FOUND)

    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

    add_executable(ComputeGlyphMaps ../src/tools/ComputeGlyphMaps/compute_glyph_maps.cxx  ../src/main/defines.cxx ../src/tools/ComputeGlyphMaps/vtkTensorGlyphOkan.cxx)
    target_link_libraries(ComputeGlyphMaps ${ITK_LIBRARIES} ${VTK_LIBRARIES} -lboost_system  )
endif(USE_VTK)

