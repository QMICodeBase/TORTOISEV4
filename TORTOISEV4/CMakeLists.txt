cmake_minimum_required(VERSION 3.20)

PROJECT(TORTOISEV4  )

SET(USEOPENMP 1)
 #  SET(CMAKE_BUILD_TYPE Debug)
 #  SET(ISDEBUG 1)



if ( NOT DEFINED USECUDA)
    SET(USECUDA  0)
endif()
if ( NOT DEFINED USE_VTK)
    SET(USE_VTK 0)
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



#SET(CMAKE_C_COMPILER /usr/bin/gcc-9)
#SET(CMAKE_CXX_COMPILER /usr/bin/g++-9)

message(${CMAKE_BINARY_DIR})
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/lib/ )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/lib/ )




IF(USEOPENMP)
    SET(CMAKE_CXX_FLAGS "-fopenmp -std=c++14 -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    SET(CMAKE_C_FLAGS "-fopenmp -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    add_definitions(-DUSE_OPENMP)
ELSE(USEOPENMP)
    SET(CMAKE_CXX_FLAGS "-fPIC -std=c++14 -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
    SET(CMAKE_C_FLAGS "-fPIC -Wno-deprecated-declarations -Wno-unused-result -Wno-narrowing")
ENDIF(USEOPENMP)



set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    ON)
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_NO_BOOST_CMAKE ON)
set(BOOST_ROOT /usr/local/boost186)
set(Boost_LIBRARY_DIRS /usr/local/boost186/lib)
set(Boost_INCLUDE_DIRS /usr/local/boost186/include)
find_package(Boost COMPONENTS iostreams filesystem system REQUIRED)

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
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11") # Or higher standard
ENDIF(USECUDA)



IF(ISDEBUG)
     if ( NOT DEFINED ITK_DIR)
       SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-5.3rc04-debug)
    #    SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-5.3rc04-release)  
   #    SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-5.4.0-release)
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
        SET(ITK_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/TORTOISEV4/external_libraries/InsightToolkit-5.3rc04-release)
    endif()
    FIND_PACKAGE(ITK)
    IF(ITK_FOUND)
       INCLUDE(${ITK_USE_FILE})
       MESSAGE("ITK ${ITK_VERSION_MAJOR}${ITK_VERSION_MINOR}${ITK_VERSION_PATCH} Release version used!")
    ELSE(ITK_FOUND)
        MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
    ENDIF(ITK_FOUND)
ENDIF(ISDEBUG)

SET(Eigen3_DIR /qmi_home/irfanogo/Desktop/codes/my_codes/DIFFPREP_CPP/code/external_codes/eigen-3.4.0/bin)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)    
    message("Eigen found.")
    message(${EIGEN3_INCLUDE_DIR})    
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()


set(SOURCES ../src/main/defines.cxx  ../src/main/TORTOISE_global.cxx ../src/main/antsCommandLineParser.cxx ../src/main/antsCommandLineOption.cxx  ../src/utilities/TORTOISE_Utilities.cxx ../src/utilities/extract_3Dvolume_from_4D.cxx  ../src/utilities/read_3Dvolume_from_4D.cxx ../src/utilities/read_bmatrix_file.cxx ../src/utilities/math_utilities.cxx ../src/external_src/cmpfit-1.3a/mpfit.c)
include_directories(../src/main)

IF(USECUDA)
     include_directories(../src/cuda_src )
     add_executable(TORTOISEProcess_cuda ../src/main/main.cxx ../src/main/register_dwi_to_b0.cxx    ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx  ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx  ../src/tools/RotateBMatrix/rotate_bmatrix.cxx  ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx  ../src/main/EPIREG.cxx ../src/main/TORTOISE_parser.cxx ../src/cuda_src/cuda_image.cxx  ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/main/run_drbuddi_stage.cxx  ../src/main/run_drbuddi_stage_TVVF.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx  ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/main/DRBUDDI_Diffeo.cxx  ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx ../src/cuda_src/register_dwi_to_b0_cuda.cxx ../src/cuda_src/quadratic_transform_image.cxx ../src/cuda_src/quadratic_transform_image.cu ../src/cuda_src/compute_entropy.cxx ../src/cuda_src/compute_entropy.cu  ${SOURCES} )
     target_include_directories(TORTOISEProcess_cuda   PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(TORTOISEProcess_cuda  Eigen3::Eigen ${ITK_LIBRARIES} ${Boost_LIBRARIES}   fftw3 ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a -lz   -static-libgcc -static-libstdc++)



     add_executable(DRBUDDI_cuda ../src/tools/DRBUDDI/DRBUDDI_main.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx   ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx   ../src/cuda_src/cuda_image.cxx  ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/main/run_drbuddi_stage.cxx ../src/main/run_drbuddi_stage_TVVF.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx  ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/main/DRBUDDI_Diffeo.cxx   ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx    ${SOURCES} )
     target_include_directories(DRBUDDI_cuda  PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(DRBUDDI_cuda  ${ITK_LIBRARIES} ${Boost_LIBRARIES}    ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a -lz    -static-libgcc -static-libstdc++)
     target_compile_definitions(DRBUDDI_cuda PUBLIC -DDRBUDDIALONE)


     add_executable(DRTAMAS_cuda  ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/drbuddi_image_utilities.cxx ../src/tools/DRTAMAS/compute_metric_dev.cu ../src/tools/DRTAMAS/compute_metric_dev.cxx   ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx ../src/cuda_src/warp_image.cu ../src/cuda_src/warp_image.cxx  ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx ../src/tools/DRTAMAS/run_drtamas_stage.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities.cu   ../src/tools/DRTAMAS/DRTAMAS_Diffeo.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/cuda_image.cxx ../src/main/rigid_register_images.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/DRTAMAS/DRTAMAS_main.cxx ../src/tools/DRTAMAS/DRTAMAS.cxx ../src/tools/DRTAMAS/DRTAMAS_parser.cxx   ${SOURCES} )
     target_include_directories(DRTAMAS_cuda   PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(DRTAMAS_cuda  Eigen3::Eigen  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  -static-libgcc -static-libstdc++ libc.a -lgcc)

  #   add_executable(DRTAMASRigid_cuda  ../src/cuda_src/rigid_transform_image.cu ../src/cuda_src/rigid_transform_image.cxx   ../src/tools/DRTAMAS/run_drtamas_stage_rigid.cxx  ../src/tools/DRTAMAS/compute_metric_dev.cu ../src/tools/DRTAMAS/compute_metric_dev.cxx   ../src/cuda_src/compute_metric.cu ../src/cuda_src/compute_metric.cxx   ../src/cuda_src/gaussian_smooth_image.cu ../src/cuda_src/gaussian_smooth_image.cxx  ../src/tools/DRTAMAS/DRTAMAS_utilities.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities.cu   ../src/tools/DRTAMAS/DRTAMASRigid_Bulk.cxx ../src/cuda_src/resample_image.cu ../src/cuda_src/resample_image.cxx ../src/cuda_src/cuda_image_utilities.cu ../src/cuda_src/cuda_image_utilities.cxx ../src/cuda_src/cuda_image.cxx  ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/DRTAMAS/DRTAMASRigid_main.cxx ../src/tools/DRTAMAS/DRTAMASRigid.cxx ../src/tools/DRTAMAS/DRTAMAS_parser.cxx   ${SOURCES} )
  #   target_include_directories(DRTAMASRigid_cuda   PRIVATE    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  #   target_link_libraries(DRTAMASRigid_cuda   ${ITK_LIBRARIES} ${Boost_LIBRARIES}    -static-libgcc -static-libstdc++ libc.a -lgcc)



ELSE(USECUDA)
    if(NOT QSIPREP)
    SET(CMAKE_EXE_LINKER_FLAGS "-static ")
    endif()
    add_executable(TORTOISEProcess ../src/main/main.cxx  ../src/main/register_dwi_to_b0.cxx  ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx  ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx   ../src/tools/RotateBMatrix/rotate_bmatrix.cxx ../src/main/FINALDATA.cxx ../src/main/TORTOISE.cxx ../src/main/DIFFPREP.cxx ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx ../src/main/EPIREG.cxx ../src/main/DRBUDDI_Diffeo.cxx ../src/main/run_drbuddi_stage.cxx ../src/main/drbuddi_image_utilities.cxx   ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/main/TORTOISE_parser.cxx  ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ${SOURCES})
    target_link_libraries(TORTOISEProcess Eigen3::Eigen ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a -lz fftw3 )


     add_executable(DRBUDDI ../src/tools/DRBUDDI/DRBUDDI_main.cxx ../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx ../src/tools/EstimateTensor/DTIModel.cxx   ../src/main/DRBUDDI.cxx ../src/main/DRBUDDIBase.cxx   ../src/main/run_drbuddi_stage.cxx ../src/main/DRBUDDI_Diffeo.cxx   ../src/tools/ResampleDWIs/resample_dwis.cxx ../src/main/rigid_register_images.cxx ../src/tools/EstimateTensor/estimate_tensor_wlls.cxx ../src/main/create_mask.cxx  ../src/main/drbuddi_image_utilities.cxx  ${SOURCES} )
     target_link_libraries(DRBUDDI  ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a -lz fftw3 )
     target_compile_definitions(DRBUDDI PUBLIC -DDRBUDDIALONE)

     add_executable(Vol2SlRegistration   ../src/tools/Vol2SlRegistration/vol2sl_registration_main.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ${SOURCES})
     target_include_directories(Vol2SlRegistration   PRIVATE   )
     target_link_libraries(Vol2SlRegistration ${ITK_LIBRARIES} ${Boost_LIBRARIES} )


ENDIF(USECUDA)


#add_executable(AffineTransformationDeterminant  ../src/tools/AffineTransformationDeterminant/AffineTransformationDeterminant.cxx  ${SOURCES})
#target_include_directories(AffineTransformationDeterminant   PRIVATE   )
#target_link_libraries(AffineTransformationDeterminant ${ITK_LIBRARIES} ${Boost_LIBRARIES} )




add_executable(ApplyTransformationToTensor   ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/DRTAMAS/ApplyTransformationToTensorDRT.cxx    ${SOURCES} )
target_link_libraries(ApplyTransformationToTensor   ${ITK_LIBRARIES} ${Boost_LIBRARIES}    -static-libgcc -static-libstdc++)

add_executable(SelectMostRepresentativeSample   ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx  ../src/tools/DRTAMAS/select_most_representative_sample.cxx    ${SOURCES} )
target_link_libraries(SelectMostRepresentativeSample   ${ITK_LIBRARIES} ${Boost_LIBRARIES}    -static-libgcc -static-libstdc++)




add_executable(CreateMask ../src/tools/CreateMask/create_mask_main.cxx ../src/main/create_mask.cxx ../src/main/defines.cxx)
target_link_libraries(CreateMask ${ITK_LIBRARIES}  ${Boost_LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/../external_libraries/bet/Linux/libbetokan.a -lz )



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


#add_executable(ConvertParametersToDefFields ../src/tools/ConvertParametersToDefFields/convert_parameters_to_def_fields_parser.cxx ../src/tools/ConvertParametersToDefFields/convert_parameters_to_def_fields.cxx ${SOURCES})
#target_link_libraries(ConvertParametersToDefFields ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )



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

#add_executable(TransformDWIDataWithTextTransformations  ../src/tools/TransformDWIDataWithTextTransformations/transform_dwi_data_with_text_transformations.cxx ../src/Main/itkNiftiImageIOHeader.cxx ../src/Main/itkNiftiImageIOHeaderFactory.cxx ${SOURCES})
#target_link_libraries(TransformDWIDataWithTextTransformations ${ITK_LIBRARIES}  ${Boost_LIBRARIES} )

add_executable(ReorientImage ../src/tools/ReorientImage/reorient_image.cxx  ../src/tools/ReorientImage/reorient_image_parser.cxx  ${SOURCES})
target_link_libraries(ReorientImage ${ITK_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(ReorientImage3D ../src/tools/ReorientImage3D/reorient_image3d.cxx  ../src/tools/ReorientImage3D/reorient_image_parser.cxx  ${SOURCES})
target_link_libraries(ReorientImage3D ${ITK_LIBRARIES}  ${Boost_LIBRARIES})


#add_executable(ReorientImageDEC ../src/tools/ReorientImage3D/reorient_imageDEC.cxx  ../src/tools/ReorientImage3D/reorient_image_parser.cxx  ${SOURCES})
#target_link_libraries(ReorientImageDEC ${ITK_LIBRARIES}  ${Boost_LIBRARIES})


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



add_executable(InvertTransformation ../src/tools/InvertTransformation/invert_transformation.cxx)
target_link_libraries(InvertTransformation ${ITK_LIBRARIES}  )


add_executable(ZeroSlices3D ../src/tools/ZeroSlices3D/ZeroSlices3D.cxx ${SOURCES})
target_link_libraries(ZeroSlices3D ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

add_executable(ApplyTransformationToScalar2 ../src/tools/ApplyTransformationToScalar2/ApplyTransformationToScalar2.cxx ${SOURCES})
target_link_libraries(ApplyTransformationToScalar2 ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(CreateGradientNonlinearityBMatrix ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx   ../src/tools/CreateGradientNonlinearityBMatrix/CreateGradientNonlinearityBMatrix.cxx  ../src/tools/CreateGradientNonlinearityBMatrix/CreateGradientNonlinearityBMatrix_parser.cxx ../src/tools/gradnonlin/init_iso_gw.cxx ../src/tools/gradnonlin/gradcal.cxx ../src/main/rigid_register_images.cxx ${SOURCES})
target_link_libraries(CreateGradientNonlinearityBMatrix  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(UnObliqueImage ../src/tools/UnObliqueImage/UnObliqueImage.cxx   ${SOURCES})
target_link_libraries(UnObliqueImage  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )



add_executable(ConvertGEGradientsToBMatrix ../src/tools/ConvertGEGradientsToBMatrix/ConvertGEGradientsToBMatrix.cxx   ${SOURCES})
target_link_libraries(ConvertGEGradientsToBMatrix  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(Combine3DImagesTo4D ../src/tools/Combine3DImagesTo4D/Combine3DImagesTo4D.cxx   ${SOURCES})
target_link_libraries(Combine3DImagesTo4D  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


add_executable(ImportBruker ../src/tools/import_bruker/ImportBruker.cxx ../src/tools/import_bruker/bruker_parser.cxx   ../src/utilities/write_bmatrix_file.cxx  ${SOURCES})
target_link_libraries(ImportBruker  ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


#add_executable(AnhApplyTransformationstoGhostImages  ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/AnhApplyTransformationstoGhostImages/AnhApplyTransformationstoGhostImages.cxx ${SOURCES})
#target_link_libraries(AnhApplyTransformationstoGhostImages ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


#add_executable(SmoothTransitionHCPStructural  ../src/tools/SmoothTransitionHCPStructural/SmoothTransitionHCPStructural.cxx)
#target_link_libraries(SmoothTransitionHCPStructural ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )

#add_executable(CombineDRBUDDIDownTransformations   ../src/tools/CombineDRBUDDIDownTransformations/CombineDRBUDDIDownTransformations.cxx ${SOURCES})
#target_link_libraries(CombineDRBUDDIDownTransformations ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )



#add_executable(MultiplyFieldwithScalar   ../src/tools/MultiplyFieldwithScalar/MultiplyFieldwithScalar.cxx ${SOURCES})
#target_link_libraries(MultiplyFieldwithScalar ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )


#add_executable(ReorderDWIVolumes   ../src/tools/ReorderDWIVolumes/ReorderDWIVolumes.cxx ${SOURCES})
#target_link_libraries(ReorderDWIVolumes ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )



#add_executable(BrukerBmatToTORTOISEBMTXT   ../src/tools/BrukerBmatToTORTOISEBMTXT/BrukerBmatToTORTOISEBMTXT.cxx ${SOURCES})
#target_link_libraries(BrukerBmatToTORTOISEBMTXT ${ITK_LIBRARIES} ${Boost_LIBRARIES} -static-libgcc -static-libstdc++ )

#add_executable(Affine2Rigid   ../src/tools/AffineToRigid/affine_to_rigid.cxx ${SOURCES})
#target_link_libraries(Affine2Rigid ${ITK_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(Rigid2Euler   ../src/tools/AffineToRigid/Rigid2Euler.cxx ${SOURCES})
#target_link_libraries(Rigid2Euler ${ITK_LIBRARIES} ${Boost_LIBRARIES})


#add_executable(SimulateS2Vmotion   ../src/tools/SimulateS2Vmotion/SimulateS2Vmotion.cxx ../src/tools/EstimateTensor/DTIModel.cxx ../src/tools/EstimateMAPMRI/MAPMRIModel.cxx ${SOURCES})
#target_link_libraries(SimulateS2Vmotion ${ITK_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(OrthoSuperRes   ../src/tools/OrthoSuperRes/OrthoSuperRes.cxx  ${SOURCES})
#target_link_libraries(OrthoSuperRes Eigen3::Eigen ${ITK_LIBRARIES} ${Boost_LIBRARIES})


add_executable(ComputeLineField   ../src/tools/ComputeLineField/ComputeLineField.cxx  ${SOURCES})
target_link_libraries(ComputeLineField Eigen3::Eigen ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(CombineTransformations   ../src/tools/CombineTransformations/combine_transformations.cxx  ${SOURCES})
target_link_libraries(CombineTransformations  ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(CombineTransformationsWithOutputName   ../src/tools/CombineTransformations/combine_transformations_with_output_name.cxx ${SOURCES})
target_link_libraries(CombineTransformationsWithOutputName  ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(CombineTransformationsWithReference   ../src/tools/CombineTransformations/combine_transformations_with_reference.cxx ${SOURCES})
target_link_libraries(CombineTransformationsWithReference  ${ITK_LIBRARIES} ${Boost_LIBRARIES})



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

    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

    add_executable(ComputeGlyphMaps ../src/tools/ComputeGlyphMaps/compute_glyph_maps.cxx  ../src/main/defines.cxx ../src/tools/ComputeGlyphMaps/vtkTensorGlyphOkan.cxx)
    target_link_libraries(ComputeGlyphMaps ${ITK_LIBRARIES} ${VTK_LIBRARIES} -lboost_system  )

    # vtk_module_autoinit is needed
    vtk_module_autoinit(
      TARGETS ComputeGlyphMaps
      MODULES ${VTK_LIBRARIES}
    )

    add_executable(WarpMesh ../src/tools/WarpMesh/WarpMesh_parser.cxx ../src/utilities/read_3Dvolume_from_4D.cxx ../src/tools/DRTAMAS/DRTAMAS_utilities_cp.cxx ../src/tools/WarpMesh/WarpMesh.cxx  ../src/main/defines.cxx ../src/main/antsCommandLineParser.cxx ../src/main/antsCommandLineOption.cxx)
    target_link_libraries(WarpMesh ${ITK_LIBRARIES} ${VTK_LIBRARIES}  -lboost_system  )
    # vtk_module_autoinit is needed
    vtk_module_autoinit(
      TARGETS WarpMesh
      MODULES ${VTK_LIBRARIES}
    )

    add_executable(Cylinders  ../src/tools/WarpMesh/Cylinders.cxx  )
    target_link_libraries(Cylinders ${ITK_LIBRARIES} ${VTK_LIBRARIES}  -lboost_system  )

    vtk_module_autoinit(
       TARGETS Cylinders
       MODULES ${VTK_LIBRARIES}
    )

endif(USE_VTK)

