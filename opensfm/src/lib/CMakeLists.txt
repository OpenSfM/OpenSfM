cmake_minimum_required(VERSION 3.15)

project(opensfm C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(OpenSfMCommon)

cmake_policy(SET CMP0063 NEW)
# Re-activate once we can have a decent version of CMake (> 3.27)
# cmake_policy(SET CMP0148 NEW)

# Prevent Windows.h from defining min/max macros that break std::min/std::max
if(MSVC)
  add_definitions(-DNOMINMAX)
endif()

# Place compilation results in opensfm/ folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/../..")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${opensfm_SOURCE_DIR}/../..")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${opensfm_SOURCE_DIR}/../..")

####### Compilation Options #######
# Note: Common options and definitions are in ../cmake/OpenSfMCommon.cmake

####### Find Dependencies #######
# Note: Common dependencies are in ../cmake/OpenSfMCommon.cmake

find_package(LAPACK)
find_package(SuiteSparse)
find_package(Ceres)
find_package(absl REQUIRED)

find_library(TCMALLOC_LIBRARY tcmalloc)

# Only link tcmalloc on Linux: on MacOS, it conflicts 
# with the system allocator and causes crashes.
if(UNIX AND NOT APPLE)
    if(TCMALLOC_LIBRARY)
        message(STATUS "Found tcmalloc: ${TCMALLOC_LIBRARY}")
        link_libraries(${TCMALLOC_LIBRARY})
    else()
        message(WARNING "tcmalloc not found. Please install gperftools.")
    endif()
endif()

# Ceres2 exposes Ceres::ceres target.
# Ceres1 exposes just ceres.
# - if there's no such target, cmake will convert it into -lceres
# and the linker will fail
if(Ceres_ceres_FOUND)
    set(CERES_LIBRARIES Ceres::ceres)
else()
    set(CERES_LIBRARIES ceres)
endif()

# Only link against the OpenCV components we actually need
# This avoids pulling in unnecessary dependencies like viz (which requires VTK)
# OpenCV's OpenCVConfig will enforce imgcodecs for < 3.0
# (even if OPTIONAL_COMPONENTS) so we handle version detection carefully
# Cause is imread/imwrite moved to imgcodecs on > 3.0
find_package(OpenCV QUIET)
if(OpenCV_FOUND AND ${OpenCV_VERSION} VERSION_LESS "3.0")
    find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d
                 OPTIONAL_COMPONENTS imgcodecs)
endif()

# Override OpenCV_LIBS to only include the components we explicitly requested
# This prevents linking against all OpenCV modules
# (especially viz which requires VTK)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d)
if(TARGET opencv_imgcodecs)
    list(APPEND OpenCV_LIBS opencv_imgcodecs)
endif()

####### Third party libraries #######
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/akaze)
add_subdirectory(third_party/vlfeat)

# simdjson — SIMD-accelerated JSON parser (amalgamated build)
add_library(simdjson STATIC third_party/simdjson/simdjson.cpp)
target_include_directories(simdjson PUBLIC third_party/simdjson)
# simdjson needs C++17 minimum; silence any warnings from third-party code.
target_compile_features(simdjson PUBLIC cxx_std_17)
set_target_properties(simdjson PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(simdjson PRIVATE -w)

include_directories(third_party/vlfeat)

####### Debugging #######
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    add_executable(debug_c_extension
        debug_c_extension.cc
    )
    target_link_libraries(debug_c_extension PRIVATE pybind11::embed)
endif()


####### Testing #######
option(OPENSFM_BUILD_TESTS "Build OpenSfM unit tests." on)

if (OPENSFM_BUILD_TESTS)
  enable_testing()
  include_directories(third_party/gtest)
  add_definitions(-DCERES_GFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})

  # Gtest is old now and triggers these warnings
  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy-with-user-provided-copy"
    )
  endif()

  # Old bundled gtest tries to use std::tr1 which doesn't exist in modern MSVC.
  # Also, MSVC doesn't report __cplusplus correctly without /Zc:__cplusplus,
  # so gtest thinks C++11 is unavailable and never brings in std::tuple.
  if(MSVC)
    add_definitions(-DGTEST_HAS_TR1_TUPLE=0 -DGTEST_LANG_CXX11=1)
  endif()

  add_library(gtest
              third_party/gtest/gmock_gtest_all.cc
              third_party/gtest/gmock_main.cc)
  target_include_directories(gtest PRIVATE ${GFLAGS_INCLUDE_DIR})
  target_link_libraries(gtest PRIVATE glog::glog)

  set(TEST_MAIN test_main)
  add_library(${TEST_MAIN} testing_main.cc)
  target_link_libraries(${TEST_MAIN}
                        ${GFLAGS_LIBRARY}
                        glog::glog
                        gtest)
endif()

####### OpenCL (optional, for PatchMatch dense reconstruction) #######
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_LIBRARIES} (version ${OpenCL_VERSION_STRING})")

    # Fetch the C++ header wrapper (CL/opencl.hpp) from Khronos.
    include(FetchContent)
    FetchContent_Declare(
        OpenCLHeadersCpp
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
        GIT_TAG        v2025.07.22
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(OpenCLHeadersCpp)
    if(NOT openclheaderscpp_POPULATED)
        FetchContent_Populate(OpenCLHeadersCpp)
    endif()
    add_library(OpenCL::HeadersCpp INTERFACE IMPORTED)
    set_target_properties(OpenCL::HeadersCpp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${openclheaderscpp_SOURCE_DIR}/include"
    )

    set(OPENSFM_HAVE_OPENCL ON CACHE BOOL "OpenCL support available" FORCE)
    add_definitions(-DOPENSFM_HAVE_OPENCL)
else()
    message(STATUS "OpenCL not found — PatchMatch dense reconstruction will be disabled")
    set(OPENSFM_HAVE_OPENCL OFF CACHE BOOL "OpenCL support available" FORCE)
endif()

####### OpenSfM libraries #######
add_subdirectory(foundation)
add_subdirectory(bundle)
add_subdirectory(dense)
add_subdirectory(features)
add_subdirectory(geo)
add_subdirectory(geometry)
add_subdirectory(pointcloud)
add_subdirectory(robust)
add_subdirectory(sfm)
add_subdirectory(map)
