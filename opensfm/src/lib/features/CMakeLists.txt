set(FEATURES_FILES
    akaze_bind.h
    hahog.h
    matching.h
    matching_opencl.h
    matching_opencl_kernels.h
    src/akaze_bind.cc
    src/hahog.cc
    src/matching.cc
    src/matching_opencl.cc
)
add_library(features ${FEATURES_FILES})
target_link_libraries(features
  PRIVATE
    ${OpenCV_LIBS}
    akaze
    foundation
    vl
)
target_include_directories(features PRIVATE ${opensfm_SOURCE_DIR})

if(OPENSFM_HAVE_OPENCL)
    target_link_libraries(features PRIVATE OpenCL::OpenCL OpenCL::HeadersCpp)
    # opencl_utils.h is header-only; just need the include path.
    target_include_directories(features PRIVATE ${opensfm_SOURCE_DIR})
endif()

if (OPENSFM_BUILD_TESTS)
    set(FEATURES_TEST_FILES
        test/vlad_test.cc
    )
    add_executable(features_test ${FEATURES_TEST_FILES})
    target_include_directories(features_test PRIVATE ${opensfm_SOURCE_DIR})
    target_link_libraries(features_test
                        PRIVATE
                        features
                        foundation
                        Eigen3::Eigen
                        ${TEST_MAIN}
                        pybind11::embed
                        )
    add_test(features_test features_test)

    set(MATCHING_OPENCL_TEST_FILES
        test/matching_opencl_test.cc
    )
    add_executable(matching_opencl_test ${MATCHING_OPENCL_TEST_FILES})
    target_include_directories(matching_opencl_test PRIVATE ${opensfm_SOURCE_DIR})
    target_link_libraries(matching_opencl_test
                        PRIVATE
                        features
                        foundation
                        Eigen3::Eigen
                        ${TEST_MAIN}
                        pybind11::embed
                        )
    if(OPENSFM_HAVE_OPENCL)
        target_link_libraries(matching_opencl_test PRIVATE OpenCL::OpenCL OpenCL::HeadersCpp)
    endif()
    add_test(matching_opencl_test matching_opencl_test)
endif()

pybind11_add_module(pyfeatures python/pybind.cc)
target_include_directories(pyfeatures PRIVATE ${GLOG_INCLUDE_DIR})
target_link_libraries(pyfeatures
  PRIVATE
    features
    foundation
    akaze
)
set_target_properties(pyfeatures PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
install(TARGETS pyfeatures LIBRARY DESTINATION .)
