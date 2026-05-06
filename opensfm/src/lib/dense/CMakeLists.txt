set(DENSE_FILES
    cleaner.h
    cluster.h
    depthmap_bind.h
    dsm_opencl_kernels.h
    estimator.h
    fuser.h
    svo.h
    svo_fuser.h
    svo_opencl.h
    svo_opencl_kernels.h
    opencl_kernels.h
    opencl_utils.h
    openmvs_exporter.h
    src/fuser.cc
    src/svo.cc
    src/svo_fuser.cc
)

# Conditionally add GPU sources (require OpenCL).
if(OPENSFM_HAVE_OPENCL)
    list(APPEND DENSE_FILES
        src/cleaner.cc
        src/cluster_estimator.cc
        src/estimator.cc
        src/svo_opencl.cc
    )
endif()

add_library(dense ${DENSE_FILES})
target_link_libraries(dense PUBLIC foundation absl::flat_hash_map PRIVATE vl)
target_include_directories(dense PRIVATE ${opensfm_SOURCE_DIR})

if(OPENSFM_HAVE_OPENCL)
    target_link_libraries(dense PRIVATE OpenCL::OpenCL OpenCL::HeadersCpp)
endif()

if (OPENSFM_BUILD_TESTS)
    set(DENSE_TEST_FILES
        test/depthmap_test.cc
    )
    add_executable(dense_test ${DENSE_TEST_FILES})
    target_include_directories(dense_test PRIVATE ${opensfm_SOURCE_DIR} ${GTEST_INCLUDE_DIRS})
    target_link_libraries(dense_test
                        PUBLIC
                        dense
                        ${TEST_MAIN}
                        ${OpenCV_LIBS})
    add_test(dense_test dense_test)
endif()

pybind11_add_module(pydense python/pybind.cc)
target_include_directories(pydense PRIVATE ${GLOG_INCLUDE_DIR})
target_link_libraries(pydense PRIVATE dense foundation)
if(OPENSFM_HAVE_OPENCL)
    target_link_libraries(pydense PRIVATE OpenCL::OpenCL OpenCL::HeadersCpp)
endif()
set_target_properties(pydense PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
install(TARGETS pydense LIBRARY DESTINATION .)
