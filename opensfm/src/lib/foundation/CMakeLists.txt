set(FOUNDATION_FILES
    memory.h
    stl_extensions.h
    threading.h
    types.h
    newton_raphson.h
    numeric.h
    optional.h
    union_find.h
    src/memory.cc
    src/types.cc
    src/newton_raphson.cc
    src/numeric.cc
)
add_library(foundation ${FOUNDATION_FILES})
target_link_libraries(foundation
  PUBLIC
    ${OpenCV_LIBS}
    ${OpenMP_libomp_LIBRARY}
    Eigen3::Eigen
  PRIVATE
    ${GFLAGS_LIBRARY}
    glog::glog
)
target_include_directories(foundation
  PUBLIC
    ${PYBIND11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    ${opensfm_SOURCE_DIR}
    ${OpenMP_CXX_INCLUDE_DIR}
)

if (OPENSFM_BUILD_TESTS)
    set(FOUNDATION_TEST_FILES
        test/newton_raphson_test.cc
        test/union_find_test.cc
    )
    add_executable(foundation_test ${FOUNDATION_TEST_FILES})
    target_include_directories(foundation_test PRIVATE ${opensfm_SOURCE_DIR} ${GMOCK_INCLUDE_DIRS})
    target_link_libraries(foundation_test
                          PUBLIC
                          foundation
                          ${TEST_MAIN})
    add_test(foundation_test foundation_test)
endif()
