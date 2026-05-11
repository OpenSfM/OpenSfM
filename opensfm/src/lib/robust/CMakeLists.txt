set(ROBUST_FILES
    model.h
    random_sampler.h
    robust_estimator.h
    essential_model.h
    line_model.h
    absolute_pose_model.h
    relative_pose_model.h
    relative_rotation_model.h
    absolute_pose_known_rotation_model.h
    triangulation_model.h
    src/instanciations.cc
)
add_library(robust ${ROBUST_FILES})
target_link_libraries(robust
  PRIVATE
    foundation
    geometry
)
target_include_directories(robust PUBLIC ${opensfm_SOURCE_DIR})

if (OPENSFM_BUILD_TESTS)
    set(ROBUST_TEST_FILES
        test/line_model_test.cc
        test/scorer_test.cc
    )
    add_executable(robust_test ${ROBUST_TEST_FILES})
    target_include_directories(robust_test
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}
                        ${GMOCK_INCLUDE_DIRS})
    target_link_libraries(robust_test
                        PRIVATE
                        robust
                        foundation
                        Eigen3::Eigen
                        ${TEST_MAIN}
                        )
    add_test(robust_test robust_test)
endif()

pybind11_add_module(pyrobust python/pybind.cc)
target_include_directories(pyrobust PRIVATE ${GLOG_INCLUDE_DIR})
target_link_libraries(pyrobust
  PRIVATE
    robust
    foundation
)
set_target_properties(pyrobust PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
install(TARGETS pyrobust LIBRARY DESTINATION .)
