set(BUNDLE_FILES
    bundle_adjuster.h
    irls_solver.h
    reconstruction_alignment.h
    data/camera.h
    data/data.h
    data/pose.h
    data/shot.h
    error/error_utils.h
    error/motion_prior_errors.h
    error/position_functors.h
    error/projection_errors.h
    error/absolute_motion_errors.h
    error/parameters_errors.h
    error/relative_motion_errors.h
    src/bundle_adjuster.cc
    src/irls_solver.cc
    src/classic_sigma_reweighting.cc
    src/mad_sigma_reweighting.cc
    src/mixture_reweighting.cc
    src/irls_helpers.cc
)
add_library(bundle ${BUNDLE_FILES})
target_link_libraries(bundle
  PRIVATE
    ${CERES_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${SUITESPARSE_LIBRARIES}
    foundation
)

if (LAPACK_FOUND)
    target_include_directories(bundle PRIVATE ${LAPACK_INCLUDE_DIRS})
endif()
if (SUITESPARSE_FOUND)
    target_include_directories(bundle PRIVATE ${SUITESPARSE_INCLUDE_DIRS})
endif()
target_include_directories(bundle PRIVATE ${CMAKE_SOURCE_DIR})

if (OPENSFM_BUILD_TESTS)
    set(BUNDLE_TEST_FILES
        test/reprojection_errors_test.cc
        test/bundle_data_test.cc
        test/reweighting_test.cc
        test/irls_helpers_test.cc
        test/bundle_adjuster_scene_test.cc
        test/density_ratio_test.cc
    )
    add_executable(bundle_test ${BUNDLE_TEST_FILES})
    target_include_directories(bundle_test
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}
                        ${GMOCK_INCLUDE_DIRS})
    target_link_libraries(bundle_test
                        PRIVATE
                        bundle
                        geometry
                        Eigen3::Eigen
                        foundation
                        ${TEST_MAIN}
                        pybind11::embed
                        )
    add_test(bundle_test bundle_test)
endif()

pybind11_add_module(pybundle python/pybind.cc)
target_link_libraries(pybundle PRIVATE
  bundle
  geometry
  foundation)
set_target_properties(pybundle PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
install(TARGETS pybundle LIBRARY DESTINATION .)
