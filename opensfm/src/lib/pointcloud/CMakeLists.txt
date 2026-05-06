set(POINTCLOUD_FILES 
    tile_io.cc
    octree_builder.cc
)

add_library(pointcloud ${POINTCLOUD_FILES})
target_include_directories(pointcloud 
  PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${opensfm_SOURCE_DIR}
)



pybind11_add_module(pypointcloud python/pybind.cc)
target_link_libraries(pypointcloud 
    PRIVATE 
        pointcloud
)

set_target_properties(pypointcloud PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
install(TARGETS pypointcloud LIBRARY DESTINATION .)