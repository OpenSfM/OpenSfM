#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pointcloud/octree_builder.h>
#include <pointcloud/tile_format.h>

namespace py = pybind11;

PYBIND11_MODULE(pypointcloud, m) {
  m.doc() = "Octree point cloud tiling for large point cloud visualization.";

  // ── OctreeBuilderConfig ──
  py::class_<pointcloud::OctreeBuilderConfig>(m, "OctreeBuilderConfig")
      .def(py::init<>())
      .def_readwrite("max_points_per_tile",
                     &pointcloud::OctreeBuilderConfig::maxPointsPerTile)
      .def_readwrite("max_depth",
                     &pointcloud::OctreeBuilderConfig::maxDepth)
      .def_readwrite("lod_sample_count",
                     &pointcloud::OctreeBuilderConfig::lodSampleCount)
      .def_readwrite("output_dir",
                     &pointcloud::OctreeBuilderConfig::outputDir);

  // ── OctreeMetadata ──
  py::class_<pointcloud::OctreeMetadata>(m, "OctreeMetadata")
      .def(py::init<>())
      .def_readonly("total_points",
                    &pointcloud::OctreeMetadata::totalPoints)
      .def_readonly("max_depth", &pointcloud::OctreeMetadata::maxDepth)
      .def_readonly("root_spacing",
                    &pointcloud::OctreeMetadata::rootSpacing)
      .def_readonly("aabb_min", &pointcloud::OctreeMetadata::aabbMin)
      .def_readonly("aabb_max", &pointcloud::OctreeMetadata::aabbMax);

  // ── build_octree ──
  m.def(
      "build_octree",
      [](py::array_t<float, py::array::c_style | py::array::forcecast>
             positions,
         py::array_t<float, py::array::c_style | py::array::forcecast> normals,
         py::array_t<uint8_t, py::array::c_style | py::array::forcecast> colors,
         py::array_t<float, py::array::c_style | py::array::forcecast> radii,
         const pointcloud::OctreeBuilderConfig& config)
          -> pointcloud::OctreeMetadata {
        auto pos_buf = positions.request();
        if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
          throw std::runtime_error("positions must be an (N, 3) float32 array");
        }
        uint64_t n = static_cast<uint64_t>(pos_buf.shape[0]);

        pointcloud::BuilderInput input;
        input.positions = static_cast<const float*>(pos_buf.ptr);
        input.numPoints = n;

        // Normals.
        if (normals.size() > 0) {
          auto nrm_buf = normals.request();
          if (nrm_buf.ndim == 2 && nrm_buf.shape[0] == static_cast<long>(n) &&
              nrm_buf.shape[1] == 3) {
            input.normals = static_cast<const float*>(nrm_buf.ptr);
          }
        }

        // Colors.
        if (colors.size() > 0) {
          auto col_buf = colors.request();
          if (col_buf.ndim == 2 && col_buf.shape[0] == static_cast<long>(n) &&
              col_buf.shape[1] == 3) {
            input.colors = static_cast<const uint8_t*>(col_buf.ptr);
          }
        }

        // Radii.
        if (radii.size() > 0) {
          auto rad_buf = radii.request();
          if (rad_buf.ndim == 1 && rad_buf.shape[0] == static_cast<long>(n)) {
            input.radii = static_cast<const float*>(rad_buf.ptr);
          }
        }

        py::gil_scoped_release release;
        return pointcloud::buildOctree(input, config);
      },
      py::arg("positions"), py::arg("normals") = py::array_t<float>(),
      py::arg("colors") = py::array_t<uint8_t>(),
      py::arg("radii") = py::array_t<float>(), py::arg("config") = py::none(),
      R"(Build an octree tile set from a flat point cloud.

Args:
    positions: (N, 3) float32 array of XYZ positions.
    normals: (N, 3) float32 array of normals (optional).
    colors: (N, 3) uint8 array of RGB colors (optional).
    radii: (N,) float32 array of splat radii (optional).
    config: OctreeBuilderConfig with output_dir set.

Returns:
    OctreeMetadata describing the generated octree.
)");
}
