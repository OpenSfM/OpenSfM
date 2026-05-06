#include <foundation/python_types.h>
#include <map/observation.h>
#include <map/tracks_manager.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sfm/ba_helpers.h>
#include <sfm/dense_helpers.h>
#include <sfm/map_helpers.h>
#include <sfm/reconstruction_grower.h>
#include <sfm/retriangulation.h>
#include <sfm/tracks_helpers.h>

PYBIND11_MODULE(pysfm, m) {
  py::module::import("opensfm.pymap");
  py::module::import("opensfm.pygeometry");
  py::module::import("opensfm.pybundle");

  m.def("count_tracks_per_shot", &sfm::tracks_helpers::CountTracksPerShot);
  m.def("add_connections", &sfm::tracks_helpers::AddConnections,
        py::call_guard<py::gil_scoped_release>());
  m.def("filter_badly_conditioned_points",
        &sfm::map_helpers::FilterBadlyConditionedPoints, py::arg("map"),
        py::arg("min_angle_deg") = 1.0, py::arg("min_abs_det") = 1e-15,
        py::call_guard<py::gil_scoped_release>());

  m.def("remove_isolated_points", &sfm::map_helpers::RemoveIsolatedPoints,
        py::arg("map"), py::arg("k") = 7,
        py::call_guard<py::gil_scoped_release>());

  py::class_<sfm::BAHelpers>(m, "BAHelpers")
      .def_static("bundle", &sfm::BAHelpers::Bundle)
      .def_static("bundle_local", &sfm::BAHelpers::BundleLocalPython)
      .def_static("bundle_shot_poses", &sfm::BAHelpers::BundleShotPoses)
      .def_static("bundle_to_map", &sfm::BAHelpers::BundleToMap)
      .def_static("shot_neighborhood_ids", &sfm::BAHelpers::ShotNeighborhoodIds)
      .def_static("detect_alignment_constraints",
                  &sfm::BAHelpers::DetectAlignmentConstraints)
      .def_static("add_gcp_to_bundle", &sfm::BAHelpers::AddGCPToBundle)
      .def_static("remove_outliers", &sfm::BAHelpers::RemoveOutliersPython,
                  py::arg("map"), py::arg("config"),
                  py::arg("point_ids") = std::vector<map::LandmarkId>());

  py::class_<sfm::RigAssignment>(m, "RigAssignment")
      .def(py::init<>())
      .def_readwrite("instance_id", &sfm::RigAssignment::instance_id)
      .def_readwrite("rig_camera_id", &sfm::RigAssignment::rig_camera_id)
      .def_readwrite("instance_shots", &sfm::RigAssignment::instance_shots);

  py::class_<sfm::ReconstructionGrower>(m, "ReconstructionGrower")
      .def_static("grow", &sfm::ReconstructionGrower::Grow, py::arg("map"),
                  py::arg("tracks_manager"), py::arg("camera_priors"),
                  py::arg("rig_camera_priors"), py::arg("shot_camera_map"),
                  py::arg("rig_assignments"), py::arg("images"),
                  py::arg("reconstruction"), py::arg("data"), py::arg("config"))
      .def_static("triangulate_new_tracks",
                  &sfm::ReconstructionGrower::TriangulateNewTracks,
                  py::arg("map"), py::arg("tracks_manager"),
                  py::arg("shot_ids"), py::arg("config"))
      .def_static("parse_exif_dict", &sfm::ReconstructionGrower::ParseExifDict,
                  py::arg("exif"), py::arg("use_altitude"), py::arg("reflat"),
                  py::arg("reflon"), py::arg("refalt"))
      .def_static("triangulation_reconstruction",
                  &sfm::ReconstructionGrower::TriangulationReconstruction,
                  py::arg("map"), py::arg("tracks_manager"),
                  py::arg("camera_priors"), py::arg("rig_camera_priors"),
                  py::arg("grid_size"), py::arg("reconstruction"),
                  py::arg("config"), py::arg("outer_iterations") = 3,
                  py::arg("inner_iterations") = 2);

  m.def("realign_maps", &sfm::retriangulation::RealignMaps,
        py::call_guard<py::gil_scoped_release>());

  m.def("reconstruct_from_tracks_manager",
        &sfm::retriangulation::ReconstructFromTracksManager, py::arg("map"),
        py::arg("tracks_manager"), py::arg("config"),
        py::arg("use_robust") = false,
        py::call_guard<py::gil_scoped_release>());
  
  // ── dense_helpers: super-point covisibility & neighbor selection ────

  py::class_<sfm::dense_helpers::SuperPoint>(m, "SuperPoint")
      .def(py::init<>())
      .def_readwrite("coord", &sfm::dense_helpers::SuperPoint::coord)
      .def_readwrite("vis", &sfm::dense_helpers::SuperPoint::vis)
      .def_readwrite("tracks", &sfm::dense_helpers::SuperPoint::tracks);

  py::class_<sfm::dense_helpers::CovisibilityGraph>(m, "CovisibilityGraph")
      .def_readonly("super_points",
                    &sfm::dense_helpers::CovisibilityGraph::super_points)
      .def_readonly("edges", &sfm::dense_helpers::CovisibilityGraph::edges)
      .def_readonly("weights", &sfm::dense_helpers::CovisibilityGraph::weights)
      .def_readonly("shot_order",
                    &sfm::dense_helpers::CovisibilityGraph::shot_order);

  py::class_<sfm::dense_helpers::NeighborResult>(m, "NeighborResult")
      .def_readonly("best_neighbors",
                    &sfm::dense_helpers::NeighborResult::best_neighbors)
      .def_readonly("all_neighbors",
                    &sfm::dense_helpers::NeighborResult::all_neighbors);

  m.def("build_covisibility_graph", &sfm::dense_helpers::BuildCovisibilityGraph,
        py::arg("tracks_manager"), py::arg("reconstruction"),
        py::arg("processable"), py::arg("fuse_knn") = 15,
        py::arg("fuse_radius_factor") = 0.5,
        py::call_guard<py::gil_scoped_release>());

  m.def("select_neighbors", &sfm::dense_helpers::SelectNeighbors,
        py::arg("tracks_manager"), py::arg("reconstruction"),
        py::arg("super_points"), py::arg("processable"),
        py::arg("num_neighbors"), py::arg("min_point_best") = 20,
        py::arg("min_point_all") = 40, py::arg("theta_min_deg") = 3.0,
        py::arg("theta_max_deg") = 60.0,
        py::call_guard<py::gil_scoped_release>());
}
