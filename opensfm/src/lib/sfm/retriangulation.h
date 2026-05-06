#pragma once
#include <map/map.h>
#include <map/tracks_manager.h>
#include <pybind11/pybind11.h>

#include <unordered_set>

namespace py = pybind11;

namespace sfm::retriangulation {
void RealignMaps(const map::Map& reference, map::Map& to_align,
                 bool update_points);
int Triangulate(map::Map& map,
                const std::unordered_set<map::TrackId>& track_ids,
                float reproj_threshold, float min_angle, float min_depth,
                int processing_threads);
void ReconstructFromTracksManager(map::Map& map,
                                  const map::TracksManager& tracks_manager,
                                  const py::dict& config,
                                  bool use_robust = false);
}  // namespace sfm::retriangulation
