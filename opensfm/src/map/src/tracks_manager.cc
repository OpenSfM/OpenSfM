#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fast_float/fast_float.h>
#include <foundation/types.h>
#include <foundation/union_find.h>
#include <map/tracks_manager.h>

#include <algorithm>
#include <charconv>
#include <csv2/csv2.hpp>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_set>

namespace {

// csv2 Reader configured for tab-separated values, no header row, trim \r.
using CsvReader =
    csv2::Reader<csv2::delimiter<'\t'>, csv2::quote_character<'"'>,
                 csv2::first_row_is_header<false>,
                 csv2::trim_policy::trim_characters<'\r'>>;

static int svToInt(std::string_view sv) {
  int val = 0;
  std::from_chars(sv.data(), sv.data() + sv.size(), val);
  return val;
}

static double svToDouble(std::string_view sv) {
  double val = 0.0;
  fast_float::from_chars(sv.data(), sv.data() + sv.size(), val);
  return val;
}

// Detect the tracks-file version from the first line of raw data.
// Returns {version, byte_offset_past_header}.  For V0 files (no header)
// both are 0 — meaning the entire buffer is data.
static std::pair<int, size_t> DetectVersion(const char* data, size_t size) {
  size_t lineEnd = 0;
  while (lineEnd < size && data[lineEnd] != '\n') {
    ++lineEnd;
  }

  std::string_view firstLine(data, lineEnd);
  if (!firstLine.empty() && firstLine.back() == '\r') {
    firstLine.remove_suffix(1);
  }

  if (firstLine.find(map::TracksManager::TRACKS_HEADER) == 0) {
    auto vStr =
        firstLine.substr(map::TracksManager::TRACKS_HEADER.length() + 2);
    int version = 0;
    std::from_chars(vStr.data(), vStr.data() + vStr.size(), version);
    const size_t offset = (lineEnd < size) ? lineEnd + 1 : size;
    return {version, offset};
  }
  return {0, 0};
}

map::Observation InstanciateObservation(
    double x, double y, double scale, int id, int r, int g, int b,
    int segm = map::Observation::NO_SEMANTIC_VALUE,
    int inst = map::Observation::NO_SEMANTIC_VALUE) {
  map::Observation observation;
  observation.point << x, y;
  observation.scale = scale;
  observation.feature_id = id;
  observation.color << r, g, b;
  observation.segmentation_id = segm;
  observation.instance_id = inst;
  return observation;
}

// Parse the body of a tracks file (after version header) into a
// TracksManager.  The expected number of columns is determined by |version|.
static map::TracksManager ParseTracksBuffer(const char* data, size_t size,
                                            int version) {
  map::TracksManager manager;
  CsvReader reader;
  reader.parse_view(std::string_view(data, size));

  const int expectedCols = (version == 0) ? 8 : (version == 1) ? 9 : 11;

  for (const auto& row : reader) {
    std::string_view cells[11];
    int n = 0;
    for (const auto& cell : row) {
      if (n < 11) {
        cells[n] = cell.read_view();
      }
      ++n;
    }

    // Skip empty trailing lines (e.g. from a final newline).
    if (n == 0 || (n == 1 && cells[0].empty())) {
      continue;
    }
    if (n != expectedCols) {
      throw std::runtime_error(
          "Encountered invalid line. A line must contain exactly " +
          std::to_string(expectedCols) + " values!");
    }

    const std::string_view image = cells[0];
    const std::string_view trackID = cells[1];
    const int featureID = svToInt(cells[2]);
    const double x = svToDouble(cells[3]);
    const double y = svToDouble(cells[4]);

    map::Observation obs;
    if (version == 0) {
      obs = InstanciateObservation(x, y, 0.0, featureID, svToInt(cells[5]),
                                   svToInt(cells[6]), svToInt(cells[7]));
    } else if (version == 1) {
      obs = InstanciateObservation(x, y, svToDouble(cells[5]), featureID,
                                   svToInt(cells[6]), svToInt(cells[7]),
                                   svToInt(cells[8]));
    } else {
      obs = InstanciateObservation(x, y, svToDouble(cells[5]), featureID,
                                   svToInt(cells[6]), svToInt(cells[7]),
                                   svToInt(cells[8]), svToInt(cells[9]),
                                   svToInt(cells[10]));
    }

    manager.AddObservationUnchecked(image, trackID, obs);
  }

  manager.BuildAdjacency();
  return manager;
}

template <class S>
void WriteToStreamCurrentVersion(S& ostream,
                                 const map::TracksManager& manager) {
  ostream << manager.TRACKS_HEADER << "_v" << manager.TRACKS_VERSION
          << std::endl;
  const auto shotsIDs = manager.GetShotIds();
  for (const auto& shotID : shotsIDs) {
    const auto observations = manager.GetShotObservations(shotID);
    for (const auto& observation : observations) {
      ostream << shotID << "\t" << observation.first << "\t"
              << observation.second.feature_id << "\t"
              << observation.second.point(0) << "\t"
              << observation.second.point(1) << "\t" << observation.second.scale
              << "\t" << observation.second.color(0) << "\t"
              << observation.second.color(1) << "\t"
              << observation.second.color(2) << "\t"
              << observation.second.segmentation_id << "\t"
              << observation.second.instance_id << std::endl;
    }
  }
}

}  // namespace

namespace map {

TracksManager::StringId TracksManager::GetShotIndex(const ShotId& id) {
  const auto it = shot_id_to_index_.find(id);
  if (it == shot_id_to_index_.end()) {
    throw std::runtime_error("Accessing invalid shot ID: " + id);
  }
  return it->second;
}

TracksManager::StringId TracksManager::GetTrackIndex(const TrackId& id) {
  const auto it = track_id_to_index_.find(id);
  if (it == track_id_to_index_.end()) {
    throw std::runtime_error("Accessing invalid track ID: " + id);
  }
  return it->second;
}

TracksManager::StringId TracksManager::GetOrInsertShotIndex(const ShotId& id) {
  const auto it = shot_id_to_index_.find(id);
  if (it != shot_id_to_index_.end()) {
    return it->second;
  }
  const StringId idx = shot_ids_.size();
  shot_ids_.push_back(id);
  shot_id_to_index_[id] = idx;
  tracks_per_shot_.emplace_back();
  return idx;
}

TracksManager::StringId TracksManager::GetOrInsertShotIndex(
    std::string_view id) {
  // Heterogeneous lookup — no std::string constructed for existing keys.
  const auto it = shot_id_to_index_.find(id);
  if (it != shot_id_to_index_.end()) {
    return it->second;
  }
  // New entry — construct string only now.
  const StringId idx = shot_ids_.size();
  shot_ids_.emplace_back(id);
  shot_id_to_index_[shot_ids_.back()] = idx;
  tracks_per_shot_.emplace_back();
  return idx;
}
TracksManager::StringId TracksManager::GetOrInsertTrackIndex(
    const TrackId& id) {
  const auto it = track_id_to_index_.find(id);
  if (it != track_id_to_index_.end()) {
    return it->second;
  }
  const StringId idx = track_ids_.size();
  track_ids_.push_back(id);
  track_id_to_index_[id] = idx;
  shots_per_track_.emplace_back();
  return idx;
}

TracksManager::StringId TracksManager::GetOrInsertTrackIndex(
    std::string_view id) {
  // Heterogeneous lookup — no std::string constructed for existing keys.
  const auto it = track_id_to_index_.find(id);
  if (it != track_id_to_index_.end()) {
    return it->second;
  }
  // New entry — construct string only now.
  const StringId idx = track_ids_.size();
  track_ids_.emplace_back(id);
  track_id_to_index_[track_ids_.back()] = idx;
  shots_per_track_.emplace_back();
  return idx;
}
void TracksManager::AddObservation(const ShotId& shot_id,
                                   const TrackId& track_id,
                                   const Observation& observation) {
  // Flush any pending bulk triples before direct adjacency access.
  BuildAdjacency();

  const StringId shot_idx = GetOrInsertShotIndex(shot_id);
  const StringId track_idx = GetOrInsertTrackIndex(track_id);

  auto& shot_tracks = tracks_per_shot_[shot_idx];
  auto it = shot_tracks.find(track_idx);
  if (it != shot_tracks.end()) {
    pool_->GetMutable(it->second) = observation;
    return;
  }

  // Allocate new index and store observation in pool
  ObservationIndex obs_idx = pool_->Add(observation);

  tracks_per_shot_[shot_idx].emplace(track_idx, obs_idx);
  shots_per_track_[track_idx].emplace(shot_idx, obs_idx);
}

void TracksManager::AddObservationUnchecked(std::string_view shot_id,
                                            std::string_view track_id,
                                            const Observation& observation) {
  // Fast path: cache the last shot_id since rows are grouped by shot.
  StringId shot_idx;
  if (shot_id == last_shot_cached_) {
    shot_idx = last_shot_idx_;
  } else {
    shot_idx = GetOrInsertShotIndex(shot_id);  // string_view overload
    last_shot_cached_ = shot_id;
    last_shot_idx_ = shot_idx;
  }
  const StringId track_idx =
      GetOrInsertTrackIndex(track_id);  // string_view overload

  ObservationIndex obs_idx = pool_->Add(observation);

  // Defer adjacency map insertion — just accumulate triples.
  pending_bulk_.push_back({shot_idx, track_idx, obs_idx});
  adjacency_built_ = false;
}

void TracksManager::BuildAdjacency() {
  if (adjacency_built_) {
    return;
  }

  // Size the outer vectors to match current shot/track counts.
  tracks_per_shot_.resize(shot_ids_.size());
  shots_per_track_.resize(track_ids_.size());

  // Pre-count entries per shot and per track so we can reserve inner maps.
  std::vector<size_t> counts_per_shot(shot_ids_.size(), 0);
  std::vector<size_t> counts_per_track(track_ids_.size(), 0);
  for (const auto& t : pending_bulk_) {
    ++counts_per_shot[t.shot_idx];
    ++counts_per_track[t.track_idx];
  }
  for (size_t i = 0; i < shot_ids_.size(); ++i) {
    tracks_per_shot_[i].reserve(counts_per_shot[i]);
  }
  for (size_t i = 0; i < track_ids_.size(); ++i) {
    shots_per_track_[i].reserve(counts_per_track[i]);
  }

  // Flush all triples into the adjacency maps.
  for (const auto& t : pending_bulk_) {
    tracks_per_shot_[t.shot_idx].emplace(t.track_idx, t.obs_idx);
    shots_per_track_[t.track_idx].emplace(t.shot_idx, t.obs_idx);
  }

  pending_bulk_.clear();
  pending_bulk_.shrink_to_fit();
  adjacency_built_ = true;
}
int TracksManager::NumShots() const { return shot_ids_.size(); }

int TracksManager::NumTracks() const { return track_ids_.size(); }

bool TracksManager::HasShotObservations(const ShotId& shot) const {
  return shot_id_to_index_.count(shot) > 0;
}

void TracksManager::SetDepthPrior(const ShotId& shot_id,
                                  const TrackId& track_id, const Depth& depth) {
  const StringId shot_idx = shot_id_to_index_.at(shot_id);
  const StringId track_idx = track_id_to_index_.at(track_id);
  const auto& shot_tracks = tracks_per_shot_[shot_idx];
  const auto it = shot_tracks.find(track_idx);
  if (it == shot_tracks.end()) {
    throw std::runtime_error("Observation not found for depth prior");
  }
  depth_priors_[it->second] = depth;
}

std::optional<Depth> TracksManager::GetDepthPrior(
    const ShotId& shot_id, const TrackId& track_id) const {
  const auto sit = shot_id_to_index_.find(shot_id);
  if (sit == shot_id_to_index_.end()) return std::nullopt;
  const auto& shot_tracks = tracks_per_shot_[sit->second];
  const auto tit = shot_tracks.find(track_id_to_index_.at(track_id));
  if (tit == shot_tracks.end()) return std::nullopt;
  const auto dit = depth_priors_.find(tit->second);
  if (dit == depth_priors_.end()) return std::nullopt;
  return dit->second;
}

std::optional<Depth> TracksManager::GetDepthPriorByIndex(
    ObservationIndex obs_idx) const {
  const auto it = depth_priors_.find(obs_idx);
  if (it == depth_priors_.end()) return std::nullopt;
  return it->second;
}

std::vector<ShotId> TracksManager::GetShotIds() const { return shot_ids_; }

std::vector<TrackId> TracksManager::GetTrackIds() const { return track_ids_; }

ObservationIndex TracksManager::GetObservationIndex(
    const ShotId& shot_id, const TrackId& track_id) const {
  const StringId shot_idx = shot_id_to_index_.at(shot_id);
  const StringId track_idx = track_id_to_index_.at(track_id);
  const auto& shot_tracks = tracks_per_shot_[shot_idx];
  const auto it = shot_tracks.find(track_idx);
  if (it == shot_tracks.end()) {
    throw std::runtime_error("Accessing invalid track ID");
  }
  return it->second;
}

Observation TracksManager::GetObservation(const ShotId& shot,
                                          const TrackId& track) const {
  // Use map::at to throw if not found, consistent with original implementation
  const StringId shot_idx = shot_id_to_index_.at(shot);
  const StringId track_idx = track_id_to_index_.at(track);

  const auto& shot_tracks = tracks_per_shot_[shot_idx];
  const auto it = shot_tracks.find(track_idx);
  if (it == shot_tracks.end()) {
    throw std::runtime_error("Accessing invalid track ID");
  }
  return pool_->Get(it->second);
}

std::unordered_map<TrackId, Observation> TracksManager::GetShotObservations(
    const ShotId& shot) const {
  const auto it = shot_id_to_index_.find(shot);
  if (it == shot_id_to_index_.end()) {
    throw std::runtime_error("Accessing invalid shot ID");
  }
  const StringId shot_idx = it->second;

  std::unordered_map<TrackId, Observation> result;
  const auto& shot_tracks = tracks_per_shot_[shot_idx];
  result.reserve(shot_tracks.size());

  for (const auto& [track_idx, obs_idx] : shot_tracks) {
    result.emplace(track_ids_[track_idx], pool_->Get(obs_idx));
  }
  return result;
}

std::unordered_map<ShotId, Observation> TracksManager::GetTrackObservations(
    const TrackId& track) const {
  const auto it = track_id_to_index_.find(track);
  if (it == track_id_to_index_.end()) {
    throw std::runtime_error("Accessing invalid track ID");
  }
  const StringId track_idx = it->second;

  std::unordered_map<ShotId, Observation> result;
  const auto& track_shots = shots_per_track_[track_idx];
  result.reserve(track_shots.size());

  for (const auto& [shot_idx, obs_idx] : track_shots) {
    result.emplace(shot_ids_[shot_idx], pool_->Get(obs_idx));
  }
  return result;
}

std::unordered_map<ShotId, ObservationIndex>
TracksManager::GetTrackObservationIndices(const TrackId& track) const {
  const auto it = track_id_to_index_.find(track);
  if (it == track_id_to_index_.end()) {
    throw std::runtime_error("Accessing invalid track ID");
  }
  const StringId track_idx = it->second;

  std::unordered_map<ShotId, ObservationIndex> result;
  const auto& track_shots = shots_per_track_[track_idx];
  result.reserve(track_shots.size());

  for (const auto& [shot_idx, obs_idx] : track_shots) {
    result.emplace(shot_ids_[shot_idx], obs_idx);
  }
  return result;
}

TracksManager TracksManager::ConstructSubTracksManager(
    const std::vector<TrackId>& tracks,
    const std::vector<ShotId>& shots) const {
  absl::flat_hash_set<StringId> allowed_shot_indices;
  allowed_shot_indices.reserve(shots.size());
  for (const auto& id : shots) {
    const auto it = shot_id_to_index_.find(id);
    if (it != shot_id_to_index_.end()) {
      allowed_shot_indices.insert(it->second);
    }
  }

  TracksManager subset;
  for (const auto& track_id : tracks) {
    const auto it_track = track_id_to_index_.find(track_id);
    if (it_track == track_id_to_index_.end()) {
      continue;
    }
    const StringId track_idx = it_track->second;

    const auto& track_shots = shots_per_track_[track_idx];
    for (const auto& [shot_idx, obs_idx] : track_shots) {
      if (allowed_shot_indices.count(shot_idx)) {
        subset.AddObservationUnchecked(shot_ids_[shot_idx], track_id,
                                       pool_->Get(obs_idx));
        // Carry depth priors into the subset
        const auto depth_it = depth_priors_.find(obs_idx);
        if (depth_it != depth_priors_.end()) {
          subset.SetDepthPrior(shot_ids_[shot_idx], track_id, depth_it->second);
        }
      }
    }
  }
  subset.BuildAdjacency();
  return subset;
}

TracksManager TracksManager::ConstructSubTracksManagerByExclusion(
    const std::vector<ShotId>& shots_to_exclude,
    const std::vector<TrackId>& tracks_to_exclude) const {
  // Build internal-index exclusion sets for fast O(1) integer lookups.
  absl::flat_hash_set<StringId> excl_shots;
  excl_shots.reserve(shots_to_exclude.size());
  for (const auto& id : shots_to_exclude) {
    const auto it = shot_id_to_index_.find(id);
    if (it != shot_id_to_index_.end()) {
      excl_shots.insert(it->second);
    }
  }

  absl::flat_hash_set<StringId> excl_tracks;
  excl_tracks.reserve(tracks_to_exclude.size());
  for (const auto& id : tracks_to_exclude) {
    const auto it = track_id_to_index_.find(id);
    if (it != track_id_to_index_.end()) {
      excl_tracks.insert(it->second);
    }
  }

  // Iterate by shot so the last-shot cache in AddObservationUnchecked
  // hits on every call after the first per shot.
  TracksManager result;
  for (StringId shot_idx = 0; shot_idx < shot_ids_.size(); ++shot_idx) {
    if (excl_shots.count(shot_idx)) {
      continue;
    }
    const auto& shot_tracks = tracks_per_shot_[shot_idx];
    for (const auto& [track_idx, obs_idx] : shot_tracks) {
      if (excl_tracks.count(track_idx)) {
        continue;
      }
      result.AddObservationUnchecked(shot_ids_[shot_idx], track_ids_[track_idx],
                                     pool_->Get(obs_idx));
    }
  }
  result.BuildAdjacency();
  return result;
}

std::vector<TracksManager::KeyPointTuple>
TracksManager::GetAllCommonObservations(const ShotId& shot1,
                                        const ShotId& shot2) const {
  const auto find_shot1 = shot_id_to_index_.find(shot1);
  const auto find_shot2 = shot_id_to_index_.find(shot2);
  if (find_shot1 == shot_id_to_index_.end() ||
      find_shot2 == shot_id_to_index_.end()) {
    throw std::runtime_error("Accessing invalid shot ID");
  }

  const StringId idx1 = find_shot1->second;
  const StringId idx2 = find_shot2->second;

  const auto& tracks1 = tracks_per_shot_[idx1];
  const auto& tracks2 = tracks_per_shot_[idx2];

  std::vector<KeyPointTuple> tuples;
  tuples.reserve(std::min(tracks1.size(), tracks2.size()));

  for (const auto& p : tracks1) {
    const auto find = tracks2.find(p.first);
    if (find != tracks2.end()) {
      tuples.emplace_back(track_ids_.at(p.first), pool_->Get(p.second),
                          pool_->Get(find->second));
    }
  }
  return tuples;
}

std::tuple<std::vector<map::TrackId>, MatX2f, MatX2f>
TracksManager::GetAllCommonObservationsArrays(const ShotId& shot1,
                                              const ShotId& shot2) const {
  const auto tuples = GetAllCommonObservations(shot1, shot2);

  std::vector<map::TrackId> track_ids(tuples.size());
  MatX2f points1(tuples.size(), 2);
  MatX2f points2(tuples.size(), 2);
  for (int i = 0; i < tuples.size(); ++i) {
    const auto& [track_id, obs1, obs2] = tuples[i];
    track_ids[i] = track_id;
    points1.row(i) = obs1.point.cast<float>();
    points2.row(i) = obs2.point.cast<float>();
  }
  return {track_ids, points1, points2};
}

std::unordered_map<TracksManager::ShotPair, int, HashPair>
TracksManager::GetAllPairsConnectivity(
    const std::vector<ShotId>& shots,
    const std::vector<TrackId>& tracks) const {
  std::unordered_map<ShotPair, int, HashPair> common_per_pair;

  std::vector<StringId> tracks_to_use;
  if (tracks.empty()) {
    tracks_to_use.resize(track_ids_.size());
    std::iota(tracks_to_use.begin(), tracks_to_use.end(), 0);
  } else {
    tracks_to_use.reserve(tracks.size());
    for (const auto& t_id : tracks) {
      auto it = track_id_to_index_.find(t_id);
      if (it != track_id_to_index_.end()) {
        tracks_to_use.push_back(it->second);
      }
    }
  }

  std::vector<bool> shots_to_use(shot_ids_.size(), false);
  if (shots.empty()) {
    std::fill(shots_to_use.begin(), shots_to_use.end(), true);
  } else {
    for (const auto& s_id : shots) {
      auto it = shot_id_to_index_.find(s_id);
      if (it != shot_id_to_index_.end()) {
        shots_to_use[it->second] = true;
      }
    }
  }

  for (StringId track_idx : tracks_to_use) {
    const auto& track_entries = shots_per_track_[track_idx];

    for (const auto& [shot_idx1, _] : track_entries) {
      if (!shots_to_use[shot_idx1]) {
        continue;
      }
      const auto& shot_id1 = shot_ids_[shot_idx1];
      for (const auto& [shot_idx2, _] : track_entries) {
        if (!shots_to_use[shot_idx2]) {
          continue;
        }
        const auto& shot_id2 = shot_ids_[shot_idx2];
        if (shot_id1 < shot_id2) {
          ++common_per_pair[std::make_pair(shot_id1, shot_id2)];
        }
      }
    }
  }
  return common_per_pair;
}

TracksManager TracksManager::MergeTracksManager(
    const std::vector<const TracksManager*>& tracks_managers) {
  using FeatureId_2 = std::pair<ShotId, int>;
  using TrackRef = std::pair<int, StringId>;
  absl::flat_hash_map<FeatureId_2, std::vector<int>, HashPair>
      observations_per_feature_id;
  std::vector<std::unique_ptr<UnionFindElement<TrackRef>>> union_find_elements;

  for (int mgr_idx = 0; mgr_idx < tracks_managers.size(); ++mgr_idx) {
    const auto* mgr = tracks_managers[mgr_idx];
    for (StringId track_idx = 0; track_idx < mgr->track_ids_.size();
         ++track_idx) {
      const auto element_id = union_find_elements.size();
      for (const auto& [shot_idx, obs_idx] : mgr->shots_per_track_[track_idx]) {
        const auto& obs = mgr->pool_->Get(obs_idx);
        const ShotId& shot_id = mgr->shot_ids_[shot_idx];

        observations_per_feature_id[{shot_id, obs.feature_id}].emplace_back(
            element_id);
      }

      union_find_elements.emplace_back(
          new UnionFindElement<TrackRef>({mgr_idx, track_idx}));
    }
  }

  TracksManager merged;
  if (union_find_elements.empty()) {
    return merged;
  }

  // Union-find any two tracks sharing a common FeatureId_2
  // For N tracks, make 0 the parent of [1, ... N-1[
  for (const auto& tracks_agg : observations_per_feature_id) {
    if (tracks_agg.second.empty()) {
      continue;
    }
    const auto e1 = union_find_elements[tracks_agg.second[0]].get();
    for (int i = 1; i < tracks_agg.second.size(); ++i) {
      const auto e2 = union_find_elements[tracks_agg.second[i]].get();
      Union(e1, e2);
    }
  }

  // Get clusters and construct new tracks
  const auto clusters = GetUnionFindClusters(&union_find_elements);
  for (int i = 0; i < clusters.size(); ++i) {
    const auto& tracks_agg = clusters[i];
    const auto merged_track_id = std::to_string(i);
    // Run over tracks to merged into a new single track
    for (const auto& manager_n_track_id : tracks_agg) {
      const auto manager_id = manager_n_track_id->data.first;
      const auto track_idx = manager_n_track_id->data.second;
      const auto* mgr = tracks_managers[manager_id];

      const auto& observations = mgr->shots_per_track_[track_idx];
      for (const auto& [shot_idx, obs_idx] : observations) {
        merged.AddObservation(mgr->shot_ids_[shot_idx], merged_track_id,
                              mgr->pool_->Get(obs_idx));
        // Carry depth priors into the merged manager
        const auto depth_it = mgr->depth_priors_.find(obs_idx);
        if (depth_it != mgr->depth_priors_.end()) {
          merged.SetDepthPrior(mgr->shot_ids_[shot_idx], merged_track_id,
                               depth_it->second);
        }
      }
    }
  }
  return merged;
}

TracksManager TracksManager::InstanciateFromFile(const std::string& filename) {
  std::error_code ec;
  mio::mmap_source mmap;
  mmap.map(filename, ec);
  if (ec) {
    throw std::runtime_error("Can't read tracks manager file");
  }
  const char* data = mmap.data();
  const size_t size = mmap.length();
  const auto [version, offset] = DetectVersion(data, size);
  return ParseTracksBuffer(data + offset, size - offset, version);
}

void TracksManager::WriteToFile(const std::string& filename) const {
  std::ofstream ostream(filename);
  if (ostream.is_open()) {
    WriteToStreamCurrentVersion(ostream, *this);
  } else {
    throw std::runtime_error("Can't write tracks manager file");
  }
}

TracksManager TracksManager::InstanciateFromString(const std::string& str) {
  const auto [version, offset] = DetectVersion(str.data(), str.size());
  return ParseTracksBuffer(str.data() + offset, str.size() - offset, version);
}

std::string TracksManager::AsString() const {
  std::stringstream sstream;
  WriteToStreamCurrentVersion(sstream, *this);
  return sstream.str();
}

std::string TracksManager::TRACKS_HEADER = "OPENSFM_TRACKS_VERSION";
int TracksManager::TRACKS_VERSION = 2;

}  // namespace map
