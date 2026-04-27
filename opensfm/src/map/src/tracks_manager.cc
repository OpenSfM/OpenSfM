#include <foundation/types.h>
#include <foundation/union_find.h>
#include <map/tracks_manager.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_set>

namespace {

template <class S>
int GetTracksFileVersion(S& fstream) {
  const auto current_position = fstream.tellg();

  std::string line;
  std::getline(fstream, line);

  int version = 0;
  if (line.find(map::TracksManager::TRACKS_HEADER) == 0) {
    version = std::atoi(
        line.substr(map::TracksManager::TRACKS_HEADER.length() + 2).c_str());
  } else {
    fstream.seekg(current_position);
  }
  return version;
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

void SeparateLineByTabs(const std::string& line,
                        std::vector<std::string>& elems) {
  elems.clear();
  std::stringstream stst(line);
  std::string elem;
  while (std::getline(stst, elem, '\t'))  // separate by tabs
  {
    elems.push_back(elem);
  }
}

template <class S>
map::TracksManager InstanciateFromStreamV0(S& fstream) {
  map::TracksManager manager;
  std::string line;
  std::vector<std::string> elems;
  constexpr auto N_ENTRIES{8};
  elems.reserve(N_ENTRIES);
  while (std::getline(fstream, line)) {
    SeparateLineByTabs(line, elems);
    if (elems.size() != N_ENTRIES)  // process only valid lines
    {
      throw std::runtime_error(
          "Encountered invalid line. A line must contain exactly " +
          std::to_string(N_ENTRIES) + " values!");
    }
    const map::ShotId image = elems[0];
    const map::TrackId trackID = elems[1];
    const int featureID = std::stoi(elems[2]);
    const double x = std::stod(elems[3]);
    const double y = std::stod(elems[4]);
    const double scale = 0.0;
    const int r = std::stoi(elems[5]);
    const int g = std::stoi(elems[6]);
    const int b = std::stoi(elems[7]);
    auto observation = InstanciateObservation(x, y, scale, featureID, r, g, b);
    manager.AddObservation(image, trackID, observation);
  }
  return manager;
}

template <class S>
map::TracksManager InstanciateFromStreamV1(S& fstream) {
  map::TracksManager manager;
  std::string line;
  std::vector<std::string> elems;
  constexpr auto N_ENTRIES{9};
  elems.reserve(N_ENTRIES);
  while (std::getline(fstream, line)) {
    SeparateLineByTabs(line, elems);
    if (elems.size() != N_ENTRIES)  // process only valid lines
    {
      throw std::runtime_error(
          "Encountered invalid line. A line must contain exactly " +
          std::to_string(N_ENTRIES) + " values!");
    }
    const map::ShotId image = elems[0];
    const map::TrackId trackID = elems[1];
    const int featureID = std::stoi(elems[2]);
    const double x = std::stod(elems[3]);
    const double y = std::stod(elems[4]);
    const double scale = std::stod(elems[5]);
    const int r = std::stoi(elems[6]);
    const int g = std::stoi(elems[7]);
    const int b = std::stoi(elems[8]);
    auto observation = InstanciateObservation(x, y, scale, featureID, r, g, b);
    manager.AddObservation(image, trackID, observation);
  }
  return manager;
}

template <class S>
map::TracksManager InstanciateFromStreamV2(S& fstream) {
  map::TracksManager manager;
  std::string line;
  std::vector<std::string> elems;
  constexpr auto N_ENTRIES{11};
  elems.reserve(N_ENTRIES);
  while (std::getline(fstream, line)) {
    SeparateLineByTabs(line, elems);
    if (elems.size() != N_ENTRIES)  // process only valid lines
    {
      throw std::runtime_error(
          "Encountered invalid line. A line must contain exactly " +
          std::to_string(N_ENTRIES) + " values!");
    }
    const map::ShotId image = elems[0];
    const map::TrackId trackID = elems[1];
    const int featureID = std::stoi(elems[2]);
    const double x = std::stod(elems[3]);
    const double y = std::stod(elems[4]);
    const double scale = std::stod(elems[5]);
    const int r = std::stoi(elems[6]);
    const int g = std::stoi(elems[7]);
    const int b = std::stoi(elems[8]);
    const int segm = std::stoi(elems[9]);
    const int inst = std::stoi(elems[10]);
    auto observation =
        InstanciateObservation(x, y, scale, featureID, r, g, b, segm, inst);
    manager.AddObservation(image, trackID, observation);
  }
  return manager;
}

template <class S>
map::TracksManager InstanciateFromStreamT(S& fstream) {
  const auto version = GetTracksFileVersion(fstream);
  switch (version) {
    case 0:
      return InstanciateFromStreamV0(fstream);
    case 1:
      return InstanciateFromStreamV1(fstream);
    case 2:
      return InstanciateFromStreamV2(fstream);
    default:
      throw std::runtime_error("Unknown tracks manager file version");
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

void TracksManager::AddObservation(const ShotId& shot_id,
                                   const TrackId& track_id,
                                   const Observation& observation) {
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
  std::unordered_set<StringId> allowed_shot_indices;
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
        subset.AddObservation(shot_ids_[shot_idx], track_id,
                              pool_->Get(obs_idx));
        // Carry depth priors into the subset
        const auto depth_it = depth_priors_.find(obs_idx);
        if (depth_it != depth_priors_.end()) {
          subset.SetDepthPrior(shot_ids_[shot_idx], track_id, depth_it->second);
        }
      }
    }
  }
  return subset;
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
  std::unordered_map<FeatureId_2, std::vector<int>, HashPair>
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
  std::ifstream istream(filename);
  if (istream.is_open()) {
    return InstanciateFromStreamT(istream);
  } else {
    throw std::runtime_error("Can't read tracks manager file");
  }
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
  std::stringstream sstream(str);
  return InstanciateFromStreamT(sstream);
}

std::string TracksManager::AsString() const {
  std::stringstream sstream;
  WriteToStreamCurrentVersion(sstream, *this);
  return sstream.str();
}

std::string TracksManager::TRACKS_HEADER = "OPENSFM_TRACKS_VERSION";
int TracksManager::TRACKS_VERSION = 2;
}  // namespace map
