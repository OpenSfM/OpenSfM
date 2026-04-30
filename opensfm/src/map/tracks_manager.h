#pragma once

#include <absl/container/flat_hash_map.h>
#include <map/defines.h>
#include <map/observation.h>
#include <map/observation_pool.h>

#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace map {

class TracksManager {
 public:
  void AddObservation(const ShotId& shot_id, const TrackId& track_id,
                      const Observation& observation);
  // string_view overload: avoids temporary std::string when IDs already exist
  void AddObservationUnchecked(std::string_view shot_id,
                               std::string_view track_id,
                               const Observation& observation);
  Observation GetObservation(const ShotId& shot, const TrackId& track) const;

  // Depth prior management (stored separately from Observation for memory)
  void SetDepthPrior(const ShotId& shot_id, const TrackId& track_id,
                     const Depth& depth);
  std::optional<Depth> GetDepthPrior(const ShotId& shot_id,
                                     const TrackId& track_id) const;
  std::optional<Depth> GetDepthPriorByIndex(ObservationIndex obs_idx) const;

  int NumShots() const;
  int NumTracks() const;
  std::vector<ShotId> GetShotIds() const;
  std::vector<TrackId> GetTrackIds() const;

  // Returns a map of track_id -> observation for a given shot
  // Note: This constructs the map on each call (no longer returns a reference)
  std::unordered_map<TrackId, Observation> GetShotObservations(
      const ShotId& shot) const;
  // Returns a map of shot_id -> observation for a given track
  // Note: This constructs the map on each call (no longer returns a reference)
  std::unordered_map<ShotId, Observation> GetTrackObservations(
      const TrackId& track) const;
  // Returns a map of shot_id -> pool index for a given track (zero-copy)
  std::unordered_map<ShotId, ObservationIndex> GetTrackObservationIndices(
      const TrackId& track) const;

  TracksManager ConstructSubTracksManager(
      const std::vector<TrackId>& tracks,
      const std::vector<ShotId>& shots) const;

  /// Construct a sub-TracksManager by excluding specific shots and tracks.
  /// More efficient than ConstructSubTracksManager when keeping the majority
  /// of the data (e.g. for the largest reconstruction).
  TracksManager ConstructSubTracksManagerByExclusion(
      const std::vector<ShotId>& shots_to_exclude,
      const std::vector<TrackId>& tracks_to_exclude) const;

  using KeyPointTuple = std::tuple<TrackId, Observation, Observation>;
  std::vector<KeyPointTuple> GetAllCommonObservations(
      const ShotId& shot1, const ShotId& shot2) const;
  std::tuple<std::vector<map::TrackId>, MatX2f, MatX2f>
  GetAllCommonObservationsArrays(const ShotId& shot1,
                                 const ShotId& shot2) const;

  using ShotPair = std::pair<ShotId, ShotId>;
  std::unordered_map<ShotPair, int, HashPair> GetAllPairsConnectivity(
      const std::vector<ShotId>& shots,
      const std::vector<TrackId>& tracks) const;

  static TracksManager InstanciateFromFile(const std::string& filename);
  void WriteToFile(const std::string& filename) const;

  static TracksManager InstanciateFromString(const std::string& str);
  std::string AsString() const;

  static TracksManager MergeTracksManager(
      const std::vector<const TracksManager*>& tracks_manager);

  bool HasShotObservations(const ShotId& shot) const;

  // Observation pool access (for sharing with Map)
  const std::shared_ptr<ObservationPool>& GetObservationPool() const {
    return pool_;
  }
  ObservationIndex GetObservationIndex(const ShotId& shot_id,
                                       const TrackId& track_id) const;

  static std::string TRACKS_HEADER;
  static int TRACKS_VERSION;

  /// Flush deferred observations into the adjacency maps.  Must be called
  /// after a batch of AddObservationUnchecked() calls before any query method
  /// is used.  Called automatically by ParseTracksBuffer / AddObservation.
  void BuildAdjacency();

 private:
  // Interning types and helpers
  using StringId = size_t;
  StringId GetShotIndex(const ShotId& id);
  StringId GetTrackIndex(const TrackId& id);
  StringId GetOrInsertShotIndex(const ShotId& id);
  StringId GetOrInsertTrackIndex(const TrackId& id);
  // string_view overloads: use absl heterogeneous lookup, only construct
  // std::string on first insertion.
  StringId GetOrInsertShotIndex(std::string_view id);
  StringId GetOrInsertTrackIndex(std::string_view id);

  // Observation storage via shared pool
  std::shared_ptr<ObservationPool> pool_ = std::make_shared<ObservationPool>();

  // Interning storage
  std::vector<ShotId> shot_ids_;
  std::vector<TrackId> track_ids_;
  absl::flat_hash_map<ShotId, StringId> shot_id_to_index_;
  absl::flat_hash_map<TrackId, StringId> track_id_to_index_;

  // Adjacency lists using integer indices
  // tracks_per_shot_[shot_index] -> map {track_index -> obs_index}
  std::vector<absl::flat_hash_map<StringId, ObservationIndex>> tracks_per_shot_;
  // shots_per_track_[track_index] -> map {shot_index -> obs_index}
  std::vector<absl::flat_hash_map<StringId, ObservationIndex>> shots_per_track_;

  // Sparse depth priors keyed by observation index.
  // Only populated when depth data is available (e.g. from depth maps).
  // Stored separately from Observation to keep the struct compact (48 bytes).
  std::unordered_map<ObservationIndex, Depth> depth_priors_;

  // Deferred bulk-load storage: triples accumulated during
  // AddObservationUnchecked, then flushed into the adjacency maps in one pass
  // by BuildAdjacency().
  struct BulkTriple {
    StringId shot_idx;
    StringId track_idx;
    ObservationIndex obs_idx;
  };
  std::vector<BulkTriple> pending_bulk_;
  bool adjacency_built_ = true;  // true when no pending triples

  // Last-seen shot cache for AddObservationUnchecked
  // (rows are grouped by shot in tracks.csv)
  std::string last_shot_cached_;
  StringId last_shot_idx_ = 0;
};
}  // namespace map
