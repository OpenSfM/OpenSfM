#pragma once

#include <map/defines.h>
#include <map/observation.h>

#include <vector>

namespace map {

// Flat, append-only pool of Observation objects.
// Shared between TracksManager and Map to avoid duplicating observations.
// Observations are referenced by ObservationIndex (offset into the pool).
class ObservationPool {
 public:
  ObservationIndex Add(const Observation& obs) {
    ObservationIndex idx = observations_.size();
    observations_.push_back(obs);
    return idx;
  }

  const Observation& Get(ObservationIndex idx) const {
    return observations_[idx];
  }

  Observation& GetMutable(ObservationIndex idx) { return observations_[idx]; }

  size_t Size() const { return observations_.size(); }

  void Reserve(size_t n) { observations_.reserve(n); }

  void Clear() {
    decltype(observations_) empty;
    observations_.swap(empty);
  }

 private:
  std::vector<Observation> observations_;
};

}  // namespace map
