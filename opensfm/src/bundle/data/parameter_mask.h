#pragma once

#include <cstdint>
#include <vector>

namespace bundle {

enum class SimilarityParameterMask : uint8_t {
  None = 0,
  Rx = 1 << 0,
  Ry = 1 << 1,
  Rz = 1 << 2,
  Tx = 1 << 3,
  Ty = 1 << 4,
  Tz = 1 << 5,
  Scale = 1 << 6,
  Translation = Tx | Ty | Tz,
  Rotation = Rx | Ry | Rz,
  All = Translation | Rotation | Scale,
};

inline SimilarityParameterMask operator|(SimilarityParameterMask a,
                                         SimilarityParameterMask b) {
  return static_cast<SimilarityParameterMask>(static_cast<uint8_t>(a) |
                                              static_cast<uint8_t>(b));
}

inline SimilarityParameterMask operator&(SimilarityParameterMask a,
                                         SimilarityParameterMask b) {
  return static_cast<SimilarityParameterMask>(static_cast<uint8_t>(a) &
                                              static_cast<uint8_t>(b));
}

inline SimilarityParameterMask operator~(SimilarityParameterMask a) {
  return static_cast<SimilarityParameterMask>(~static_cast<uint8_t>(a));
}

// Maps SimilarityParameterMask bits to Similarity::Parameter indices.
// The mapping is: Tx->TX(3), Ty->TY(4), Tz->TZ(5),
//                 Rx->RX(0), Ry->RY(1), Rz->RZ(2), Scale->SCALE(6)
// matching the Similarity::Parameter enum layout {RX,RY,RZ,TX,TY,TZ,SCALE}.
inline std::vector<int> SimilarityMaskToIndices(SimilarityParameterMask mask) {
  // bit_index -> Similarity::Parameter index
  constexpr int kBitToParam[] = {
      0,  // bit 0 (Rx)
      1,  // bit 1 (Ry)
      2,  // bit 2 (Rz)
      3,  // bit 3 (Tx)
      4,  // bit 4 (Ty)
      5,  // bit 5 (Tz)
      6,  // bit 6 (Scale) -> SCALE=6
  };

  std::vector<int> indices;
  const auto bits = static_cast<uint8_t>(mask);
  for (int i = 0; i < 7; ++i) {
    if (bits & (1 << i)) {
      indices.push_back(kBitToParam[i]);
    }
  }
  return indices;
}

}  // namespace bundle
