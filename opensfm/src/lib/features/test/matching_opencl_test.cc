#include <features/matching_opencl.h>
#include <foundation/python_types.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <random>
#include <vector>

namespace py = pybind11;

namespace {

// Generate N random 128-dim float32 descriptors.
std::vector<float> RandomDescriptors(int n, int dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> data(n * dim);
  for (auto& v : data) {
    v = dist(rng);
  }
  return data;
}

// CPU brute-force 2-NN for reference.
struct Match {
  int query;
  int ref;
};

std::vector<Match> CpuBruteForce(const float* f1, int n1, const float* f2,
                                 int n2, int dim, float lowes_ratio) {
  float ratio_sq = lowes_ratio * lowes_ratio;
  std::vector<Match> matches;
  for (int i = 0; i < n1; ++i) {
    float bd = INFINITY, sd = INFINITY;
    int bi = -1;
    for (int j = 0; j < n2; ++j) {
      float dist = 0.0f;
      for (int d = 0; d < dim; ++d) {
        float diff = f1[i * dim + d] - f2[j * dim + d];
        dist += diff * diff;
      }
      if (dist < bd) {
        sd = bd;
        bd = dist;
        bi = j;
      } else if (dist < sd) {
        sd = dist;
      }
    }
    if (bi >= 0 && bd < ratio_sq * sd) {
      matches.push_back({i, bi});
    }
  }
  return matches;
}

}  // namespace

TEST(MatchingOpenCL, Availability) {
  // Just check that the function doesn't crash.
  bool avail = features::opencl_matching_available();
  if (!avail) {
    GTEST_SUCCEED() << "OpenCL not available, skipping remaining tests";
    return;
  }
}

TEST(MatchingOpenCL, BasicMatching) {
  if (!features::opencl_matching_available()) {
    GTEST_SUCCEED() << "OpenCL not available";
    return;
  }

  constexpr int dim = 128;
  constexpr int n1 = 200;
  constexpr int n2 = 300;
  constexpr float ratio = 0.8f;

  auto data1 = RandomDescriptors(n1, dim, 42);
  auto data2 = RandomDescriptors(n2, dim, 123);

  // Make some known matches: copy f1[0..9] into f2[50..59] with slight noise.
  std::mt19937 rng(999);
  std::normal_distribution<float> noise(0.0f, 0.001f);
  for (int i = 0; i < 10; ++i) {
    for (int d = 0; d < dim; ++d) {
      data2[(50 + i) * dim + d] = data1[i * dim + d] + noise(rng);
    }
  }

  // Create numpy arrays.
  py::array_t<float> f1({n1, dim});
  py::array_t<float> f2({n2, dim});
  std::memcpy(f1.mutable_data(), data1.data(), data1.size() * sizeof(float));
  std::memcpy(f2.mutable_data(), data2.data(), data2.size() * sizeof(float));

  auto result = features::match_brute_force_opencl(f1, f2, ratio);
  ASSERT_GE(result.shape(0), 10);
  ASSERT_EQ(result.shape(1), 2);

  // Verify the 10 planted matches are found.
  auto r = result.unchecked<2>();
  int found = 0;
  for (int k = 0; k < result.shape(0); ++k) {
    int qi = r(k, 0);
    int ri = r(k, 1);
    if (qi < 10 && ri == qi + 50) {
      ++found;
    }
  }
  EXPECT_EQ(found, 10);
}

TEST(MatchingOpenCL, SymmetricMatching) {
  if (!features::opencl_matching_available()) {
    GTEST_SUCCEED() << "OpenCL not available";
    return;
  }

  constexpr int dim = 128;
  constexpr int n1 = 150;
  constexpr int n2 = 200;
  constexpr float ratio = 0.8f;

  auto data1 = RandomDescriptors(n1, dim, 77);
  auto data2 = RandomDescriptors(n2, dim, 88);

  // Plant 5 mutual matches.
  std::mt19937 rng(555);
  std::normal_distribution<float> noise(0.0f, 0.001f);
  for (int i = 0; i < 5; ++i) {
    for (int d = 0; d < dim; ++d) {
      data2[(30 + i) * dim + d] = data1[(10 + i) * dim + d] + noise(rng);
    }
  }

  py::array_t<float> f1({n1, dim});
  py::array_t<float> f2({n2, dim});
  std::memcpy(f1.mutable_data(), data1.data(), data1.size() * sizeof(float));
  std::memcpy(f2.mutable_data(), data2.data(), data2.size() * sizeof(float));

  auto result = features::match_brute_force_opencl_symmetric(f1, f2, ratio);
  ASSERT_GE(result.shape(0), 5);
  ASSERT_EQ(result.shape(1), 2);

  auto r = result.unchecked<2>();
  int found = 0;
  for (int k = 0; k < result.shape(0); ++k) {
    int qi = r(k, 0);
    int ri = r(k, 1);
    if (qi >= 10 && qi < 15 && ri == qi + 20) {
      ++found;
    }
  }
  EXPECT_EQ(found, 5);

  // Symmetric should return fewer or equal matches than asymmetric.
  auto asym = features::match_brute_force_opencl(f1, f2, ratio);
  EXPECT_LE(result.shape(0), asym.shape(0));
}

TEST(MatchingOpenCL, EmptyInput) {
  if (!features::opencl_matching_available()) {
    GTEST_SUCCEED() << "OpenCL not available";
    return;
  }

  py::array_t<float> f1(std::vector<int>{0, 128});
  py::array_t<float> f2(std::vector<int>{100, 128});

  auto result = features::match_brute_force_opencl(f1, f2, 0.8f);
  EXPECT_EQ(result.shape(0), 0);
  EXPECT_EQ(result.shape(1), 2);

  auto result2 = features::match_brute_force_opencl_symmetric(f1, f2, 0.8f);
  EXPECT_EQ(result2.shape(0), 0);
}

TEST(MatchingOpenCL, ConsistencyWithCPU) {
  if (!features::opencl_matching_available()) {
    GTEST_SUCCEED() << "OpenCL not available";
    return;
  }

  constexpr int dim = 128;
  constexpr int n1 = 100;
  constexpr int n2 = 120;
  constexpr float ratio = 0.75f;

  auto data1 = RandomDescriptors(n1, dim, 1234);
  auto data2 = RandomDescriptors(n2, dim, 5678);

  // Plant exact copies to ensure matches.
  for (int i = 0; i < 8; ++i) {
    for (int d = 0; d < dim; ++d) {
      data2[(i + 10) * dim + d] = data1[i * dim + d];
    }
  }

  py::array_t<float> f1({n1, dim});
  py::array_t<float> f2({n2, dim});
  std::memcpy(f1.mutable_data(), data1.data(), data1.size() * sizeof(float));
  std::memcpy(f2.mutable_data(), data2.data(), data2.size() * sizeof(float));

  auto gpu_result = features::match_brute_force_opencl(f1, f2, ratio);
  auto cpu_matches =
      CpuBruteForce(data1.data(), n1, data2.data(), n2, dim, ratio);

  // GPU and CPU brute-force should produce identical results.
  ASSERT_EQ(gpu_result.shape(0), (int)cpu_matches.size());

  auto r = gpu_result.unchecked<2>();
  for (size_t k = 0; k < cpu_matches.size(); ++k) {
    EXPECT_EQ(r(k, 0), cpu_matches[k].query);
    EXPECT_EQ(r(k, 1), cpu_matches[k].ref);
  }
}
