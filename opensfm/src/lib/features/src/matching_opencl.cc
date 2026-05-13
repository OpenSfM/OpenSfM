#include <features/matching_opencl.h>

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/opencl_utils.h>
#include <features/matching_opencl_kernels.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace features {

namespace {

// Default tile size for local-memory tiling of f2 descriptors.
// 16 descriptors × 128 floats × 4 bytes = 8 KB, well within typical
// local memory limits (32–64 KB).
constexpr int kTileSize = 16;

// Maximum number of descriptors per batch when GPU memory is limited.
// Each descriptor is DESC_DIM * 4 bytes; two sets + output buffers.
constexpr size_t kMemoryReserveFraction = 4;  // use at most 1/4 of device mem

// Tile size for local-memory tiling of binary (Hamming) descriptors.
// 256 descriptors × 4 words × 4 bytes = 4 KB — much larger tiles than
// the L2 kernel because binary descriptors are 32× smaller.
constexpr int kHammingTileSize = 256;

struct KNN2Result {
  std::vector<int> best_idx;
  std::vector<float> best_dist;
  std::vector<float> second_dist;
};

/// KNN2 result with integer distances (for Hamming matching).
struct KNN2ResultInt {
  std::vector<int> best_idx;
  std::vector<int> best_dist;
  std::vector<int> second_dist;
};

/// Translate a cl::Error into a std::runtime_error with the error code.
[[noreturn]] void ThrowCLError(const cl::Error& e, const char* context,
                               int n1 = 0, int n2 = 0, int desc_dim = 0) {
  std::string msg = std::string("OpenCL error in ") + context + ": " +
                    e.what() + " (code " + std::to_string(e.err()) + ")";
  if (n1 > 0) {
    msg += " [N1=" + std::to_string(n1) + ", N2=" + std::to_string(n2) +
           ", dim=" + std::to_string(desc_dim) + "]";
  }
  throw std::runtime_error(msg);
}

/// Run the brute_force_knn2 kernel for a batch of queries against all of f2.
/// Must be called WITHOUT the Python GIL held.
KNN2Result RunKNN2Kernel(dense::opencl::CLDevice& device, const float* f1_data,
                         int n1, const float* f2_data, int n2, int desc_dim) {
  try {
    auto& ctx = device.context();
    auto& queue = device.queue();

    const size_t f1_bytes = (size_t)n1 * desc_dim * sizeof(float);
    const size_t f2_bytes = (size_t)n2 * desc_dim * sizeof(float);

    // Create buffers and upload data via explicit writes (more portable
    // than CL_MEM_COPY_HOST_PTR across different OpenCL implementations).
    cl::Buffer buf_f1(ctx, CL_MEM_READ_ONLY, f1_bytes);
    cl::Buffer buf_f2(ctx, CL_MEM_READ_ONLY, f2_bytes);
    cl::Buffer buf_best_idx(ctx, CL_MEM_READ_WRITE, n1 * sizeof(int));
    cl::Buffer buf_best_dist(ctx, CL_MEM_READ_WRITE, n1 * sizeof(float));
    cl::Buffer buf_second_dist(ctx, CL_MEM_READ_WRITE, n1 * sizeof(float));

    queue.enqueueWriteBuffer(buf_f1, CL_TRUE, 0, f1_bytes, f1_data);
    queue.enqueueWriteBuffer(buf_f2, CL_TRUE, 0, f2_bytes, f2_data);

    // Build/cache the program with DESC_DIM and TILE_SIZE defines.
    std::string key = "bf_match_" + std::to_string(desc_dim) + "_" +
                      std::to_string(kTileSize);
    std::string options = "-DDESC_DIM=" + std::to_string(desc_dim) +
                          " -DTILE_SIZE=" + std::to_string(kTileSize);
    cl::Program program =
        device.GetOrBuildProgram(key, kBruteForceMatchKernelSource, options);
    cl::Kernel kernel(program, "brute_force_knn2");

    kernel.setArg(0, buf_f1);
    kernel.setArg(1, buf_f2);
    kernel.setArg(2, buf_best_idx);
    kernel.setArg(3, buf_best_dist);
    kernel.setArg(4, buf_second_dist);
    kernel.setArg(5, n1);
    kernel.setArg(6, n2);

    // Determine work-group size.
    size_t preferred =
        kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            device.device());
    size_t max_wg =
        kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device.device());
    // Use a multiple of preferred that doesn't exceed max_wg, targeting 256.
    size_t wg_size = preferred;
    while (wg_size * 2 <= max_wg && wg_size * 2 <= 256) {
      wg_size *= 2;
    }

    // Local memory: TILE_SIZE * desc_dim * sizeof(float)
    size_t local_mem_needed = (size_t)kTileSize * desc_dim * sizeof(float);
    size_t local_mem_available =
        device.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    if (local_mem_needed > local_mem_available) {
      throw std::runtime_error(
          "OpenCL matching: local memory insufficient for tile size");
    }

    size_t global_size = ((n1 + wg_size - 1) / wg_size) * wg_size;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size),
                               cl::NDRange(wg_size));

    // Wait for kernel to complete before reading back results.
    // This separates kernel execution errors from buffer read errors.
    queue.finish();

    // Read back results.
    KNN2Result result;
    result.best_idx.resize(n1);
    result.best_dist.resize(n1);
    result.second_dist.resize(n1);
    queue.enqueueReadBuffer(buf_best_idx, CL_TRUE, 0, n1 * sizeof(int),
                            result.best_idx.data());
    queue.enqueueReadBuffer(buf_best_dist, CL_TRUE, 0, n1 * sizeof(float),
                            result.best_dist.data());
    queue.enqueueReadBuffer(buf_second_dist, CL_TRUE, 0, n1 * sizeof(float),
                            result.second_dist.data());
    return result;
  } catch (const cl::Error& e) {
    ThrowCLError(e, "RunKNN2Kernel", n1, n2, desc_dim);
  }
}

/// Apply Lowe's ratio test and return (query_idx, ref_idx) pairs.
std::vector<std::pair<int, int>> ApplyRatioTest(const KNN2Result& knn,
                                                float lowes_ratio, int n1,
                                                int query_offset = 0) {
  const float ratio_sq = lowes_ratio * lowes_ratio;
  std::vector<std::pair<int, int>> matches;
  matches.reserve(n1 / 4);  // heuristic pre-alloc
  for (int i = 0; i < n1; ++i) {
    if (knn.best_idx[i] >= 0 &&
        knn.best_dist[i] < ratio_sq * knn.second_dist[i]) {
      matches.emplace_back(query_offset + i, knn.best_idx[i]);
    }
  }
  return matches;
}

/// Convert a vector of (i,j) pairs to a [M x 2] numpy int array.
py::array_t<int> PairsToArray(const std::vector<std::pair<int, int>>& pairs) {
  py::array_t<int> result({(int)pairs.size(), 2});
  auto r = result.mutable_unchecked<2>();
  for (size_t k = 0; k < pairs.size(); ++k) {
    r(k, 0) = pairs[k].first;
    r(k, 1) = pairs[k].second;
  }
  return result;
}

/// Compute how many query descriptors we can process in one batch given
/// the device memory and the fixed f2 buffer size.
int ComputeBatchSize(dense::opencl::CLDevice& device, int n2, int desc_dim) {
  size_t avail = device.GlobalMemSize() / kMemoryReserveFraction;
  size_t f2_bytes = (size_t)n2 * desc_dim * sizeof(float);
  // Per-query overhead: desc_dim*4 (f1 row) + 4 (idx) + 4 (bd) + 4 (sd)
  size_t per_query = (size_t)desc_dim * sizeof(float) + 3 * sizeof(float);
  if (avail <= f2_bytes) {
    // Minimum viable batch.
    return 1024;
  }
  size_t batch = (avail - f2_bytes) / per_query;
  return std::max(1024, (int)std::min(batch, (size_t)INT_MAX));
}

/// Run the Hamming KNN2 kernel for packed binary descriptors.
/// Must be called WITHOUT the Python GIL held.
KNN2ResultInt RunHammingKNN2Kernel(dense::opencl::CLDevice& device,
                                   const uint32_t* f1_data, int n1,
                                   const uint32_t* f2_data, int n2,
                                   int n_words) {
  try {
    auto& ctx = device.context();
    auto& queue = device.queue();

    const size_t f1_bytes = (size_t)n1 * n_words * sizeof(uint32_t);
    const size_t f2_bytes = (size_t)n2 * n_words * sizeof(uint32_t);

    cl::Buffer buf_f1(ctx, CL_MEM_READ_ONLY, f1_bytes);
    cl::Buffer buf_f2(ctx, CL_MEM_READ_ONLY, f2_bytes);
    cl::Buffer buf_best_idx(ctx, CL_MEM_READ_WRITE, n1 * sizeof(int));
    cl::Buffer buf_best_dist(ctx, CL_MEM_READ_WRITE, n1 * sizeof(int));
    cl::Buffer buf_second_dist(ctx, CL_MEM_READ_WRITE, n1 * sizeof(int));

    queue.enqueueWriteBuffer(buf_f1, CL_TRUE, 0, f1_bytes, f1_data);
    queue.enqueueWriteBuffer(buf_f2, CL_TRUE, 0, f2_bytes, f2_data);

    std::string key = "hamming_match_" + std::to_string(n_words) + "_" +
                      std::to_string(kHammingTileSize);
    std::string options =
        "-DN_WORDS=" + std::to_string(n_words) +
        " -DHAMMING_TILE_SIZE=" + std::to_string(kHammingTileSize);
    cl::Program program =
        device.GetOrBuildProgram(key, kHammingMatchKernelSource, options);
    cl::Kernel kernel(program, "brute_force_hamming_knn2");

    kernel.setArg(0, buf_f1);
    kernel.setArg(1, buf_f2);
    kernel.setArg(2, buf_best_idx);
    kernel.setArg(3, buf_best_dist);
    kernel.setArg(4, buf_second_dist);
    kernel.setArg(5, n1);
    kernel.setArg(6, n2);

    // Check local memory for the tile.
    size_t local_mem_needed =
        (size_t)kHammingTileSize * n_words * sizeof(uint32_t);
    size_t local_mem_available =
        device.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    if (local_mem_needed > local_mem_available) {
      throw std::runtime_error(
          "OpenCL Hamming matching: local memory insufficient for tile size");
    }

    size_t preferred =
        kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            device.device());
    size_t max_wg =
        kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device.device());
    size_t wg_size = preferred;
    while (wg_size * 2 <= max_wg && wg_size * 2 <= 256) {
      wg_size *= 2;
    }

    size_t global_size = ((n1 + wg_size - 1) / wg_size) * wg_size;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size),
                               cl::NDRange(wg_size));
    queue.finish();

    KNN2ResultInt result;
    result.best_idx.resize(n1);
    result.best_dist.resize(n1);
    result.second_dist.resize(n1);
    queue.enqueueReadBuffer(buf_best_idx, CL_TRUE, 0, n1 * sizeof(int),
                            result.best_idx.data());
    queue.enqueueReadBuffer(buf_best_dist, CL_TRUE, 0, n1 * sizeof(int),
                            result.best_dist.data());
    queue.enqueueReadBuffer(buf_second_dist, CL_TRUE, 0, n1 * sizeof(int),
                            result.second_dist.data());
    return result;
  } catch (const cl::Error& e) {
    ThrowCLError(e, "RunHammingKNN2Kernel", n1, n2, n_words);
  }
}

/// Apply Lowe's ratio test on integer Hamming distances.
/// For Hamming distances (linear, not squared), the test is:
///   best_dist < ratio * second_dist
std::vector<std::pair<int, int>> ApplyRatioTestHamming(const KNN2ResultInt& knn,
                                                       float lowes_ratio,
                                                       int n1,
                                                       int query_offset = 0,
                                                       int read_offset = 0) {
  std::vector<std::pair<int, int>> matches;
  matches.reserve(n1 / 4);
  for (int i = 0; i < n1; ++i) {
    int ri = read_offset + i;
    if (knn.best_idx[ri] >= 0 &&
        (float)knn.best_dist[ri] < lowes_ratio * (float)knn.second_dist[ri]) {
      matches.emplace_back(query_offset + i, knn.best_idx[ri]);
    }
  }
  return matches;
}

/// Per-pair offsets for the batched kernel.
struct PairOffsets {
  int f1_offset;
  int n1;
  int f2_offset;
  int n2;
};

/// Run the batched Hamming KNN2 kernel across all pairs in one dispatch.
/// Must be called WITHOUT the Python GIL held.
KNN2ResultInt RunBatchedHammingKNN2Kernel(
    dense::opencl::CLDevice& device, const uint32_t* query_data,
    int total_queries, const uint32_t* ref_data, int total_refs,
    const std::vector<PairOffsets>& pairs, int n_words) {
  try {
    auto& ctx = device.context();
    auto& queue = device.queue();

    const size_t q_bytes = (size_t)total_queries * n_words * sizeof(uint32_t);
    const size_t r_bytes = (size_t)total_refs * n_words * sizeof(uint32_t);

    cl::Buffer buf_q(ctx, CL_MEM_READ_ONLY, q_bytes);
    cl::Buffer buf_r(ctx, CL_MEM_READ_ONLY, r_bytes);
    cl::Buffer buf_best_idx(ctx, CL_MEM_READ_WRITE,
                            total_queries * sizeof(int));
    cl::Buffer buf_best_dist(ctx, CL_MEM_READ_WRITE,
                             total_queries * sizeof(int));
    cl::Buffer buf_second_dist(ctx, CL_MEM_READ_WRITE,
                               total_queries * sizeof(int));

    queue.enqueueWriteBuffer(buf_q, CL_TRUE, 0, q_bytes, query_data);
    queue.enqueueWriteBuffer(buf_r, CL_TRUE, 0, r_bytes, ref_data);

    // Build/cache the batched program.
    std::string key = "hamming_match_batched_" + std::to_string(n_words) + "_" +
                      std::to_string(kHammingTileSize);
    std::string options =
        "-DN_WORDS=" + std::to_string(n_words) +
        " -DHAMMING_TILE_SIZE=" + std::to_string(kHammingTileSize);
    cl::Program program = device.GetOrBuildProgram(
        key, kHammingMatchBatchedKernelSource, options);
    cl::Kernel kernel(program, "brute_force_hamming_knn2_batched");

    // Determine work-group size.
    size_t preferred =
        kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            device.device());
    size_t max_wg =
        kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device.device());
    size_t wg_size = preferred;
    while (wg_size * 2 <= max_wg && wg_size * 2 <= 256) {
      wg_size *= 2;
    }

    // Build per-work-group info: [f1_off, n1_valid, f2_off, n2].
    int total_wgs = 0;
    for (auto& p : pairs) {
      total_wgs += (p.n1 + (int)wg_size - 1) / (int)wg_size;
    }

    std::vector<int> wg_info(total_wgs * 4);
    int wg_idx = 0;
    for (auto& p : pairs) {
      for (int off = 0; off < p.n1; off += (int)wg_size) {
        int valid = std::min((int)wg_size, p.n1 - off);
        wg_info[wg_idx * 4 + 0] = p.f1_offset + off;
        wg_info[wg_idx * 4 + 1] = valid;
        wg_info[wg_idx * 4 + 2] = p.f2_offset;
        wg_info[wg_idx * 4 + 3] = p.n2;
        ++wg_idx;
      }
    }

    cl::Buffer buf_wg_info(ctx, CL_MEM_READ_ONLY, wg_info.size() * sizeof(int));
    queue.enqueueWriteBuffer(buf_wg_info, CL_TRUE, 0,
                             wg_info.size() * sizeof(int), wg_info.data());

    kernel.setArg(0, buf_q);
    kernel.setArg(1, buf_r);
    kernel.setArg(2, buf_best_idx);
    kernel.setArg(3, buf_best_dist);
    kernel.setArg(4, buf_second_dist);
    kernel.setArg(5, buf_wg_info);
    kernel.setArg(6, total_queries);

    size_t global_size = (size_t)total_wgs * wg_size;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size),
                               cl::NDRange(wg_size));
    queue.finish();

    KNN2ResultInt result;
    result.best_idx.resize(total_queries);
    result.best_dist.resize(total_queries);
    result.second_dist.resize(total_queries);
    queue.enqueueReadBuffer(buf_best_idx, CL_TRUE, 0,
                            total_queries * sizeof(int),
                            result.best_idx.data());
    queue.enqueueReadBuffer(buf_best_dist, CL_TRUE, 0,
                            total_queries * sizeof(int),
                            result.best_dist.data());
    queue.enqueueReadBuffer(buf_second_dist, CL_TRUE, 0,
                            total_queries * sizeof(int),
                            result.second_dist.data());
    return result;
  } catch (const cl::Error& e) {
    ThrowCLError(e, "RunBatchedHammingKNN2Kernel", total_queries, total_refs,
                 n_words);
  }
}

/// Batch size for Hamming matching (binary descriptors are much smaller).
int ComputeHammingBatchSize(dense::opencl::CLDevice& device, int n2,
                            int n_words) {
  size_t avail = device.GlobalMemSize() / kMemoryReserveFraction;
  size_t f2_bytes = (size_t)n2 * n_words * sizeof(uint32_t);
  size_t per_query = (size_t)n_words * sizeof(uint32_t) + 3 * sizeof(int);
  if (avail <= f2_bytes) {
    return 1024;
  }
  size_t batch = (avail - f2_bytes) / per_query;
  return std::max(1024, (int)std::min(batch, (size_t)INT_MAX));
}

}  // namespace

bool opencl_matching_available() {
  return dense::opencl::CLContext::Instance().IsAvailable();
}

int opencl_num_devices() {
  return dense::opencl::CLContext::Instance().NumDevices();
}

py::array_t<int> match_brute_force_opencl(foundation::pyarray_f f1,
                                          foundation::pyarray_f f2,
                                          float lowes_ratio, int device_idx) {
  if (!opencl_matching_available()) {
    throw std::runtime_error("OpenCL is not available");
  }

  auto& cl_ctx = dense::opencl::CLContext::Instance();
  if (device_idx < 0 || device_idx >= cl_ctx.NumDevices()) {
    throw std::runtime_error("Invalid OpenCL device index");
  }
  auto& device = cl_ctx.Device(device_idx);

  const int n1 = static_cast<int>(f1.shape(0));
  const int n2 = static_cast<int>(f2.shape(0));
  const int desc_dim = static_cast<int>(f1.shape(1));

  if (n1 == 0 || n2 == 0) {
    return py::array_t<int>(std::vector<int>{0, 2});
  }
  if (desc_dim != static_cast<int>(f2.shape(1))) {
    throw std::runtime_error(
        "Descriptor dimensions must match between f1 and f2");
  }
  if (desc_dim % 4 != 0) {
    throw std::runtime_error(
        "Descriptor dimension must be a multiple of 4 for OpenCL matching");
  }

  const float* f1_data = f1.data();
  const float* f2_data = f2.data();

  // --- Release GIL for the heavy GPU computation ---
  std::vector<std::pair<int, int>> all_matches;
  {
    py::gil_scoped_release release;

    int batch_size = ComputeBatchSize(device, n2, desc_dim);
    for (int offset = 0; offset < n1; offset += batch_size) {
      int count = std::min(batch_size, n1 - offset);
      auto knn = RunKNN2Kernel(device, f1_data + (size_t)offset * desc_dim,
                               count, f2_data, n2, desc_dim);
      auto batch_matches = ApplyRatioTest(knn, lowes_ratio, count, offset);
      all_matches.insert(all_matches.end(), batch_matches.begin(),
                         batch_matches.end());
    }
  }
  // --- GIL re-acquired: safe to create Python objects ---

  return PairsToArray(all_matches);
}

py::array_t<int> match_brute_force_opencl_symmetric(foundation::pyarray_f f1,
                                                    foundation::pyarray_f f2,
                                                    float lowes_ratio,
                                                    int device_idx) {
  if (!opencl_matching_available()) {
    throw std::runtime_error("OpenCL is not available");
  }

  auto& cl_ctx = dense::opencl::CLContext::Instance();
  if (device_idx < 0 || device_idx >= cl_ctx.NumDevices()) {
    throw std::runtime_error("Invalid OpenCL device index");
  }
  auto& device = cl_ctx.Device(device_idx);

  const int n1 = static_cast<int>(f1.shape(0));
  const int n2 = static_cast<int>(f2.shape(0));
  const int desc_dim = static_cast<int>(f1.shape(1));

  if (n1 == 0 || n2 == 0) {
    return py::array_t<int>(std::vector<int>{0, 2});
  }
  if (desc_dim != static_cast<int>(f2.shape(1))) {
    throw std::runtime_error(
        "Descriptor dimensions must match between f1 and f2");
  }
  if (desc_dim % 4 != 0) {
    throw std::runtime_error(
        "Descriptor dimension must be a multiple of 4 for OpenCL matching");
  }

  const float* f1_data = f1.data();
  const float* f2_data = f2.data();

  // --- Release GIL for the heavy GPU computation ---
  std::vector<std::pair<int, int>> symmetric;
  {
    py::gil_scoped_release release;

    // Forward pass: for each descriptor in f1, find NN in f2.
    int batch_size_fwd = ComputeBatchSize(device, n2, desc_dim);
    std::vector<std::pair<int, int>> matches_ij;
    for (int offset = 0; offset < n1; offset += batch_size_fwd) {
      int count = std::min(batch_size_fwd, n1 - offset);
      auto knn = RunKNN2Kernel(device, f1_data + (size_t)offset * desc_dim,
                               count, f2_data, n2, desc_dim);
      auto batch = ApplyRatioTest(knn, lowes_ratio, count, offset);
      matches_ij.insert(matches_ij.end(), batch.begin(), batch.end());
    }

    // Reverse pass: for each descriptor in f2, find NN in f1.
    int batch_size_rev = ComputeBatchSize(device, n1, desc_dim);
    struct PairHash {
      size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<long long>()(((long long)p.first << 32) | p.second);
      }
    };
    std::unordered_set<std::pair<int, int>, PairHash> reverse_set;
    for (int offset = 0; offset < n2; offset += batch_size_rev) {
      int count = std::min(batch_size_rev, n2 - offset);
      auto knn = RunKNN2Kernel(device, f2_data + (size_t)offset * desc_dim,
                               count, f1_data, n1, desc_dim);
      auto batch = ApplyRatioTest(knn, lowes_ratio, count, offset);
      for (auto& p : batch) {
        // p = (j_offset, i_idx) → store as (i_idx, j_offset)
        reverse_set.emplace(p.second, p.first);
      }
    }

    // Intersect forward and reverse matches.
    symmetric.reserve(std::min(matches_ij.size(), reverse_set.size()));
    for (auto& p : matches_ij) {
      if (reverse_set.count(p)) {
        symmetric.push_back(p);
      }
    }
  }
  // --- GIL re-acquired: safe to create Python objects ---

  return PairsToArray(symmetric);
}

py::array_t<int> match_hamming_opencl(py::array_t<uint32_t> f1,
                                      py::array_t<uint32_t> f2,
                                      float lowes_ratio, int device_idx) {
  if (!opencl_matching_available()) {
    throw std::runtime_error("OpenCL is not available");
  }

  auto& cl_ctx = dense::opencl::CLContext::Instance();
  if (device_idx < 0 || device_idx >= cl_ctx.NumDevices()) {
    throw std::runtime_error("Invalid OpenCL device index");
  }
  auto& device = cl_ctx.Device(device_idx);

  const int n1 = static_cast<int>(f1.shape(0));
  const int n2 = static_cast<int>(f2.shape(0));
  const int n_words = static_cast<int>(f1.shape(1));

  if (n1 == 0 || n2 == 0) {
    return py::array_t<int>(std::vector<int>{0, 2});
  }
  if (n_words != static_cast<int>(f2.shape(1))) {
    throw std::runtime_error(
        "Binary descriptor word counts must match between f1 and f2");
  }

  const uint32_t* f1_data = f1.data();
  const uint32_t* f2_data = f2.data();

  std::vector<std::pair<int, int>> all_matches;
  {
    py::gil_scoped_release release;

    int batch_size = ComputeHammingBatchSize(device, n2, n_words);
    for (int offset = 0; offset < n1; offset += batch_size) {
      int count = std::min(batch_size, n1 - offset);
      auto knn =
          RunHammingKNN2Kernel(device, f1_data + (size_t)offset * n_words,
                               count, f2_data, n2, n_words);
      auto batch = ApplyRatioTestHamming(knn, lowes_ratio, count, offset);
      all_matches.insert(all_matches.end(), batch.begin(), batch.end());
    }
  }

  return PairsToArray(all_matches);
}

py::array_t<int> match_hamming_opencl_symmetric(py::array_t<uint32_t> f1,
                                                py::array_t<uint32_t> f2,
                                                float lowes_ratio,
                                                int device_idx) {
  if (!opencl_matching_available()) {
    throw std::runtime_error("OpenCL is not available");
  }

  auto& cl_ctx = dense::opencl::CLContext::Instance();
  if (device_idx < 0 || device_idx >= cl_ctx.NumDevices()) {
    throw std::runtime_error("Invalid OpenCL device index");
  }
  auto& device = cl_ctx.Device(device_idx);

  const int n1 = static_cast<int>(f1.shape(0));
  const int n2 = static_cast<int>(f2.shape(0));
  const int n_words = static_cast<int>(f1.shape(1));

  if (n1 == 0 || n2 == 0) {
    return py::array_t<int>(std::vector<int>{0, 2});
  }
  if (n_words != static_cast<int>(f2.shape(1))) {
    throw std::runtime_error(
        "Binary descriptor word counts must match between f1 and f2");
  }

  const uint32_t* f1_data = f1.data();
  const uint32_t* f2_data = f2.data();

  std::vector<std::pair<int, int>> symmetric;
  {
    py::gil_scoped_release release;

    // Forward pass
    int batch_fwd = ComputeHammingBatchSize(device, n2, n_words);
    std::vector<std::pair<int, int>> matches_ij;
    for (int offset = 0; offset < n1; offset += batch_fwd) {
      int count = std::min(batch_fwd, n1 - offset);
      auto knn =
          RunHammingKNN2Kernel(device, f1_data + (size_t)offset * n_words,
                               count, f2_data, n2, n_words);
      auto batch = ApplyRatioTestHamming(knn, lowes_ratio, count, offset);
      matches_ij.insert(matches_ij.end(), batch.begin(), batch.end());
    }

    // Reverse pass
    int batch_rev = ComputeHammingBatchSize(device, n1, n_words);
    struct PairHash {
      size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<long long>()(((long long)p.first << 32) | p.second);
      }
    };
    std::unordered_set<std::pair<int, int>, PairHash> reverse_set;
    for (int offset = 0; offset < n2; offset += batch_rev) {
      int count = std::min(batch_rev, n2 - offset);
      auto knn =
          RunHammingKNN2Kernel(device, f2_data + (size_t)offset * n_words,
                               count, f1_data, n1, n_words);
      auto batch = ApplyRatioTestHamming(knn, lowes_ratio, count, offset);
      for (auto& p : batch) {
        reverse_set.emplace(p.second, p.first);
      }
    }

    // Intersect
    symmetric.reserve(std::min(matches_ij.size(), reverse_set.size()));
    for (auto& p : matches_ij) {
      if (reverse_set.count(p)) {
        symmetric.push_back(p);
      }
    }
  }

  return PairsToArray(symmetric);
}

py::list match_hamming_opencl_batch_symmetric(py::list f1_list,
                                              py::list f2_list,
                                              float lowes_ratio,
                                              int device_idx) {
  if (!opencl_matching_available()) {
    throw std::runtime_error("OpenCL is not available");
  }
  auto& cl_ctx = dense::opencl::CLContext::Instance();
  if (device_idx < 0 || device_idx >= cl_ctx.NumDevices()) {
    throw std::runtime_error("Invalid OpenCL device index");
  }
  auto& device = cl_ctx.Device(device_idx);

  int n_pairs = static_cast<int>(py::len(f1_list));
  if (n_pairs != static_cast<int>(py::len(f2_list))) {
    throw std::runtime_error("f1_list and f2_list must have the same length");
  }
  if (n_pairs == 0) {
    return py::list();
  }

  // Extract per-pair metadata.  Keep py::array_t handles alive so data
  // pointers remain valid throughout (even after GIL release — the
  // reference keeps the numpy array alive).
  struct PairArrays {
    py::array_t<uint32_t> a1, a2;
    const uint32_t* p1;
    const uint32_t* p2;
    int n1, n2;
  };
  std::vector<PairArrays> pair_info(n_pairs);

  int n_words = 0;
  for (int i = 0; i < n_pairs; ++i) {
    pair_info[i].a1 = f1_list[i].cast<py::array_t<uint32_t>>();
    pair_info[i].a2 = f2_list[i].cast<py::array_t<uint32_t>>();
    pair_info[i].n1 = static_cast<int>(pair_info[i].a1.shape(0));
    pair_info[i].n2 = static_cast<int>(pair_info[i].a2.shape(0));
    pair_info[i].p1 = pair_info[i].a1.data();
    pair_info[i].p2 = pair_info[i].a2.data();
    if (n_words == 0 && pair_info[i].n1 > 0) {
      n_words = static_cast<int>(pair_info[i].a1.shape(1));
    }
  }
  if (n_words == 0) {
    py::list result;
    for (int i = 0; i < n_pairs; ++i) {
      result.append(py::array_t<int>(std::vector<int>{0, 2}));
    }
    return result;
  }

  // Memory budget:  use 1/kMemoryReserveFraction of device global memory.
  // Per RunBatchedHammingKNN2Kernel call the GPU buffers are:
  //   buf_q      = total_queries * n_words * 4
  //   buf_r      = total_refs    * n_words * 4
  //   buf_results = total_queries * 3 * 4   (best_idx, best_dist, second_dist)
  //   buf_wg_info ≈ negligible
  // We run forward (q=f1, r=f2) then reverse (q=f2, r=f1) sequentially,
  // so peak = (sum_n1 + sum_n2) * desc_bytes + max(sum_n1, sum_n2) * 12.
  size_t avail = device.GlobalMemSize() / kMemoryReserveFraction;
  size_t desc_bytes = (size_t)n_words * sizeof(uint32_t);
  constexpr size_t kResultBytesPerQuery = 3 * sizeof(int);

  // Greedily partition pairs into chunks that fit in GPU memory.
  struct Chunk {
    int start, end;  // [start, end) into pair_info
    size_t sum_n1, sum_n2;
  };
  std::vector<Chunk> chunks;
  {
    int chunk_start = 0;
    size_t sum_n1 = 0, sum_n2 = 0;
    for (int i = 0; i < n_pairs; ++i) {
      size_t new_n1 = sum_n1 + pair_info[i].n1;
      size_t new_n2 = sum_n2 + pair_info[i].n2;
      size_t peak = (new_n1 + new_n2) * desc_bytes +
                    std::max(new_n1, new_n2) * kResultBytesPerQuery;
      if (peak > avail && i > chunk_start) {
        chunks.push_back({chunk_start, i, sum_n1, sum_n2});
        chunk_start = i;
        sum_n1 = pair_info[i].n1;
        sum_n2 = pair_info[i].n2;
      } else {
        sum_n1 = new_n1;
        sum_n2 = new_n2;
      }
    }
    if (chunk_start < n_pairs) {
      chunks.push_back({chunk_start, n_pairs, sum_n1, sum_n2});
    }
  }

  // Per-pair symmetric match results (populated under GIL release).
  std::vector<std::vector<std::pair<int, int>>> all_sym(n_pairs);

  {
    py::gil_scoped_release release;

    struct PairHash {
      size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<long long>()(((long long)p.first << 32) | p.second);
      }
    };

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      auto& chunk = chunks[ci];
      int count = chunk.end - chunk.start;
      int total_n1 = static_cast<int>(chunk.sum_n1);
      int total_n2 = static_cast<int>(chunk.sum_n2);

      // Build chunk-local flat buffers and offsets.
      std::vector<uint32_t> chunk_f1((size_t)total_n1 * n_words);
      std::vector<uint32_t> chunk_f2((size_t)total_n2 * n_words);
      std::vector<PairOffsets> fwd_offsets(count);

      int off1 = 0, off2 = 0;
      for (int j = 0; j < count; ++j) {
        int i = chunk.start + j;
        fwd_offsets[j].f1_offset = off1;
        fwd_offsets[j].n1 = pair_info[i].n1;
        fwd_offsets[j].f2_offset = off2;
        fwd_offsets[j].n2 = pair_info[i].n2;
        if (pair_info[i].n1 > 0) {
          std::memcpy(chunk_f1.data() + (size_t)off1 * n_words, pair_info[i].p1,
                      (size_t)pair_info[i].n1 * n_words * sizeof(uint32_t));
        }
        if (pair_info[i].n2 > 0) {
          std::memcpy(chunk_f2.data() + (size_t)off2 * n_words, pair_info[i].p2,
                      (size_t)pair_info[i].n2 * n_words * sizeof(uint32_t));
        }
        off1 += pair_info[i].n1;
        off2 += pair_info[i].n2;
      }

      // Reverse offsets: query=f2, ref=f1.
      std::vector<PairOffsets> rev_offsets(count);
      for (int j = 0; j < count; ++j) {
        rev_offsets[j].f1_offset = fwd_offsets[j].f2_offset;
        rev_offsets[j].n1 = fwd_offsets[j].n2;
        rev_offsets[j].f2_offset = fwd_offsets[j].f1_offset;
        rev_offsets[j].n2 = fwd_offsets[j].n1;
      }

      // Forward: queries=f1, refs=f2.
      auto fwd = RunBatchedHammingKNN2Kernel(device, chunk_f1.data(), total_n1,
                                             chunk_f2.data(), total_n2,
                                             fwd_offsets, n_words);

      // Reverse: queries=f2, refs=f1.
      auto rev = RunBatchedHammingKNN2Kernel(device, chunk_f2.data(), total_n2,
                                             chunk_f1.data(), total_n1,
                                             rev_offsets, n_words);

      // Per-pair: ratio test + symmetric intersection.
      for (int j = 0; j < count; ++j) {
        int i = chunk.start + j;
        if (fwd_offsets[j].n1 == 0 || fwd_offsets[j].n2 == 0) {
          continue;
        }

        auto fwd_m = ApplyRatioTestHamming(fwd, lowes_ratio, fwd_offsets[j].n1,
                                           0, fwd_offsets[j].f1_offset);
        auto rev_m = ApplyRatioTestHamming(rev, lowes_ratio, rev_offsets[j].n1,
                                           0, rev_offsets[j].f1_offset);

        std::unordered_set<std::pair<int, int>, PairHash> rev_set;
        for (auto& p : rev_m) {
          rev_set.emplace(p.second, p.first);
        }

        auto& sym = all_sym[i];
        sym.reserve(std::min(fwd_m.size(), rev_set.size()));
        for (auto& p : fwd_m) {
          if (rev_set.count(p)) {
            sym.push_back(p);
          }
        }
      }
    }
  }
  // GIL re-acquired.

  py::list result;
  for (int i = 0; i < n_pairs; ++i) {
    result.append(PairsToArray(all_sym[i]));
  }
  return result;
}

}  // namespace features

#else  // !OPENSFM_HAVE_OPENCL

namespace features {

bool opencl_matching_available() { return false; }

int opencl_num_devices() { return 0; }

py::array_t<int> match_brute_force_opencl(foundation::pyarray_f /*f1*/,
                                          foundation::pyarray_f /*f2*/,
                                          float /*lowes_ratio*/,
                                          int /*device_idx*/) {
  throw std::runtime_error(
      "OpenCL matching is not available (built without OpenCL support)");
}

py::array_t<int> match_brute_force_opencl_symmetric(
    foundation::pyarray_f /*f1*/, foundation::pyarray_f /*f2*/,
    float /*lowes_ratio*/, int /*device_idx*/) {
  throw std::runtime_error(
      "OpenCL matching is not available (built without OpenCL support)");
}

py::array_t<int> match_hamming_opencl(py::array_t<uint32_t> /*f1*/,
                                      py::array_t<uint32_t> /*f2*/,
                                      float /*lowes_ratio*/,
                                      int /*device_idx*/) {
  throw std::runtime_error(
      "OpenCL matching is not available (built without OpenCL support)");
}

py::array_t<int> match_hamming_opencl_symmetric(py::array_t<uint32_t> /*f1*/,
                                                py::array_t<uint32_t> /*f2*/,
                                                float /*lowes_ratio*/,
                                                int /*device_idx*/) {
  throw std::runtime_error(
      "OpenCL matching is not available (built without OpenCL support)");
}

py::list match_hamming_opencl_batch_symmetric(py::list /*f1_list*/,
                                              py::list /*f2_list*/,
                                              float /*lowes_ratio*/,
                                              int /*device_idx*/) {
  throw std::runtime_error(
      "OpenCL matching is not available (built without OpenCL support)");
}

}  // namespace features

#endif  // OPENSFM_HAVE_OPENCL
