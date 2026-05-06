#pragma once

#include <foundation/python_types.h>

// Conditional compilation: if OpenCL is not available, provide a stub
// that raises a clear error at runtime.
#ifdef OPENSFM_HAVE_OPENCL

#include <dense/cleaner.h>
#include <dense/cluster.h>
#include <dense/dsm_opencl_kernels.h>
#include <pybind11/eigen.h>

namespace dense {

// Row-major (N, 3) double matrix for SfM points (matches numpy C-order).
using PointsMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

class DepthmapClusterEstimatorWrapper {
 public:
  void SetMaxIterations(int n) { params_.max_iterations = n; }
  void SetPatchSize(int size) { params_.patch_size = size; }
  void SetMaxImageSize(int size) { params_.max_image_size = size; }
  void SetHierarchyLevels(int levels) { params_.hierarchy_levels = levels; }
  void SetSigmaSpatial(float v) { params_.sigma_spatial = v; }
  void SetSigmaColor(float v) { params_.sigma_color = v; }
  void SetTopK(int k) { params_.top_k = k; }
  void SetCensusWeight(float w) { params_.census_weight = w; }
  void SetSmoothWeight(float w) { params_.smooth_weight = w; }
  void SetGeomConsistencyWeight(float w) {
    cluster_.SetGeomConsistencyWeight(w);
  }
  void SetDevice(int idx) { cluster_.SetDevice(idx); }

  int BeginRefView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                   Eigen::Ref<const ImageU8> image, float depth_min,
                   float depth_max) {
    return cluster_.BeginRefView(K, R, t, image, depth_min, depth_max);
  }

  void AddSourceView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                     Eigen::Ref<const ImageU8> image) {
    cluster_.AddSourceView(K, R, t, image);
  }

  void SetSfMPoints(Eigen::Ref<const PointsMatrix> points) {
    cluster_.SetSfMPoints(points);
  }

  void AddGeomLink(int ref_idx, int source_view_idx, int from_ref_idx) {
    cluster_.AddGeomLink(ref_idx, source_view_idx, from_ref_idx);
  }

  py::list Run() {
    cluster_.SetParams(params_);
    std::vector<DepthmapResult> results;
    {
      py::gil_scoped_release release;
      cluster_.Run(&results);
    }
    py::list retn;
    for (auto& r : results) {
      const int h = static_cast<int>(r.depth.rows());
      const int w = static_cast<int>(r.depth.cols());
      py::list entry;
      entry.append(foundation::py_array_from_data(r.depth.data(), h, w));
      entry.append(foundation::py_array_from_data(r.normal.data(), h, w, 3));
      entry.append(foundation::py_array_from_data(r.cost.data(), h, w));
      entry.append(foundation::py_array_from_data(r.confidence.data(), h, w));
      retn.append(entry);
    }
    return retn;
  }

  void Clear() { cluster_.Clear(); }

  static bool IsAvailable() {
    return opencl::CLContext::Instance().IsAvailable();
  }

  static size_t GpuMemoryBytes() {
    return DepthmapClusterEstimator::GpuMemoryBytes();
  }

  static int NumDevices() { return opencl::CLContext::Instance().NumDevices(); }

  static std::string DeviceName(int idx) {
    return opencl::CLContext::Instance().Device(idx).name();
  }

  static size_t DeviceMemoryBytes(int idx) {
    return opencl::CLContext::Instance().Device(idx).GlobalMemSize();
  }

  static bool DeviceIsGPU(int idx) {
    return opencl::CLContext::Instance().Device(idx).IsGPU();
  }

  static size_t DeviceAvailableMemory(int idx) {
    return opencl::CLContext::Instance().Device(idx).AvailableMemory();
  }

  static void ReserveDeviceMemory(int idx, size_t bytes) {
    opencl::CLContext::Instance().Device(idx).ReserveMemory(bytes);
  }

  static void ReleaseDeviceMemory(int idx, size_t bytes) {
    opencl::CLContext::Instance().Device(idx).ReleaseMemory(bytes);
  }

 private:
  DepthmapClusterEstimator cluster_;
  DepthmapParams params_;
};

class DepthmapCleanerWrapper {
 public:
  void SetSameDepthThreshold(float t) { cleaner_.SetSameDepthThreshold(t); }
  void SetMinConsistentViews(int n) { cleaner_.SetMinConsistentViews(n); }
  void SetCarvingThreshold(float t) { cleaner_.SetCarvingThreshold(t); }
  void SetMaxCarvedViews(int n) { cleaner_.SetMaxCarvedViews(n); }
  void SetDevice(int idx) { cleaner_.SetDevice(idx); }

  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               Eigen::Ref<const ImageF> depth) {
    cleaner_.AddView(K, R, t, depth);
  }

  void Clear() { cleaner_.Clear(); }

  foundation::pyarray_f Clean(int ref_idx,
                              const py::array_t<int>& neighbor_ids) {
    std::vector<int> nbrs(neighbor_ids.data(),
                          neighbor_ids.data() + neighbor_ids.size());
    cv::Mat cleaned;
    {
      py::gil_scoped_release release;
      cleaned = cleaner_.Clean(ref_idx, nbrs);
    }
    return foundation::py_array_from_data(cleaned.ptr<float>(0), cleaned.rows,
                                          cleaned.cols);
  }

  static bool IsAvailable() {
    return opencl::CLContext::Instance().IsAvailable();
  }

 private:
  GPUDepthmapCleaner cleaner_;
};

}  // namespace dense

#else  // !OPENSFM_HAVE_OPENCL

#include <dense/fuser.h>
#include <pybind11/eigen.h>

namespace dense {

// Same type alias as in the OpenCL-enabled section above.
using PointsMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

/// Stub DepthmapClusterEstimator when OpenCL is not compiled in.
class DepthmapClusterEstimatorWrapper {
 public:
  void SetMaxIterations(int) {}
  void SetPatchSize(int) {}
  void SetMaxImageSize(int) {}
  void SetHierarchyLevels(int) {}
  void SetSigmaSpatial(float) {}
  void SetSigmaColor(float) {}
  void SetTopK(int) {}
  void SetCensusWeight(float) {}
  void SetSmoothWeight(float) {}
  void SetGeomConsistencyWeight(float) {}
  void SetDevice(int) {}
  int BeginRefView(const Mat3d&, const Mat3d&, const Vec3d&,
                   Eigen::Ref<const ImageU8>, float, float) {
    throw std::runtime_error("PatchMatch: OpenCL not available");
  }
  void AddSourceView(const Mat3d&, const Mat3d&, const Vec3d&,
                     Eigen::Ref<const ImageU8>) {
    throw std::runtime_error("PatchMatch: OpenCL not available");
  }
  void SetSfMPoints(Eigen::Ref<const PointsMatrix>) {}
  void AddGeomLink(int, int, int) {}
  py::list Run() {
    throw std::runtime_error("PatchMatch: OpenCL not available");
  }
  void Clear() {}
  static bool IsAvailable() { return false; }
  static size_t GpuMemoryBytes() { return 0; }
  static int NumDevices() { return 0; }
  static std::string DeviceName(int) { return ""; }
  static size_t DeviceMemoryBytes(int) { return 0; }
  static bool DeviceIsGPU(int) { return false; }
  static size_t DeviceAvailableMemory(int) { return 0; }
  static void ReserveDeviceMemory(int, size_t) {}
  static void ReleaseDeviceMemory(int, size_t) {}
};

/// Stub GPUDepthmapCleaner when OpenCL is not compiled in.
class DepthmapCleanerWrapper {
 public:
  void SetSameDepthThreshold(float) {}
  void SetMinConsistentViews(int) {}
  void SetCarvingThreshold(float) {}
  void SetMaxCarvedViews(int) {}
  void SetDevice(int) {}
  void AddView(const Mat3d&, const Mat3d&, const Vec3d&,
               Eigen::Ref<const ImageF>) {
    throw std::runtime_error("GPUDepthmapCleaner: OpenCL not available");
  }
  foundation::pyarray_f Clean(int, const py::array_t<int>&) {
    throw std::runtime_error("GPUDepthmapCleaner: OpenCL not available");
  }
  void Clear() {}
  static bool IsAvailable() { return false; }
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL

#include <dense/fuser.h>
#include <dense/svo_fuser.h>
#include <pybind11/eigen.h>

namespace dense {

class DepthmapFuserWrapper {
 public:
  void SetMinNumConsistent(int n) { df_.SetMinNumConsistent(n); }
  void SetMaxReprojError(float px) { df_.SetMaxReprojError(px); }
  void SetMaxDepthError(float ratio) { df_.SetMaxDepthError(ratio); }
  void SetMaxNormalError(float degrees) { df_.SetMaxNormalError(degrees); }
  void SetBorderMargin(int px) { df_.SetBorderMargin(px); }
  void SetNumThreads(int n) { df_.SetNumThreads(n); }
  void SetSORParams(int knn, float stddev_factor) {
    df_.SetSORParams(knn, stddev_factor);
  }
  void SetBehindDepthFactor(float f) { df_.SetBehindDepthFactor(f); }

  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               Eigen::Ref<const ImageF> depth,
               const py::array_t<float, py::array::c_style>& normal,
               const py::array_t<uint8_t, py::array::c_style>& color,
               Eigen::Ref<const ImageU8> mask,
               const py::array_t<int>& neighbor_ids, bool primary = true) {
    const int h = static_cast<int>(depth.rows());
    const int w = static_cast<int>(depth.cols());
    if (normal.ndim() != 3 || normal.shape(0) != h || normal.shape(1) != w) {
      throw std::invalid_argument(
          "depth and normal must have matching shapes.");
    }
    if (color.ndim() != 3 || color.shape(0) != h || color.shape(1) != w) {
      throw std::invalid_argument("depth and color must have matching shapes.");
    }
    Eigen::Map<const PixelData3f> n_e(normal.data(), 3, h * w);
    Eigen::Map<const PixelData3u8> c_e(color.data(), 3, h * w);
    std::vector<int> nbrs(neighbor_ids.data(),
                          neighbor_ids.data() + neighbor_ids.size());
    df_.AddView(K, R, t, depth, n_e, c_e, mask, nbrs, primary);
  }

  py::list Fuse() {
    std::vector<Vec3f> points;
    std::vector<Vec3f> normals;
    std::vector<Vec3<uint8_t>> colors;

    {
      py::gil_scoped_release release;
      df_.Fuse(&points, &normals, &colors);
    }

    const int n = static_cast<int>(points.size());
    // Vec3f is 3 contiguous floats — reinterpret as (N,3) flat array.
    py::list retn;
    retn.append(foundation::py_array_from_data(points.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(normals.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(colors.data()->data(), n, 3));
    return retn;
  }

 private:
  DepthmapFuser df_;
};

// ---- SVOFuser wrapper ----

class SVOFuserWrapper {
 public:
  void SetVoxelSize(float size) { sf_.SetVoxelSize(size); }
  void SetTruncFactor(float factor) { sf_.SetTruncFactor(factor); }
  void SetMinWeight(float w) { sf_.SetMinWeight(w); }
  void SetDevice(int device_idx) { sf_.SetDevice(device_idx); }
  void SetBBox(const Vec3f& min_world, const Vec3f& max_world) {
    sf_.SetBBox(min_world, max_world);
  }
  static bool IsGPUAvailable() { return SVOFuser::IsGPUAvailable(); }

  uint32_t CountVoxels() {
    py::gil_scoped_release release;
    return sf_.CountVoxels();
  }

  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               Eigen::Ref<const ImageF> depth,
               const py::array_t<float, py::array::c_style>& normal,
               const py::array_t<uint8_t, py::array::c_style>& color,
               Eigen::Ref<const ImageU8> mask,
               const py::object& confidence = py::none(),
               const std::string& name = "") {
    const int h = static_cast<int>(depth.rows());
    const int w = static_cast<int>(depth.cols());
    if (normal.ndim() != 3 || normal.shape(0) != h || normal.shape(1) != w) {
      throw std::invalid_argument(
          "depth and normal must have matching shapes.");
    }
    if (color.ndim() != 3 || color.shape(0) != h || color.shape(1) != w) {
      throw std::invalid_argument("depth and color must have matching shapes.");
    }
    Eigen::Map<const PixelData3f> n_e(normal.data(), 3, h * w);
    Eigen::Map<const PixelData3u8> c_e(color.data(), 3, h * w);

    ImageF weight;
    if (!confidence.is_none()) {
      auto conf_arr = confidence.cast<py::array_t<float, py::array::c_style>>();
      if (conf_arr.ndim() == 2 && conf_arr.shape(0) == h &&
          conf_arr.shape(1) == w) {
        weight = Eigen::Map<const ImageF>(conf_arr.data(), h, w);
      }
    }
    sf_.AddView(K, R, t, depth, n_e, c_e, mask, weight, name);
  }

  py::list Fuse() {
    std::vector<Vec3f> points;
    std::vector<Vec3f> normals;
    std::vector<Vec3<uint8_t>> colors;

    {
      py::gil_scoped_release release;
      sf_.Fuse(&points, &normals, &colors);
    }

    const int n = static_cast<int>(points.size());
    py::list retn;
    retn.append(foundation::py_array_from_data(points.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(normals.data()->data(), n, 3));
    retn.append(foundation::py_array_from_data(colors.data()->data(), n, 3));
    return retn;
  }

 private:
  SVOFuser sf_;
};

// ---- DSM rasterizer wrapper (two-pass depthmap scatter + bilateral) ----

#ifdef OPENSFM_HAVE_OPENCL

class DSMRasterizerWrapper {
 public:
  void SetGSD(float gsd) { gsd_ = gsd; }
  void SetBBox(const Eigen::Vector2f& min_xy, const Eigen::Vector2f& max_xy) {
    origin_x_ = min_xy.x();
    origin_y_ = min_xy.y();
    max_x_ = max_xy.x();
    max_y_ = max_xy.y();
    has_bbox_ = true;
  }
  void SetDevice(int idx) { device_idx_ = idx; }
  void SetOutlierThreshold(float meters) { outlier_threshold_ = meters; }
  void SetMinCount(int n) { min_count_ = n; }
  void SetZBias(float alpha) { z_bias_ = alpha; }
  void SetBilateral(bool enabled, int radius, float range_sigma) {
    bilateral_enabled_ = enabled;
    bilateral_radius_ = radius;
    bilateral_range_ = range_sigma;
  }

  // Allocate GPU grids and prepare for pass-1 scatter.
  void Begin() {
    if (!has_bbox_ || gsd_ <= 0.0f) {
      throw std::runtime_error(
          "DSMRasterizer: must set GSD and bbox before Begin()");
    }

    grid_w_ = static_cast<int>(std::ceil((max_x_ - origin_x_) / gsd_));
    grid_h_ = static_cast<int>(std::ceil((max_y_ - origin_y_) / gsd_));
    if (grid_w_ <= 0 || grid_h_ <= 0) {
      throw std::runtime_error("DSMRasterizer: degenerate grid size");
    }

    const size_t ncells = static_cast<size_t>(grid_h_) * grid_w_;

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    prog_ = dev.GetOrBuildProgram("dsm_rasterizer_v2", kDSMKernelSource);

    auto& ctx = dev.context();
    const size_t int_bytes = ncells * sizeof(int32_t);
    const size_t float_bytes = ncells * sizeof(float);

    cl_sum_zw_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, int_bytes);
    cl_sum_w_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, int_bytes);
    cl_count_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, int_bytes);
    cl_mean_z_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, float_bytes);
    cl_out_a_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, float_bytes);
    cl_out_b_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, float_bytes);

    ClearGrids();
    in_pass2_ = false;
  }

  // Scatter one depthmap view.  In pass 1 calls dsm_backproject_scatter;
  // after BeginPass2() calls dsm_reject_scatter.
  void Scatter(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               Eigen::Ref<const ImageF> depth,
               const py::array_t<float, py::array::c_style>& normal,
               Eigen::Ref<const ImageF> confidence) {
    const int h = static_cast<int>(depth.rows());
    const int w = static_cast<int>(depth.cols());
    const int npx = h * w;

    if (normal.ndim() != 3 || normal.shape(0) != h || normal.shape(1) != w ||
        normal.shape(2) != 3) {
      throw std::invalid_argument("normal must be (H, W, 3) float32");
    }

    // Pack camera params: K_inv(9) + R_inv(9) + t(3) = 21 floats row-major
    Eigen::Matrix3f Kinvf = K.cast<float>().inverse();
    Eigen::Matrix3f Rinvf = R.cast<float>().transpose();
    Eigen::Vector3f tf = t.cast<float>();

    float cam[21];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        cam[i * 3 + j] = Kinvf(i, j);
        cam[9 + i * 3 + j] = Rinvf(i, j);
      }
    }
    cam[18] = tf.x();
    cam[19] = tf.y();
    cam[20] = tf.z();

    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& ctx = dev.context();
    auto& q = dev.queue();

    // Upload view data
    cl::Buffer cl_depth(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        npx * sizeof(float), const_cast<float*>(depth.data()));
    cl::Buffer cl_normal(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         npx * 3 * sizeof(float),
                         const_cast<float*>(normal.data()));
    cl::Buffer cl_conf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       npx * sizeof(float),
                       const_cast<float*>(confidence.data()));
    cl::Buffer cl_cam(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      21 * sizeof(float), cam);

    const float inv_gsd = 1.0f / gsd_;
    const size_t global = ((static_cast<size_t>(npx) + 255) / 256) * 256;

    if (!in_pass2_) {
      cl::Kernel k(prog_, "dsm_backproject_scatter");
      k.setArg(0, cl_depth);
      k.setArg(1, cl_normal);
      k.setArg(2, cl_conf);
      k.setArg(3, cl_cam);
      k.setArg(4, cl_sum_zw_);
      k.setArg(5, cl_sum_w_);
      k.setArg(6, cl_count_);
      k.setArg(7, origin_x_);
      k.setArg(8, origin_y_);
      k.setArg(9, inv_gsd);
      k.setArg(10, grid_w_);
      k.setArg(11, grid_h_);
      k.setArg(12, w);
      k.setArg(13, h);

      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
      q.finish();
    } else {
      cl::Kernel k(prog_, "dsm_reject_scatter");
      k.setArg(0, cl_depth);
      k.setArg(1, cl_normal);
      k.setArg(2, cl_conf);
      k.setArg(3, cl_cam);
      k.setArg(4, cl_mean_z_);
      k.setArg(5, cl_sum_zw_);
      k.setArg(6, cl_sum_w_);
      k.setArg(7, cl_count_);
      k.setArg(8, origin_x_);
      k.setArg(9, origin_y_);
      k.setArg(10, inv_gsd);
      k.setArg(11, grid_w_);
      k.setArg(12, grid_h_);
      k.setArg(13, w);
      k.setArg(14, h);
      k.setArg(15, outlier_threshold_);
      k.setArg(16, z_bias_);

      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
      q.finish();
    }
  }

  // Finalize pass-1 mean, clear accumulators, switch to pass-2 mode.
  void BeginPass2() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const int ncells = grid_h_ * grid_w_;

    // Finalize pass 1 → mean_z
    cl::Kernel k_fin(prog_, "dsm_finalize");
    k_fin.setArg(0, cl_sum_zw_);
    k_fin.setArg(1, cl_sum_w_);
    k_fin.setArg(2, cl_count_);
    k_fin.setArg(3, cl_mean_z_);
    k_fin.setArg(4, ncells);
    // Pass 1 finalize uses min_count=1, min_weight=1 (lenient) so that
    // the mean_z reference grid is as full as possible for outlier rejection.
    k_fin.setArg(5, 1);
    k_fin.setArg(6, 1);
    const size_t global = ((static_cast<size_t>(ncells) + 255) / 256) * 256;
    {
      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k_fin, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
      q.finish();
    }

    ClearGrids();
    in_pass2_ = true;
  }

  // Finalize pass 2, optional bilateral filter, download result.
  foundation::pyarray_f Finish() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const int ncells = grid_h_ * grid_w_;
    const size_t global1d = ((static_cast<size_t>(ncells) + 255) / 256) * 256;

    // Finalize pass 2 (biased weighted mean) → cl_out_a_
    cl::Kernel k_fin(prog_, "dsm_finalize");
    k_fin.setArg(0, cl_sum_zw_);
    k_fin.setArg(1, cl_sum_w_);
    k_fin.setArg(2, cl_count_);
    k_fin.setArg(3, cl_out_a_);
    k_fin.setArg(4, ncells);
    k_fin.setArg(5, min_count_);
    k_fin.setArg(6, 1);  // min_weight = 1 (guard division by zero)
    {
      py::gil_scoped_release release;
      q.enqueueNDRangeKernel(k_fin, cl::NullRange, cl::NDRange(global1d),
                             cl::NDRange(256));
      q.finish();
    }

    // Bilateral filter: cl_out_a_ → cl_out_b_
    cl::Buffer* result_buf = &cl_out_a_;
    if (bilateral_enabled_ && bilateral_radius_ > 0) {
      cl::Kernel k_bi(prog_, "dsm_bilateral");
      k_bi.setArg(0, cl_out_a_);
      k_bi.setArg(1, cl_out_b_);
      k_bi.setArg(2, grid_w_);
      k_bi.setArg(3, grid_h_);
      k_bi.setArg(4, bilateral_radius_);
      const float sigma_s = static_cast<float>(bilateral_radius_) / 2.0f;
      const float inv_2ss = 1.0f / (2.0f * sigma_s * sigma_s);
      const float inv_2sr = 1.0f / (2.0f * bilateral_range_ * bilateral_range_);
      k_bi.setArg(5, inv_2ss);
      k_bi.setArg(6, inv_2sr);

      const size_t gx = ((static_cast<size_t>(grid_w_) + 15) / 16) * 16;
      const size_t gy = ((static_cast<size_t>(grid_h_) + 15) / 16) * 16;
      {
        py::gil_scoped_release release;
        q.enqueueNDRangeKernel(k_bi, cl::NullRange, cl::NDRange(gx, gy),
                               cl::NDRange(16, 16));
        q.finish();
      }
      result_buf = &cl_out_b_;
    }

    // Download
    const size_t float_bytes = static_cast<size_t>(ncells) * sizeof(float);
    std::vector<float> host_grid(ncells);
    q.enqueueReadBuffer(*result_buf, CL_TRUE, 0, float_bytes, host_grid.data());

    return foundation::py_array_from_data(host_grid.data(), grid_h_, grid_w_);
  }

  static bool IsAvailable() {
    return opencl::CLContext::Instance().IsAvailable();
  }

  // ------------------------------------------------------------------
  // CPU percentile path (debug / reference).
  //
  // After BeginPass2(), call ScatterCPU() for each view instead of
  // Scatter().  Then call FinishPercentile(0.9) instead of Finish().
  // ------------------------------------------------------------------

  void ScatterCPU(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                  Eigen::Ref<const ImageF> depth,
                  const py::array_t<float, py::array::c_style>& normal,
                  Eigen::Ref<const ImageF> confidence) {
    const int h = static_cast<int>(depth.rows());
    const int w = static_cast<int>(depth.cols());

    if (normal.ndim() != 3 || normal.shape(0) != h || normal.shape(1) != w ||
        normal.shape(2) != 3) {
      throw std::invalid_argument("normal must be (H, W, 3) float32");
    }

    const size_t ncells = static_cast<size_t>(grid_h_) * grid_w_;
    if (cell_z_.empty()) {
      cell_z_.resize(ncells);
    }

    // Download mean_z from GPU on first CPU scatter call (if no external
    // reference was provided via SetReferenceZ).
    if (host_mean_z_.empty()) {
      host_mean_z_.resize(ncells);
      auto& dev = opencl::CLContext::Instance().Device(device_idx_);
      dev.queue().enqueueReadBuffer(
          cl_mean_z_, CL_TRUE, 0, ncells * sizeof(float), host_mean_z_.data());
    }

    Eigen::Matrix3f Kinvf = K.cast<float>().inverse();
    Eigen::Matrix3f Rinvf = R.cast<float>().transpose();
    Eigen::Vector3f tf = t.cast<float>();

    const float* dp = depth.data();
    const float* np_data = normal.data();
    const float inv_gsd = 1.0f / gsd_;

    py::gil_scoped_release release;

    for (int v = 0; v < h; ++v) {
      for (int u = 0; u < w; ++u) {
        const int idx = v * w + u;
        const float d = dp[idx];
        if (d <= 0.0f) {
          continue;
        }

        const float fu = static_cast<float>(u);
        const float fv = static_cast<float>(v);
        Eigen::Vector3f cam_pt;
        cam_pt.x() = (Kinvf(0, 0) * fu + Kinvf(0, 1) * fv + Kinvf(0, 2)) * d;
        cam_pt.y() = (Kinvf(1, 0) * fu + Kinvf(1, 1) * fv + Kinvf(1, 2)) * d;
        cam_pt.z() = (Kinvf(2, 0) * fu + Kinvf(2, 1) * fv + Kinvf(2, 2)) * d;

        Eigen::Vector3f world = Rinvf * (cam_pt - tf);

        const int gx =
            static_cast<int>(std::floor((world.x() - origin_x_) * inv_gsd));
        const int gy =
            static_cast<int>(std::floor((world.y() - origin_y_) * inv_gsd));
        if (gx < 0 || gx >= grid_w_ || gy < 0 || gy >= grid_h_) {
          continue;
        }

        const int cell = gy * grid_w_ + gx;
        const float mz = host_mean_z_[cell];
        if (std::isnan(mz) || (mz - world.z()) > outlier_threshold_) {
          continue;
        }

        const int nbase = idx * 3;
        Eigen::Vector3f cn(np_data[nbase], np_data[nbase + 1],
                           np_data[nbase + 2]);
        const float wnz = Rinvf.row(2).dot(cn);
        if (wnz < 0.3f) {
          continue;
        }

        cell_z_[cell].push_back(world.z());
      }
    }
  }

  foundation::pyarray_f FinishPercentile(float percentile) {
    const size_t ncells = static_cast<size_t>(grid_h_) * grid_w_;
    std::vector<float> result(ncells, std::numeric_limits<float>::quiet_NaN());

    {
      py::gil_scoped_release release;
      for (size_t i = 0; i < ncells; ++i) {
        auto& zs = cell_z_[i];
        if (static_cast<int>(zs.size()) < min_count_) {
          continue;
        }
        std::sort(zs.begin(), zs.end());
        const float rank = percentile * static_cast<float>(zs.size() - 1);
        const int lo = static_cast<int>(rank);
        const int hi = std::min(lo + 1, static_cast<int>(zs.size()) - 1);
        const float frac = rank - static_cast<float>(lo);
        result[i] = zs[lo] * (1.0f - frac) + zs[hi] * frac;
      }
    }

    // CPU bilateral filter for the debug path.
    if (bilateral_enabled_ && bilateral_radius_ > 0) {
      std::vector<float> filtered(ncells,
                                  std::numeric_limits<float>::quiet_NaN());
      const float sigma_s = static_cast<float>(bilateral_radius_) / 2.0f;
      const float inv_2ss = 1.0f / (2.0f * sigma_s * sigma_s);
      const float inv_2sr = 1.0f / (2.0f * bilateral_range_ * bilateral_range_);

      py::gil_scoped_release release;
      for (int y = 0; y < grid_h_; ++y) {
        for (int x = 0; x < grid_w_; ++x) {
          const float center = result[y * grid_w_ + x];
          if (std::isnan(center)) {
            continue;
          }
          float sum_val = 0.0f, sum_wt = 0.0f;
          for (int dy = -bilateral_radius_; dy <= bilateral_radius_; ++dy) {
            const int ny = y + dy;
            if (ny < 0 || ny >= grid_h_) {
              continue;
            }
            for (int dx = -bilateral_radius_; dx <= bilateral_radius_; ++dx) {
              const int nx = x + dx;
              if (nx < 0 || nx >= grid_w_) {
                continue;
              }
              const float nval = result[ny * grid_w_ + nx];
              if (std::isnan(nval)) {
                continue;
              }
              const float sd = static_cast<float>(dx * dx + dy * dy);
              const float rd = center - nval;
              const float wt = std::exp(-sd * inv_2ss - rd * rd * inv_2sr);
              sum_val += nval * wt;
              sum_wt += wt;
            }
          }
          filtered[y * grid_w_ + x] =
              (sum_wt > 0.0f) ? (sum_val / sum_wt)
                              : std::numeric_limits<float>::quiet_NaN();
        }
      }
      result = std::move(filtered);
    }

    cell_z_.clear();
    cell_z_.shrink_to_fit();
    host_mean_z_.clear();
    host_mean_z_.shrink_to_fit();

    return foundation::py_array_from_data(result.data(), grid_h_, grid_w_);
  }

  // Override the outlier-rejection reference grid with an external one
  // (e.g., upsampled coarse P90).  Must match grid_h_ × grid_w_.
  void SetReferenceZ(Eigen::Ref<const ImageF> ref_z) {
    if (ref_z.rows() != grid_h_ || ref_z.cols() != grid_w_) {
      throw std::invalid_argument("SetReferenceZ: shape must match grid");
    }
    const size_t ncells = static_cast<size_t>(grid_h_) * grid_w_;
    host_mean_z_.resize(ncells);
    std::memcpy(host_mean_z_.data(), ref_z.data(), ncells * sizeof(float));
  }

 private:
  void ClearGrids() {
    auto& dev = opencl::CLContext::Instance().Device(device_idx_);
    auto& q = dev.queue();
    const int ncells = grid_h_ * grid_w_;
    cl::Kernel k(prog_, "dsm_clear_grid");
    k.setArg(0, cl_sum_zw_);
    k.setArg(1, cl_sum_w_);
    k.setArg(2, cl_count_);
    k.setArg(3, ncells);
    const size_t global = ((static_cast<size_t>(ncells) + 255) / 256) * 256;
    q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global),
                           cl::NDRange(256));
    q.finish();
  }

  float gsd_ = 0.0f;
  float origin_x_ = 0.0f;
  float origin_y_ = 0.0f;
  float max_x_ = 0.0f;
  float max_y_ = 0.0f;
  bool has_bbox_ = false;
  int device_idx_ = 0;
  float outlier_threshold_ = 1.0f;
  float z_bias_ = 2.5f;
  int min_count_ = 3;
  int bilateral_radius_ = 2;
  float bilateral_range_ = 0.3f;
  bool bilateral_enabled_ = true;

  int grid_w_ = 0;
  int grid_h_ = 0;
  bool in_pass2_ = false;
  cl::Program prog_;
  cl::Buffer cl_sum_zw_;
  cl::Buffer cl_sum_w_;
  cl::Buffer cl_count_;
  cl::Buffer cl_mean_z_;
  cl::Buffer cl_out_a_;
  cl::Buffer cl_out_b_;

  // CPU percentile path storage.
  std::vector<std::vector<float>> cell_z_;
  std::vector<float> host_mean_z_;
};

#else  // !OPENSFM_HAVE_OPENCL

class DSMRasterizerWrapper {
 public:
  void SetGSD(float) {}
  void SetBBox(const Eigen::Vector2f&, const Eigen::Vector2f&) {}
  void SetDevice(int) {}
  void SetOutlierThreshold(float) {}
  void SetMinCount(int) {}
  void SetZBias(float) {}
  void SetBilateral(bool, int, float) {}
  void Begin() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  void Scatter(const Mat3d&, const Mat3d&, const Vec3d&,
               Eigen::Ref<const ImageF>,
               const py::array_t<float, py::array::c_style>&,
               Eigen::Ref<const ImageF>) {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  void BeginPass2() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  foundation::pyarray_f Finish() {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  void ScatterCPU(const Mat3d&, const Mat3d&, const Vec3d&,
                  Eigen::Ref<const ImageF>,
                  const py::array_t<float, py::array::c_style>&,
                  Eigen::Ref<const ImageF>) {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  foundation::pyarray_f FinishPercentile(float) {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  void SetReferenceZ(Eigen::Ref<const ImageF>) {
    throw std::runtime_error("DSMRasterizer: OpenCL not available");
  }
  static bool IsAvailable() { return false; }
};

#endif  // OPENSFM_HAVE_OPENCL

}  // namespace dense
