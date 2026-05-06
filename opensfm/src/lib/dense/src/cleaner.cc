#include <dense/cleaner.h>

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/opencl_kernels.h>

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace dense {

// =====================================================================
// GPUDepthmapCleaner implementation.
// =====================================================================

GPUDepthmapCleaner::GPUDepthmapCleaner() = default;

void GPUDepthmapCleaner::SetSameDepthThreshold(float t) {
  same_depth_threshold_ = t;
}

void GPUDepthmapCleaner::SetMinConsistentViews(int n) {
  min_consistent_views_ = n;
}

void GPUDepthmapCleaner::SetCarvingThreshold(float t) {
  carving_threshold_ = std::max(0.0f, t);
}

void GPUDepthmapCleaner::SetMaxCarvedViews(int n) {
  max_carved_views_ = std::max(0, n);
}

void GPUDepthmapCleaner::SetDevice(int device_idx) { device_idx_ = device_idx; }

int GPUDepthmapCleaner::AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                                const ImageF& depth) {
  ViewEntry v;
  v.depth =
      cv::Mat(static_cast<int>(depth.rows()), static_cast<int>(depth.cols()),
              CV_32F, const_cast<float*>(depth.data()))
          .clone();
  v.K = K;
  v.R = R;
  v.t = t;
  v.width = static_cast<int>(depth.cols());
  v.height = static_cast<int>(depth.rows());
  views_.push_back(std::move(v));
  return static_cast<int>(views_.size()) - 1;
}

cv::Mat GPUDepthmapCleaner::Clean(int ref_idx,
                                  const std::vector<int>& neighbor_ids) {
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  cl::Context& cl_ctx = dev.context();
  cl::CommandQueue& queue = dev.queue();

  if (!kernel_built_) {
    program_ = dev.BuildProgram(kCleanKernelSource);
    k_clean_ = cl::Kernel(program_, "acmmp_clean_depthmap");
    kernel_built_ = true;
  }

  if (ref_idx < 0 || ref_idx >= static_cast<int>(views_.size())) {
    throw std::out_of_range("Clean: ref_idx " + std::to_string(ref_idx) +
                            " out of range [0, " +
                            std::to_string(views_.size()) + ")");
  }
  for (int i = 0; i < static_cast<int>(neighbor_ids.size()); ++i) {
    if (neighbor_ids[i] < 0 ||
        neighbor_ids[i] >= static_cast<int>(views_.size())) {
      throw std::out_of_range("Clean: neighbor_ids[" + std::to_string(i) +
                              "] = " + std::to_string(neighbor_ids[i]) +
                              " out of range [0, " +
                              std::to_string(views_.size()) + ")");
    }
  }

  auto& ref = views_[ref_idx];
  const int w = ref.width;
  const int h = ref.height;
  const int npix = w * h;
  const int num_neighbors = static_cast<int>(neighbor_ids.size());
  const int num_views = 1 + num_neighbors;

  // --- Ensure image2d_t pool matches current dimensions ---
  if (w != cl_img_w_ || h != cl_img_h_) {
    cl_int err;
    cl::ImageFormat fmt(CL_R, CL_FLOAT);
    cl_ref_depth_img_ =
        cl::Image2D(cl_ctx, CL_MEM_READ_ONLY, fmt, w, h, 0, nullptr, &err);
    opencl::CheckCL(err, "Clean: ref depth Image2D create");
    cl_src_depth_imgs_.clear();
    cl_src_depth_imgs_.reserve(kMaxCleanSources);
    for (int i = 0; i < kMaxCleanSources; ++i) {
      cl::Image2D img(cl_ctx, CL_MEM_READ_ONLY, fmt, w, h, 0, nullptr, &err);
      opencl::CheckCL(err, "Clean: src depth Image2D create");
      cl_src_depth_imgs_.push_back(std::move(img));
    }
    cl_img_w_ = w;
    cl_img_h_ = h;
  }

  // --- Upload depth data into cached images ---
  cl::array<cl::size_type, 3> origin = {{0, 0, 0}};
  cl::array<cl::size_type, 3> region = {
      {static_cast<cl::size_type>(w), static_cast<cl::size_type>(h), 1}};
  queue.enqueueWriteImage(cl_ref_depth_img_, CL_TRUE, origin, region,
                          ref.depth.step[0], 0, ref.depth.data);

  for (int vi = 0; vi < num_neighbors && vi < kMaxCleanSources; ++vi) {
    auto& other = views_[neighbor_ids[vi]];
    cv::Mat other_depth;
    if (other.depth.cols != w || other.depth.rows != h) {
      cv::resize(other.depth, other_depth, cv::Size(w, h), 0, 0,
                 cv::INTER_LINEAR);
    } else {
      other_depth = other.depth;
    }
    queue.enqueueWriteImage(cl_src_depth_imgs_[vi], CL_TRUE, origin, region,
                            other_depth.step[0], 0, other_depth.data);
  }

  // --- Upload cameras (small buffer, safe as __global) ---
  std::vector<CLCamera> cameras(num_views);
  cameras[0] = MakeCLCamera(ref.K, ref.R, ref.t, w, h);
  for (int vi = 0; vi < num_neighbors; ++vi) {
    auto& other = views_[neighbor_ids[vi]];
    cameras[vi + 1] = MakeCLCamera(other.K, other.R, other.t, w, h);
  }
  const size_t cameras_bytes = cameras.size() * sizeof(CLCamera);
  if (cameras_bytes > cl_cameras_bytes_) {
    cl_cameras_ = cl::Buffer(cl_ctx, CL_MEM_READ_ONLY, cameras_bytes);
    cl_cameras_bytes_ = cameras_bytes;
  }
  queue.enqueueWriteBuffer(cl_cameras_, CL_FALSE, 0, cameras_bytes,
                           cameras.data());

  // --- Output buffer ---
  const size_t clean_bytes = npix * sizeof(float);
  if (clean_bytes > cl_clean_depth_bytes_) {
    cl_clean_depth_ = cl::Buffer(cl_ctx, CL_MEM_WRITE_ONLY, clean_bytes);
    cl_clean_depth_bytes_ = clean_bytes;
  }

  // --- Set kernel arguments ---
  int effective_min = std::min(min_consistent_views_, num_views);

  int arg = 0;
  k_clean_.setArg(arg++, cl_ref_depth_img_);
  for (int i = 0; i < kMaxCleanSources; ++i) {
    k_clean_.setArg(arg++, cl_src_depth_imgs_[i]);
  }
  k_clean_.setArg(arg++, cl_cameras_);
  k_clean_.setArg(arg++, cl_clean_depth_);
  k_clean_.setArg(arg++, w);
  k_clean_.setArg(arg++, h);
  k_clean_.setArg(arg++, num_views);
  k_clean_.setArg(arg++, same_depth_threshold_);
  k_clean_.setArg(arg++, effective_min);
  k_clean_.setArg(arg++, carving_threshold_);
  k_clean_.setArg(arg++, max_carved_views_);

  cl::NDRange global(static_cast<size_t>((w + 15) / 16 * 16),
                     static_cast<size_t>((h + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_clean_, cl::NullRange, global, local);

  queue.finish();

  cv::Mat cleaned(h, w, CV_32F);
  queue.enqueueReadBuffer(cl_clean_depth_, CL_TRUE, 0, clean_bytes,
                          cleaned.ptr<float>());
  return cleaned;
}

void GPUDepthmapCleaner::Clear() {
  views_.clear();
  // Release GPU resources.
  cl_cameras_ = cl::Buffer();
  cl_clean_depth_ = cl::Buffer();
  cl_cameras_bytes_ = 0;
  cl_clean_depth_bytes_ = 0;
  cl_ref_depth_img_ = cl::Image2D();
  cl_src_depth_imgs_.clear();
  cl_img_w_ = 0;
  cl_img_h_ = 0;
}

GPUDepthmapCleaner::~GPUDepthmapCleaner() {
  k_clean_ = cl::Kernel();
  program_ = cl::Program();
}

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
