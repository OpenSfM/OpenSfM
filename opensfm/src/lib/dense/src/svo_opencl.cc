#ifdef OPENSFM_HAVE_OPENCL

#include <dense/svo_opencl.h>
#include <dense/svo_opencl_kernels.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace dense {

SVOIntegratorCL::SVOIntegratorCL(int device_idx) : device_idx_(device_idx) {}

void SVOIntegratorCL::BuildKernels() {
  if (kernels_built_) {
    return;
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  program_ = dev.GetOrBuildProgram("svo", kSVOKernelSource);

  cl_int err;
  k_clear_ = cl::Kernel(program_, "svo_clear_table", &err);
  opencl::CheckCL(err, "kernel svo_clear_table");

  k_integrate_ = cl::Kernel(program_, "svo_integrate", &err);
  opencl::CheckCL(err, "kernel svo_integrate");

  k_count_clear_ = cl::Kernel(program_, "svo_clear_count_table", &err);
  opencl::CheckCL(err, "kernel svo_clear_count_table");

  k_count_ = cl::Kernel(program_, "svo_count", &err);
  opencl::CheckCL(err, "kernel svo_count");

  k_extract_ = cl::Kernel(program_, "svo_extract_points", &err);
  opencl::CheckCL(err, "kernel svo_extract_points");

  // 1-byte dummy buffer used as placeholder when normal/color/mask is absent.
  cl_dummy_ = cl::Buffer(dev.context(), static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
                         static_cast<cl::size_type>(1), nullptr, &err);
  opencl::CheckCL(err, "dummy buffer");

  kernels_built_ = true;
}

void SVOIntegratorCL::Initialize(uint32_t capacity) {
  BuildKernels();

  // Release counting-pass resources (if any) before allocating the
  // much larger integration table — avoids holding both in GPU memory.
  if (count_capacity_ > 0) {
    cl_count_table_ = cl::Buffer();
    cl_counter_ = cl::Buffer();
    count_capacity_ = 0;
    count_mask_ = 0;
  }

  // Round up to power of two.
  uint32_t cap = 1;
  while (cap < capacity) {
    cap <<= 1;
  }

  // Clamp to device memory limits.
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  const size_t max_alloc = dev.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  const size_t avail = dev.AvailableMemory();
  // Use at most 60% of available memory or the max single-alloc size.
  const size_t mem_limit = std::min(max_alloc, avail * 6 / 10);
  const uint32_t max_slots = static_cast<uint32_t>(std::min<size_t>(
      mem_limit / sizeof(GPUVoxelSlot), std::numeric_limits<uint32_t>::max()));
  // Round max_slots down to power of two.
  uint32_t max_cap = 1;
  while (max_cap <= max_slots / 2) {
    max_cap <<= 1;
  }
  if (cap > max_cap) {
    std::cerr << "[SVOIntegratorCL] Clamping capacity from " << cap << " to "
              << max_cap << " (max_alloc=" << (max_alloc >> 20)
              << "MB, avail=" << (avail >> 20) << "MB)\n";
    cap = max_cap;
  }

  capacity_ = cap;
  capacity_mask_ = cap - 1;

  auto& ctx = dev.context();
  auto& queue = dev.queue();

  const size_t table_bytes =
      static_cast<size_t>(capacity_) * sizeof(GPUVoxelSlot);
  std::cerr << "[SVOIntegratorCL] GPU hash table: " << capacity_ << " slots ("
            << (table_bytes >> 20) << " MB) on " << dev.name() << "\n";
  cl_int err;
  cl_table_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, table_bytes, nullptr, &err);
  opencl::CheckCL(err, "hash table buffer");

  cl_camera_ =
      cl::Buffer(ctx, CL_MEM_READ_ONLY, sizeof(SVOCameraGPU), nullptr, &err);
  opencl::CheckCL(err, "camera buffer");

  // Clear the table.
  k_clear_.setArg(0, cl_table_);
  k_clear_.setArg(1, capacity_);

  const size_t global = ((capacity_ + 255u) / 256u) * 256u;
  queue.enqueueNDRangeKernel(k_clear_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();
}

void SVOIntegratorCL::EnsureFrameBuffers(int rows, int cols, bool has_normal,
                                         bool has_color, bool has_mask,
                                         bool has_weight) {
  auto& ctx = opencl::CLContext::Instance().Device(device_idx_).context();
  cl_int err;
  const size_t npix = static_cast<size_t>(rows) * cols;

  const size_t d_bytes = npix * sizeof(float);
  if (d_bytes > depth_bytes_) {
    cl_depth_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, d_bytes, nullptr, &err);
    opencl::CheckCL(err, "depth buffer");
    depth_bytes_ = d_bytes;
  }
  if (has_normal) {
    const size_t n_bytes = npix * 3 * sizeof(float);
    if (n_bytes > normal_bytes_) {
      cl_normal_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, n_bytes, nullptr, &err);
      opencl::CheckCL(err, "normal buffer");
      normal_bytes_ = n_bytes;
    }
  }
  if (has_color) {
    const size_t c_bytes = npix * 3 * sizeof(uint8_t);
    if (c_bytes > color_bytes_) {
      cl_color_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, c_bytes, nullptr, &err);
      opencl::CheckCL(err, "color buffer");
      color_bytes_ = c_bytes;
    }
  }
  if (has_mask) {
    const size_t m_bytes = npix * sizeof(uint8_t);
    if (m_bytes > mask_bytes_) {
      cl_mask_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, m_bytes, nullptr, &err);
      opencl::CheckCL(err, "mask buffer");
      mask_bytes_ = m_bytes;
    }
  }
  if (has_weight) {
    const size_t w_bytes = npix * sizeof(float);
    if (w_bytes > weight_bytes_) {
      cl_weight_ = cl::Buffer(ctx, CL_MEM_READ_ONLY, w_bytes, nullptr, &err);
      opencl::CheckCL(err, "weight buffer");
      weight_bytes_ = w_bytes;
    }
  }
}

// Copy an Eigen ColMajor 3×3 matrix into a row-major float[9] array.
static void ToRowMajor(const Mat3f& M, float* out) {
  // ColMajor data of M^T == RowMajor data of M.
  const Mat3f Mt = M.transpose();
  std::memcpy(out, Mt.data(), 9 * sizeof(float));
}

void SVOIntegratorCL::Integrate(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                                const float* depth, int rows, int cols,
                                const float* normal, const uint8_t* color,
                                const uint8_t* mask, const float* weight,
                                float voxel_size, float trunc_dist,
                                const Eigen::Vector3i* bbox_min,
                                const Eigen::Vector3i* bbox_max) {
  if (capacity_ == 0) {
    throw std::runtime_error("SVOIntegratorCL: Initialize() not called");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  const bool has_normal = (normal != nullptr);
  const bool has_color = (color != nullptr);
  const bool has_mask = (mask != nullptr);
  const bool has_weight = (weight != nullptr);

  EnsureFrameBuffers(rows, cols, has_normal, has_color, has_mask, has_weight);

  const size_t npix = static_cast<size_t>(rows) * cols;

  // Upload depth.
  queue.enqueueWriteBuffer(cl_depth_, CL_FALSE, 0, npix * sizeof(float), depth);
  // Upload optional arrays.
  if (has_normal) {
    queue.enqueueWriteBuffer(cl_normal_, CL_FALSE, 0, npix * 3 * sizeof(float),
                             normal);
  }
  if (has_color) {
    queue.enqueueWriteBuffer(cl_color_, CL_FALSE, 0, npix * 3 * sizeof(uint8_t),
                             color);
  }
  if (has_mask) {
    queue.enqueueWriteBuffer(cl_mask_, CL_FALSE, 0, npix * sizeof(uint8_t),
                             mask);
  }
  if (has_weight) {
    queue.enqueueWriteBuffer(cl_weight_, CL_FALSE, 0, npix * sizeof(float),
                             weight);
  }

  // Prepare camera parameters (row-major matrices).
  SVOCameraGPU cam{};
  const Mat3f Kinv = K.inverse();
  const Mat3f Rinv = R.transpose();
  const Vec3f cam_pos = -Rinv * t;

  ToRowMajor(Kinv, cam.Kinv);
  ToRowMajor(Rinv, cam.Rinv);
  ToRowMajor(R, cam.R);
  cam.t[0] = t.x();
  cam.t[1] = t.y();
  cam.t[2] = t.z();
  cam.cam_pos[0] = cam_pos.x();
  cam.cam_pos[1] = cam_pos.y();
  cam.cam_pos[2] = cam_pos.z();

  queue.enqueueWriteBuffer(cl_camera_, CL_FALSE, 0, sizeof(SVOCameraGPU), &cam);

  // Finish all uploads before launching the kernel.
  queue.finish();

  // Set kernel arguments.
  int arg = 0;
  k_integrate_.setArg(arg++, cl_table_);
  k_integrate_.setArg(arg++, capacity_mask_);
  k_integrate_.setArg(arg++, cl_depth_);
  k_integrate_.setArg(arg++, has_normal ? cl_normal_ : cl_dummy_);
  k_integrate_.setArg(arg++, has_color ? cl_color_ : cl_dummy_);
  k_integrate_.setArg(arg++, has_mask ? cl_mask_ : cl_dummy_);
  k_integrate_.setArg(arg++, has_weight ? cl_weight_ : cl_dummy_);
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_normal));
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_color));
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_mask));
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_weight));
  k_integrate_.setArg(arg++, cl_camera_);
  k_integrate_.setArg(arg++, static_cast<cl_int>(rows));
  k_integrate_.setArg(arg++, static_cast<cl_int>(cols));
  k_integrate_.setArg(arg++, trunc_dist);
  k_integrate_.setArg(arg++, voxel_size);
  k_integrate_.setArg(arg++, 1.0f / voxel_size);

  // Bounding box arguments (voxel integer coordinates).
  const bool has_bbox = (bbox_min != nullptr && bbox_max != nullptr);
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_min).x() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_min).y() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_min).z() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_max).x() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_max).y() : 0));
  k_integrate_.setArg(arg++,
                      static_cast<cl_int>(has_bbox ? (*bbox_max).z() : 0));
  k_integrate_.setArg(arg++, static_cast<cl_int>(has_bbox ? 1 : 0));

  // Launch: one work-item per pixel.
  cl::NDRange global(static_cast<size_t>((cols + 15) / 16 * 16),
                     static_cast<size_t>((rows + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_integrate_, cl::NullRange, global, local);
  queue.finish();
}

// ── Counting pass ─────────────────────────────────────────────────────

void SVOIntegratorCL::InitializeCounting(uint32_t capacity) {
  BuildKernels();

  // Round up to power of two.
  uint32_t cap = 1;
  while (cap < capacity) {
    cap <<= 1;
  }

  // Clamp to device memory limits.
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  const size_t max_alloc = dev.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  const size_t avail = dev.AvailableMemory();
  const size_t mem_limit = std::min(max_alloc, avail * 6 / 10);
  const uint32_t max_slots = static_cast<uint32_t>(std::min<size_t>(
      mem_limit / sizeof(GPUCountSlot), std::numeric_limits<uint32_t>::max()));
  uint32_t max_cap = 1;
  while (max_cap <= max_slots / 2) {
    max_cap <<= 1;
  }
  if (cap > max_cap) {
    std::cerr << "[SVOIntegratorCL] Clamping count capacity from " << cap
              << " to " << max_cap << "\n";
    cap = max_cap;
  }

  count_capacity_ = cap;
  count_mask_ = cap - 1;

  auto& ctx = dev.context();
  auto& queue = dev.queue();

  const size_t table_bytes =
      static_cast<size_t>(count_capacity_) * sizeof(GPUCountSlot);
  std::cerr << "[SVOIntegratorCL] Count table: " << count_capacity_
            << " slots (" << (table_bytes >> 20) << " MB)\n";

  cl_int err;
  cl_count_table_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, table_bytes, nullptr, &err);
  opencl::CheckCL(err, "count table buffer");

  // Atomic counter for unique inserts (single uint32).
  cl_counter_ =
      cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &err);
  opencl::CheckCL(err, "counter buffer");

  // Camera params buffer (shared with integration path).
  cl_camera_ =
      cl::Buffer(ctx, CL_MEM_READ_ONLY, sizeof(SVOCameraGPU), nullptr, &err);
  opencl::CheckCL(err, "camera buffer (counting)");

  // Zero the counter.
  uint32_t zero = 0;
  queue.enqueueWriteBuffer(cl_counter_, CL_FALSE, 0, sizeof(uint32_t), &zero);

  // Clear the counting table.
  k_count_clear_.setArg(0, cl_count_table_);
  k_count_clear_.setArg(1, count_capacity_);
  const size_t global = ((count_capacity_ + 255u) / 256u) * 256u;
  queue.enqueueNDRangeKernel(k_count_clear_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();
}

void SVOIntegratorCL::Count(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                            const float* depth, int rows, int cols,
                            const uint8_t* mask, float voxel_size,
                            float trunc_dist, const Eigen::Vector3i* bbox_min,
                            const Eigen::Vector3i* bbox_max) {
  if (count_capacity_ == 0) {
    throw std::runtime_error(
        "SVOIntegratorCL: InitializeCounting() not called");
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  const bool has_mask = (mask != nullptr);
  // Reuse the depth/mask frame buffers from the integration path.
  EnsureFrameBuffers(rows, cols, /*has_normal=*/false, /*has_color=*/false,
                     has_mask, /*has_weight=*/false);

  const size_t npix = static_cast<size_t>(rows) * cols;

  // Upload depth.
  queue.enqueueWriteBuffer(cl_depth_, CL_FALSE, 0, npix * sizeof(float), depth);
  if (has_mask) {
    queue.enqueueWriteBuffer(cl_mask_, CL_FALSE, 0, npix * sizeof(uint8_t),
                             mask);
  }

  // Camera params (reuse cl_camera_ buffer).
  SVOCameraGPU cam{};
  const Mat3f Kinv = K.inverse();
  const Mat3f Rinv = R.transpose();
  const Vec3f cam_pos = -Rinv * t;
  ToRowMajor(Kinv, cam.Kinv);
  ToRowMajor(Rinv, cam.Rinv);
  ToRowMajor(R, cam.R);
  cam.t[0] = t.x();
  cam.t[1] = t.y();
  cam.t[2] = t.z();
  cam.cam_pos[0] = cam_pos.x();
  cam.cam_pos[1] = cam_pos.y();
  cam.cam_pos[2] = cam_pos.z();

  queue.enqueueWriteBuffer(cl_camera_, CL_FALSE, 0, sizeof(SVOCameraGPU), &cam);
  queue.finish();

  // Set kernel arguments.
  int arg = 0;
  k_count_.setArg(arg++, cl_count_table_);
  k_count_.setArg(arg++, count_mask_);
  k_count_.setArg(arg++, cl_counter_);
  k_count_.setArg(arg++, cl_depth_);
  k_count_.setArg(arg++, has_mask ? cl_mask_ : cl_dummy_);
  k_count_.setArg(arg++, static_cast<cl_int>(has_mask));
  k_count_.setArg(arg++, cl_camera_);
  k_count_.setArg(arg++, static_cast<cl_int>(rows));
  k_count_.setArg(arg++, static_cast<cl_int>(cols));
  k_count_.setArg(arg++, trunc_dist);
  k_count_.setArg(arg++, voxel_size);
  k_count_.setArg(arg++, 1.0f / voxel_size);

  // Bounding box arguments (voxel integer coordinates).
  const bool has_bbox_c = (bbox_min != nullptr && bbox_max != nullptr);
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_min).x() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_min).y() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_min).z() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_max).x() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_max).y() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? (*bbox_max).z() : 0));
  k_count_.setArg(arg++, static_cast<cl_int>(has_bbox_c ? 1 : 0));

  cl::NDRange global(static_cast<size_t>((cols + 15) / 16 * 16),
                     static_cast<size_t>((rows + 15) / 16 * 16));
  cl::NDRange local(16, 16);
  queue.enqueueNDRangeKernel(k_count_, cl::NullRange, global, local);
  queue.finish();
}

uint32_t SVOIntegratorCL::GetUniqueCount() const {
  if (count_capacity_ == 0) {
    return 0;
  }
  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();
  uint32_t count = 0;
  queue.enqueueReadBuffer(cl_counter_, CL_TRUE, 0, sizeof(uint32_t), &count);
  return count;
}

VoxelMap SVOIntegratorCL::Download() const {
  if (capacity_ == 0) {
    return {};
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& queue = dev.queue();

  std::vector<GPUVoxelSlot> host_table(capacity_);
  queue.enqueueReadBuffer(cl_table_, CL_TRUE, 0,
                          capacity_ * sizeof(GPUVoxelSlot), host_table.data());

  VoxelMap voxels;
  int occupied = 0;
  int merged = 0;

  for (uint32_t i = 0; i < capacity_; ++i) {
    const auto& slot = host_table[i];
    if (slot.key_ab == kEmptyKey || slot.count == 0) {
      continue;
    }

    ++occupied;

    const int kx = static_cast<int>((slot.key_ab >> 16) & 0xFFFF) - 32768;
    const int ky = static_cast<int>(slot.key_ab & 0xFFFF) - 32768;
    const int kz = slot.key_c;

    // Values are pre-multiplied by weight in the kernel, so divide by
    // sum_weight (not count) to get weighted averages.
    const float sw = static_cast<float>(slot.sum_weight);
    if (sw < 1.0f) {
      continue;  // degenerate — no meaningful weight
    }
    const float inv_sw = 1.0f / sw;
    const float inv_fp_sw = inv_sw / static_cast<float>(kFPScale);

    const float avg_tsdf = static_cast<float>(slot.sum_tsdf) * inv_fp_sw;
    const float avg_nx = static_cast<float>(slot.sum_nx) * inv_fp_sw;
    const float avg_ny = static_cast<float>(slot.sum_ny) * inv_fp_sw;
    const float avg_nz = static_cast<float>(slot.sum_nz) * inv_fp_sw;
    const float avg_r = static_cast<float>(slot.sum_r) * inv_sw;
    const float avg_g = static_cast<float>(slot.sum_g) * inv_sw;
    const float avg_b = static_cast<float>(slot.sum_b) * inv_sw;

    SVOVoxel v;
    v.tsdf = PackSnorm16(std::clamp(avg_tsdf, -1.0f, 1.0f));
    // Use sum_weight (in kWeightScale units) for the weight so that
    // confidence-weighted observations accumulate correctly.
    v.weight = static_cast<uint16_t>(
        std::min(static_cast<float>(slot.sum_weight), 65535.0f));
    v.nx = PackSnorm16(std::clamp(avg_nx, -1.0f, 1.0f));
    v.ny = PackSnorm16(std::clamp(avg_ny, -1.0f, 1.0f));
    v.nz = PackSnorm16(std::clamp(avg_nz, -1.0f, 1.0f));
    v.r = static_cast<uint8_t>(std::clamp(avg_r, 0.0f, 255.0f));
    v.g = static_cast<uint8_t>(std::clamp(avg_g, 0.0f, 255.0f));
    v.b = static_cast<uint8_t>(std::clamp(avg_b, 0.0f, 255.0f));

    VoxelCoord coord{kx, ky, kz};
    auto [it, inserted] = voxels.try_emplace(coord, v);
    if (!inserted) {
      // Merge duplicate slot created by hash contention.
      auto& e = it->second;
      const float ew = static_cast<float>(e.weight);
      const float nw = static_cast<float>(v.weight);
      const float tw = ew + nw;
      const float inv_tw = 1.0f / tw;
      e.tsdf = PackSnorm16(
          (UnpackSnorm16(e.tsdf) * ew + UnpackSnorm16(v.tsdf) * nw) * inv_tw);
      e.nx = PackSnorm16((UnpackSnorm16(e.nx) * ew + UnpackSnorm16(v.nx) * nw) *
                         inv_tw);
      e.ny = PackSnorm16((UnpackSnorm16(e.ny) * ew + UnpackSnorm16(v.ny) * nw) *
                         inv_tw);
      e.nz = PackSnorm16((UnpackSnorm16(e.nz) * ew + UnpackSnorm16(v.nz) * nw) *
                         inv_tw);
      auto clamp_u8 = [](float x) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(x, 0.0f, 255.0f));
      };
      e.r = clamp_u8(
          (static_cast<float>(e.r) * ew + static_cast<float>(v.r) * nw) *
          inv_tw);
      e.g = clamp_u8(
          (static_cast<float>(e.g) * ew + static_cast<float>(v.g) * nw) *
          inv_tw);
      e.b = clamp_u8(
          (static_cast<float>(e.b) * ew + static_cast<float>(v.b) * nw) *
          inv_tw);
      e.weight = static_cast<uint16_t>(std::min(tw, 65535.0f));
      ++merged;
    }
  }

  return voxels;
}

void SVOIntegratorCL::ExtractPoints(float min_weight, float voxel_size,
                                    std::vector<Vec3f>* points,
                                    std::vector<Vec3f>* normals,
                                    std::vector<Vec3<uint8_t>>* colors) {
  points->clear();
  normals->clear();
  colors->clear();

  if (capacity_ == 0) {
    return;
  }

  auto& dev = opencl::CLContext::Instance().Device(device_idx_);
  auto& ctx = dev.context();
  auto& queue = dev.queue();

  // Allocate output buffers.  Surface points are typically 5-10% of voxels;
  // capacity/4 is a generous upper bound.
  const size_t max_alloc = dev.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  // Each point needs 3*float(pos) + 3*float(norm) + 3*uint8(color) = 27 bytes.
  const uint32_t mem_limit_pts = static_cast<uint32_t>(std::min<size_t>(
      max_alloc / (3 * sizeof(float)), std::numeric_limits<uint32_t>::max()));
  const uint32_t max_output = std::min(capacity_ / 4, mem_limit_pts);

  const size_t pts_bytes = static_cast<size_t>(max_output) * 3 * sizeof(float);
  const size_t clr_bytes =
      static_cast<size_t>(max_output) * 3 * sizeof(uint8_t);

  std::cerr << "[SVOIntegratorCL] ExtractPoints: max_output=" << max_output
            << " (" << ((pts_bytes * 2 + clr_bytes) >> 20) << " MB buffers)\n";

  cl_int err;
  cl::Buffer cl_out_pts(ctx, CL_MEM_WRITE_ONLY, pts_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract points buffer");
  cl::Buffer cl_out_nrm(ctx, CL_MEM_WRITE_ONLY, pts_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract normals buffer");
  cl::Buffer cl_out_clr(ctx, CL_MEM_WRITE_ONLY, clr_bytes, nullptr, &err);
  opencl::CheckCL(err, "extract colors buffer");
  cl::Buffer cl_out_counter(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr,
                            &err);
  opencl::CheckCL(err, "extract counter buffer");

  // Zero the counter.
  uint32_t zero = 0;
  queue.enqueueWriteBuffer(cl_out_counter, CL_TRUE, 0, sizeof(uint32_t), &zero);

  // min_weight is in SVOVoxel weight units (already scaled by kWeightScale).
  // The GPU stores sum_weight in WEIGHT_SCALE (128) units.
  const float min_weight_scaled = min_weight * kWeightScale;

  // Set kernel arguments.
  int arg = 0;
  k_extract_.setArg(arg++, cl_table_);
  k_extract_.setArg(arg++, capacity_mask_);
  k_extract_.setArg(arg++, capacity_);
  k_extract_.setArg(arg++, min_weight_scaled);
  k_extract_.setArg(arg++, voxel_size);
  k_extract_.setArg(arg++, cl_out_pts);
  k_extract_.setArg(arg++, cl_out_nrm);
  k_extract_.setArg(arg++, cl_out_clr);
  k_extract_.setArg(arg++, cl_out_counter);
  k_extract_.setArg(arg++, max_output);

  // Launch: one work-item per hash table slot.
  const size_t global = ((static_cast<size_t>(capacity_) + 255u) / 256u) * 256u;
  queue.enqueueNDRangeKernel(k_extract_, cl::NullRange, cl::NDRange(global),
                             cl::NDRange(256));
  queue.finish();

  // Read back the count.
  uint32_t n_points = 0;
  queue.enqueueReadBuffer(cl_out_counter, CL_TRUE, 0, sizeof(uint32_t),
                          &n_points);
  if (n_points > max_output) {
    std::cerr << "[SVOIntegratorCL] Warning: extraction buffer overflow, "
              << n_points << " > " << max_output << ". Clamping.\n";
    n_points = max_output;
  }

  std::cerr << "[SVOIntegratorCL] GPU extracted " << n_points
            << " surface points\n";

  if (n_points == 0) {
    return;
  }

  // Read back only the used portion.
  const size_t read_pts = static_cast<size_t>(n_points) * 3 * sizeof(float);
  const size_t read_clr = static_cast<size_t>(n_points) * 3 * sizeof(uint8_t);

  std::vector<float> host_pts(n_points * 3);
  std::vector<float> host_nrm(n_points * 3);
  std::vector<uint8_t> host_clr(n_points * 3);

  queue.enqueueReadBuffer(cl_out_pts, CL_FALSE, 0, read_pts, host_pts.data());
  queue.enqueueReadBuffer(cl_out_nrm, CL_FALSE, 0, read_pts, host_nrm.data());
  queue.enqueueReadBuffer(cl_out_clr, CL_FALSE, 0, read_clr, host_clr.data());
  queue.finish();

  // Convert to output vectors.
  points->resize(n_points);
  normals->resize(n_points);
  colors->resize(n_points);

  for (uint32_t i = 0; i < n_points; ++i) {
    (*points)[i] =
        Vec3f(host_pts[i * 3], host_pts[i * 3 + 1], host_pts[i * 3 + 2]);
    (*normals)[i] =
        Vec3f(host_nrm[i * 3], host_nrm[i * 3 + 1], host_nrm[i * 3 + 2]);
    (*colors)[i] = Vec3<uint8_t>(host_clr[i * 3], host_clr[i * 3 + 1],
                                 host_clr[i * 3 + 2]);
  }
}

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
