#pragma once

#ifdef OPENSFM_HAVE_OPENCL

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <foundation/types.h>

#include <CL/opencl.hpp>
#include <Eigen/Core>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dense {
namespace opencl {

/// Check an OpenCL error code and throw on failure.
inline void CheckCL(cl_int err, const char* context) {
  if (err != CL_SUCCESS) {
    throw std::runtime_error(std::string("OpenCL error in ") + context +
                             ": code " + std::to_string(err));
  }
}

/// A single OpenCL device with its own context and command queue.
class CLDevice {
 public:
  explicit CLDevice(cl::Device dev) : device_(std::move(dev)) {
    context_ = cl::Context(device_);
    queue_ = cl::CommandQueue(context_, device_, 0);
    name_ = device_.getInfo<CL_DEVICE_NAME>();
    global_mem_size_ = device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    is_gpu_ = (device_.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU) != 0;
  }

  // std::mutex is not movable, so we need a custom move constructor.
  CLDevice(CLDevice&& other) noexcept
      : device_(std::move(other.device_)),
        context_(std::move(other.context_)),
        queue_(std::move(other.queue_)),
        name_(std::move(other.name_)),
        global_mem_size_(other.global_mem_size_),
        is_gpu_(other.is_gpu_),
        reserved_bytes_(other.reserved_bytes_) {
    // program_cache_ starts empty — programs haven't been built yet
    // during device discovery, so the source object's cache is also empty.
  }
  CLDevice& operator=(CLDevice&&) = delete;
  CLDevice(const CLDevice&) = delete;
  CLDevice& operator=(const CLDevice&) = delete;

  cl::Context& context() { return context_; }
  cl::CommandQueue& queue() { return queue_; }
  cl::Device& device() { return device_; }
  const std::string& name() const { return name_; }
  bool IsGPU() const { return is_gpu_; }
  size_t GlobalMemSize() const { return global_mem_size_; }

  /// Track memory reservations for Python-level device scheduling.
  size_t AvailableMemory() const {
    return (reserved_bytes_ >= global_mem_size_)
               ? 0
               : global_mem_size_ - reserved_bytes_;
  }
  void ReserveMemory(size_t bytes) { reserved_bytes_ += bytes; }
  void ReleaseMemory(size_t bytes) {
    reserved_bytes_ = (bytes > reserved_bytes_) ? 0 : reserved_bytes_ - bytes;
  }
  size_t ReservedMemory() const { return reserved_bytes_; }

  /// Build an OpenCL program from source (one-shot, not cached).
  cl::Program BuildProgram(const std::string& source,
                           const std::string& options = "") {
    cl::Program program(context_, source);

    std::string full_opts = "-cl-std=CL1.2 -cl-mad-enable " + options;
    try {
      program.build({device_}, full_opts.c_str());
    } catch (const cl::BuildError& e) {
      std::string log;
      for (auto& pair : e.getBuildLog()) {
        log += pair.second;
      }
      throw std::runtime_error(std::string("OpenCL build error (") + e.what() +
                               "):\n" + log);
    }
    return program;
  }

  /// Thread-safe cached program build.  The program identified by *key*
  /// is compiled exactly once from *source* and kept alive for the
  /// process lifetime.
  cl::Program GetOrBuildProgram(const std::string& key,
                                const std::string& source,
                                const std::string& options = "") {
    std::lock_guard<std::mutex> lock(program_cache_mu_);
    auto it = program_cache_.find(key);
    if (it != program_cache_.end()) {
      return it->second;
    }
    cl::Program prog = BuildProgram(source, options);
    program_cache_.emplace(key, prog);
    return prog;
  }

 private:
  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
  std::string name_;
  size_t global_mem_size_ = 0;
  bool is_gpu_ = false;
  size_t reserved_bytes_ = 0;

  std::mutex program_cache_mu_;
  std::unordered_map<std::string, cl::Program> program_cache_;
};

/// Singleton that discovers all OpenCL devices (GPUs first, then CPUs)
/// and provides indexed access to them.
class CLContext {
 public:
  /// Get the singleton instance (lazily initialised).
  static CLContext& Instance() {
    static CLContext ctx;
    return ctx;
  }

  /// Returns true if at least one device was found.
  bool IsAvailable() const { return !devices_.empty(); }

  /// Number of discovered devices.
  int NumDevices() const { return static_cast<int>(devices_.size()); }

  /// Access a specific device by index.
  CLDevice& Device(int idx) { return devices_.at(idx); }
  const CLDevice& Device(int idx) const { return devices_.at(idx); }

  // Backward-compatible accessors delegating to device 0.
  cl::Context& context() { return devices_[0].context(); }
  cl::CommandQueue& queue() { return devices_[0].queue(); }
  cl::Device& device() { return devices_[0].device(); }
  size_t GlobalMemSize() const {
    if (devices_.empty()) {
      return 0;
    }
    return devices_[0].GlobalMemSize();
  }
  cl::Program BuildProgram(const std::string& source,
                           const std::string& options = "") {
    return devices_[0].BuildProgram(source, options);
  }

 private:
  CLContext() {
    try {
      // When running inside a conda environment the ICD loader may not
      // find vendor ICDs because OCL_ICD_VENDORS is not set.  Set the
      // well-known system path as a fallback BEFORE the first OpenCL call,
      // since the ICD loader caches results after initialisation.
#ifdef __linux__
      if (!std::getenv("OCL_ICD_VENDORS")) {
        ::setenv("OCL_ICD_VENDORS", "/etc/OpenCL/vendors", 0);
      }
#endif

      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      if (platforms.empty()) {
        return;
      }

      // Collect GPUs first (higher priority), then CPUs.
      std::vector<cl::Device> gpus, cpus;
      for (auto& p : platforms) {
        std::vector<cl::Device> devs;
        try {
          p.getDevices(CL_DEVICE_TYPE_GPU, &devs);
        } catch (...) {
        }
        gpus.insert(gpus.end(), devs.begin(), devs.end());
        devs.clear();
        try {
          p.getDevices(CL_DEVICE_TYPE_CPU, &devs);
        } catch (...) {
        }
        cpus.insert(cpus.end(), devs.begin(), devs.end());
      }

      devices_.reserve(gpus.size() + cpus.size());
      for (auto& d : gpus) {
        devices_.emplace_back(std::move(d));
      }
      for (auto& d : cpus) {
        devices_.emplace_back(std::move(d));
      }

      if (!devices_.empty()) {
        std::cerr << "[OpenCL] " << devices_.size()
                  << " device(s) available:\n";
        for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
          std::cerr << "  [" << i << "] " << devices_[i].name() << " ("
                    << (devices_[i].GlobalMemSize() >> 20) << " MB)\n";
        }
      }
    } catch (...) {
      devices_.clear();
    }
  }

  CLContext(const CLContext&) = delete;
  CLContext& operator=(const CLContext&) = delete;

  std::vector<CLDevice> devices_;
};

}  // namespace opencl

// =====================================================================
// Host-side camera struct matching the OpenCL kernel layout.
// =====================================================================
struct CLCamera {
  float K[9];
  float R[9];
  float t[3];
  int width;
  int height;
};

inline CLCamera MakeCLCamera(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                             int w, int h) {
  static_assert(sizeof(CLCamera) == 92,
                "CLCamera struct padding mismatch with OpenCL kernel Camera");
  CLCamera c{};
  // Eigen is col-major; CLCamera expects row-major float[9].
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(c.K) =
      K.cast<float>();
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(c.R) =
      R.cast<float>();
  Eigen::Map<Eigen::Vector3f>(c.t) = t.cast<float>();
  c.width = w;
  c.height = h;
  return c;
}

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
