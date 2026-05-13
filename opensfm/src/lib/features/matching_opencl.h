#pragma once

#include <foundation/python_types.h>

#include <vector>

namespace features {

/// Returns true if at least one OpenCL device is available for matching.
bool opencl_matching_available();

/// Returns the number of available OpenCL devices.
int opencl_num_devices();

/// Brute-force descriptor matching on OpenCL with Lowe's ratio test.
///
/// Finds the nearest neighbour in f2 for each descriptor in f1, keeping
/// only matches that pass the ratio test (best_dist < ratio² * second_dist).
///
/// @param f1          [N1 x D] float32 descriptors (query set).
/// @param f2          [N2 x D] float32 descriptors (reference set).
/// @param lowes_ratio Lowe's ratio threshold (typically 0.8).
/// @param device_idx  OpenCL device index (0 = first GPU if available).
/// @return [M x 2] int32 array of (query_idx, ref_idx) matches.
py::array_t<int> match_brute_force_opencl(foundation::pyarray_f f1,
                                          foundation::pyarray_f f2,
                                          float lowes_ratio,
                                          int device_idx = 0);

/// Symmetric brute-force matching: matches in both directions and keeps
/// only mutually consistent pairs.
///
/// @param f1          [N1 x D] float32 descriptors.
/// @param f2          [N2 x D] float32 descriptors.
/// @param lowes_ratio Lowe's ratio threshold.
/// @param device_idx  OpenCL device index.
/// @return [M x 2] int32 array of (idx_in_f1, idx_in_f2) matches.
py::array_t<int> match_brute_force_opencl_symmetric(foundation::pyarray_f f1,
                                                    foundation::pyarray_f f2,
                                                    float lowes_ratio,
                                                    int device_idx = 0);

/// Brute-force Hamming matching on OpenCL for binary descriptors.
///
/// @param f1          [N1 x W] uint32 packed binary descriptors (W words).
/// @param f2          [N2 x W] uint32 packed binary descriptors.
/// @param lowes_ratio Lowe's ratio threshold.
/// @param device_idx  OpenCL device index.
/// @return [M x 2] int32 array of (query_idx, ref_idx) matches.
py::array_t<int> match_hamming_opencl(py::array_t<uint32_t> f1,
                                      py::array_t<uint32_t> f2,
                                      float lowes_ratio, int device_idx = 0);

/// Symmetric Hamming matching on OpenCL for binary descriptors.
///
/// @param f1          [N1 x W] uint32 packed binary descriptors.
/// @param f2          [N2 x W] uint32 packed binary descriptors.
/// @param lowes_ratio Lowe's ratio threshold.
/// @param device_idx  OpenCL device index.
/// @return [M x 2] int32 array of (idx_in_f1, idx_in_f2) matches.
py::array_t<int> match_hamming_opencl_symmetric(py::array_t<uint32_t> f1,
                                                py::array_t<uint32_t> f2,
                                                float lowes_ratio,
                                                int device_idx = 0);

/// Batched symmetric Hamming matching on OpenCL.
///
/// Matches all pairs in a single kernel dispatch for maximum GPU occupancy.
///
/// @param f1_list     List of [Ni x W] uint32 packed binary descriptor arrays.
/// @param f2_list     List of [Mi x W] uint32 packed binary descriptor arrays.
/// @param lowes_ratio Lowe's ratio threshold.
/// @param device_idx  OpenCL device index.
/// @return List of [Ki x 2] int32 arrays of (idx_in_f1, idx_in_f2) matches.
py::list match_hamming_opencl_batch_symmetric(py::list f1_list,
                                              py::list f2_list,
                                              float lowes_ratio,
                                              int device_idx = 0);

}  // namespace features
