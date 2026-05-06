#include <dense/svo.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace dense {

SparseVoxelOctree::SparseVoxelOctree(float voxel_size, float trunc_factor)
    : voxel_size_(voxel_size),
      inv_voxel_size_(1.0f / voxel_size),
      trunc_dist_(trunc_factor * voxel_size) {}

VoxelCoord SparseVoxelOctree::WorldToVoxel(const Vec3f& p) const {
  return {static_cast<int32_t>(std::floor(p.x() * inv_voxel_size_)),
          static_cast<int32_t>(std::floor(p.y() * inv_voxel_size_)),
          static_cast<int32_t>(std::floor(p.z() * inv_voxel_size_))};
}

Vec3f SparseVoxelOctree::VoxelCenter(const VoxelCoord& vc) const {
  return Vec3f((vc.x + 0.5f) * voxel_size_, (vc.y + 0.5f) * voxel_size_,
               (vc.z + 0.5f) * voxel_size_);
}

void SparseVoxelOctree::Integrate(const Mat3f& K, const Mat3f& R,
                                  const Vec3f& t, const float* depth, int rows,
                                  int cols, const float* normal,
                                  const uint8_t* color, const uint8_t* mask,
                                  const float* weight,
                                  const Eigen::Vector3i* bbox_min,
                                  const Eigen::Vector3i* bbox_max) {
  const Mat3f Kinv = K.inverse();
  const Mat3f Rinv = R.transpose();
  // Camera center in world frame: C = -R^T * t.
  const Vec3f cam_pos = -Rinv * t;
  const float trunc = trunc_dist_;

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      int idx = r * cols + c;

      if (mask && mask[idx] == 0) {
        continue;
      }

      float d = depth[idx];
      if (d <= 0.0f) {
        continue;
      }

      // Back-project to world.
      Vec3f p_cam = d * Kinv * Vec3f(c, r, 1.0f);
      Vec3f p_world = Rinv * (p_cam - t);

      // Ray direction (camera center → surface point) in world frame.
      Vec3f ray = p_world - cam_pos;
      float ray_len = ray.norm();
      if (ray_len < 1e-8f) {
        continue;
      }
      ray /= ray_len;

      // Walk voxels along the ray within [d - trunc, d + trunc].
      float t_start = std::max(0.0f, ray_len - trunc);
      float t_end = ray_len + trunc;

      // Step size = voxel side length (each step moves roughly one voxel).
      float step = voxel_size_;

      // Normal in world frame.
      // PixelData3f is (3, H*W) ColMajor: interleaved [nx0,ny0,nz0, nx1,...].
      float nx_w = 0, ny_w = 0, nz_w = 0;
      if (normal) {
        Vec3f n_cam(normal[idx * 3 + 0], normal[idx * 3 + 1],
                    normal[idx * 3 + 2]);
        float nlen = n_cam.norm();
        if (nlen > 1e-6f) {
          n_cam /= nlen;
          Vec3f n_world = Rinv * n_cam;
          nx_w = n_world.x();
          ny_w = n_world.y();
          nz_w = n_world.z();

          // cos(angle between ray and surface normal).
          float cos_theta = std::abs(ray.dot(n_world));
          if (cos_theta < 0.15f) {
            continue;  // grazing angle — skip pixel
          }
        }
      }

      // Color (PixelData3u8 is also interleaved).
      float cr = 0, cg = 0, cb = 0;
      if (color) {
        cr = static_cast<float>(color[idx * 3 + 0]);
        cg = static_cast<float>(color[idx * 3 + 1]);
        cb = static_cast<float>(color[idx * 3 + 2]);
      }

      for (float tt = t_start; tt <= t_end; tt += step) {
        Vec3f sample = cam_pos + tt * ray;
        VoxelCoord vc = WorldToVoxel(sample);

        // Skip voxels outside the bounding box (sub-volume clipping).
        if (bbox_min && bbox_max) {
          if (vc.x < (*bbox_min).x() || vc.x > (*bbox_max).x() ||
              vc.y < (*bbox_min).y() || vc.y > (*bbox_max).y() ||
              vc.z < (*bbox_min).z() || vc.z > (*bbox_max).z())
            continue;
        }

        // Signed distance: positive = in front of surface.
        Vec3f center = VoxelCenter(vc);
        // Project voxel center into camera to get its depth.
        Vec3f p_cam_v = R * center + t;
        if (p_cam_v.z() <= 0.0f) {
          continue;
        }

        float sdf = d - p_cam_v.z();  // positive if voxel is in front

        if (sdf < -trunc) {
          continue;
        }
        float tsdf = std::min(sdf / trunc, 1.0f);  // clamp to [-1, 1]

        // Weight: confidence only (cos theta is too noisy).
        float w = (weight ? weight[idx] : 1.0f) * kWeightScale;

        SVOVoxel& v = voxels_[vc];
        float old_w = static_cast<float>(v.weight);
        float new_w = old_w + w;
        float inv_new_w = 1.0f / new_w;

        // Running weighted average (unpack → blend → repack).
        v.tsdf =
            PackSnorm16((UnpackSnorm16(v.tsdf) * old_w + tsdf * w) * inv_new_w);

        v.nx =
            PackSnorm16((UnpackSnorm16(v.nx) * old_w + nx_w * w) * inv_new_w);
        v.ny =
            PackSnorm16((UnpackSnorm16(v.ny) * old_w + ny_w * w) * inv_new_w);
        v.nz =
            PackSnorm16((UnpackSnorm16(v.nz) * old_w + nz_w * w) * inv_new_w);

        auto clamp_u8 = [](float x) -> uint8_t {
          return static_cast<uint8_t>(std::clamp(x, 0.0f, 255.0f));
        };
        v.r = clamp_u8((static_cast<float>(v.r) * old_w + cr * w) * inv_new_w);
        v.g = clamp_u8((static_cast<float>(v.g) * old_w + cg * w) * inv_new_w);
        v.b = clamp_u8((static_cast<float>(v.b) * old_w + cb * w) * inv_new_w);

        v.weight = static_cast<uint16_t>(std::min(new_w, 65535.0f));
      }
    }
  }
}

void SparseVoxelOctree::ExtractPoints(
    float min_weight, std::vector<Vec3f>* points, std::vector<Vec3f>* normals,
    std::vector<Vec3<uint8_t>>* colors) const {
  points->clear();
  normals->clear();
  colors->clear();

  // Zero-crossing extraction (OpenVDB-style):  for each voxel, check the
  // three positive-axis neighbours (+X, +Y, +Z).  When the TSDF sign flips
  // between a voxel and its neighbour, linearly interpolate the crossing
  // position, normal and color.  Each edge is visited exactly once, so no
  // duplicates are produced.
  static constexpr VoxelCoord kOffsets[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  const uint16_t min_w =
      static_cast<uint16_t>(std::max(0.0f, min_weight * kWeightScale));
  for (const auto& [vc, v] : voxels_) {
    if (v.weight < min_w) {
      continue;
    }
    float tsdf_a = UnpackSnorm16(v.tsdf);

    for (const auto& off : kOffsets) {
      VoxelCoord nb = {vc.x + off.x, vc.y + off.y, vc.z + off.z};
      auto it = voxels_.find(nb);
      if (it == voxels_.end()) {
        continue;
      }

      const SVOVoxel& vn = it->second;
      if (vn.weight < min_w) {
        continue;
      }
      float tsdf_b = UnpackSnorm16(vn.tsdf);

      // Sign change?  (one positive, one negative — skip if both same sign)
      if ((tsdf_a > 0) == (tsdf_b > 0)) {
        continue;
      }
      // Skip if either is exactly zero (degenerate).
      if (tsdf_a == 0.0f || tsdf_b == 0.0f) {
        continue;
      }

      // Interpolation factor: solve  tsdf_a + t*(tsdf_b - tsdf_a) = 0.
      float t = tsdf_a / (tsdf_a - tsdf_b);
      t = std::clamp(t, 0.0f, 1.0f);

      // Interpolated position between the two voxel centres.
      Vec3f pa = VoxelCenter(vc);
      Vec3f pb = VoxelCenter(nb);
      Vec3f pos = pa + t * (pb - pa);

      // Interpolated normal.
      Vec3f na(UnpackSnorm16(v.nx), UnpackSnorm16(v.ny), UnpackSnorm16(v.nz));
      Vec3f nb_n(UnpackSnorm16(vn.nx), UnpackSnorm16(vn.ny),
                 UnpackSnorm16(vn.nz));
      Vec3f n = na + t * (nb_n - na);
      float nlen = n.norm();
      if (nlen > 1e-6f) {
        n /= nlen;
      }

      // Interpolated color.
      auto lerp_u8 = [](uint8_t a, uint8_t b, float t) -> uint8_t {
        return static_cast<uint8_t>(
            std::clamp(static_cast<float>(a) +
                           t * (static_cast<float>(b) - static_cast<float>(a)),
                       0.0f, 255.0f));
      };

      points->push_back(pos);
      normals->push_back(n);
      colors->push_back(Vec3<uint8_t>(
          lerp_u8(v.r, vn.r, t), lerp_u8(v.g, vn.g, t), lerp_u8(v.b, vn.b, t)));
    }
  }
}

}  // namespace dense
