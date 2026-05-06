#pragma once

// This file contains the OpenCL kernel source for PatchMatch, embedded as a C++
// raw string.  It is #included by acmmp.cc.
//
// The kernels implement the core PatchMatch stereo algorithm adapted from:
//   Xu et al., "Multi-Scale Geometric Consistency Guided and Planar Prior
//   Assisted Multi-View Stereo", TPAMI 2022.
//
// Major differences from the original CUDA code:
//   - Uses OpenCL 1.2 instead of CUDA.
//   - Uses a Philox-style PRNG instead of curand.
//   - Texture sampling is done via read_imagef on CL image objects.

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline constexpr int kMaxSources = 16;

inline const char* kPatchMatchKernelSource =
R"CL(

// Maximum number of source images the kernels support.
#define MAX_SOURCES 16

// =====================================================================
// Camera struct – matches host-side layout.
// K is row-major 3x3, R is row-major 3x3, t is 3-vector.
// =====================================================================
typedef struct {
  float K[9];
  float R[9];
  float t[3];
  int width;
  int height;
} Camera;

// =====================================================================
// Plane hypothesis: (nx, ny, nz, depth)
// =====================================================================
typedef float4 PlaneHypothesis;

// =====================================================================
// Philox-2x32-10 PRNG (stateless, counter-based).
// =====================================================================
uint2 philox2x32_round(uint2 ctr, uint key) {
  uint lo = ctr.x;
  uint hi = ctr.y;
  // Philox S-box constant
  uint product_lo = lo * 0xD256D193u;
  uint product_hi = mul_hi(lo, 0xD256D193u);
  ctr.x = hi ^ key ^ product_hi;
  ctr.y = product_lo;
  return ctr;
}

float rand_float(uint2 *state, uint key) {
  // 4 rounds of Philox (standard for GPU PRNG, matches cuRAND quality).
  // 10 rounds was overkill for PatchMatch randomisation.
  for (int i = 0; i < 4; i++) {
    *state = philox2x32_round(*state, key + (uint)i * 0x9E3779B9u);
  }
  // Convert to [0, 1)
  return (float)(state->x) / 4294967296.0f;
}

// =====================================================================
// Project a 3D point onto a camera, returning (u, v, depth).
// =====================================================================
float3 project(float3 X, __global const Camera *cam) {
  // R * X + t
  float3 Xc;
  Xc.x = cam->R[0]*X.x + cam->R[1]*X.y + cam->R[2]*X.z + cam->t[0];
  Xc.y = cam->R[3]*X.x + cam->R[4]*X.y + cam->R[5]*X.z + cam->t[1];
  Xc.z = cam->R[6]*X.x + cam->R[7]*X.y + cam->R[8]*X.z + cam->t[2];
  // K * Xc
  float3 p;
  p.x = cam->K[0]*Xc.x + cam->K[1]*Xc.y + cam->K[2]*Xc.z;
  p.y = cam->K[3]*Xc.x + cam->K[4]*Xc.y + cam->K[5]*Xc.z;
  p.z = cam->K[6]*Xc.x + cam->K[7]*Xc.y + cam->K[8]*Xc.z;
  return p;
}

// =====================================================================
// Back-project pixel (x, y) at depth d to 3D using camera 0.
// =====================================================================
float3 backproject(float x, float y, float depth,
                   __global const Camera *cam) {
  // K_inv applied via direct formulas for simplicity
  float fx = cam->K[0];
  float fy = cam->K[4];
  float cx = cam->K[2];
  float cy = cam->K[5];
  float3 pt_cam;
  pt_cam.x = depth * (x - cx) / fx;
  pt_cam.y = depth * (y - cy) / fy;
  pt_cam.z = depth;
  // R^T * (pt_cam - t)
  float3 diff;
  diff.x = pt_cam.x - cam->t[0];
  diff.y = pt_cam.y - cam->t[1];
  diff.z = pt_cam.z - cam->t[2];
  float3 pt_world;
  pt_world.x = cam->R[0]*diff.x + cam->R[3]*diff.y + cam->R[6]*diff.z;
  pt_world.y = cam->R[1]*diff.x + cam->R[4]*diff.y + cam->R[7]*diff.z;
  pt_world.z = cam->R[2]*diff.x + cam->R[5]*diff.y + cam->R[8]*diff.z;
  return pt_world;
}

// =====================================================================
// Compute the depth of a plane at pixel (x, y)
// plane = (nx, ny, nz, d) where n.dot(X) + d = 0 in camera space.
// =====================================================================
float depth_from_plane(float x, float y, PlaneHypothesis plane,
                       __global const Camera *cam) {
  float fx = cam->K[0];
  float fy = cam->K[4];
  float cx = cam->K[2];
  float cy = cam->K[5];
  float denom = plane.x * (x - cx) / fx + plane.y * (y - cy) / fy + plane.z;
  if (fabs(denom) < 1e-6f) return 0.0f;
  return -plane.w / denom;
}

// =====================================================================
// Compute plane from depth and normal at pixel (x, y).
// =====================================================================
PlaneHypothesis plane_from_depth_normal(float x, float y, float depth,
                                        float3 normal,
                                        __global const Camera *cam) {
  float fx = cam->K[0];
  float fy = cam->K[4];
  float cx = cam->K[2];
  float cy = cam->K[5];
  float3 pt;
  pt.x = depth * (x - cx) / fx;
  pt.y = depth * (y - cy) / fy;
  pt.z = depth;
  float d = -dot(normal, pt);
  return (PlaneHypothesis)(normal.x, normal.y, normal.z, d);
}

// =====================================================================
// Bilateral ZNCC matching cost between reference and source patches.
// Uses sampler-based image reads for sub-pixel interpolation.
//
// Multi-scale approach: when a patch has low variance (textureless
// region), the effective radius grows by 1.5x, 2x, 2.5x, 3x while
// keeping the same number of sample points (strided sampling).
// This captures more context on uniform surfaces without extra cost.
//
// Returns the raw NCC correlation in [-1, 1] (higher = better match).
// Returns -1.0 when the match is invalid (degenerate geometry or
// insufficient patch variance at all scales).
// =====================================================================
float compute_ncc_at_stride(
    read_only image2d_t ref_img,
    read_only image2d_t src_img,
    float ref_center,
    float src_center,
    float src_cx, float src_cy,
    int cx, int cy,
    int half_patch,
    float stride,
    float dfdx_x, float dfdx_y,
    float dfdy_x, float dfdy_y,
    float spatial_factor,
    float color_factor
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  // Center-shifted accumulation to avoid catastrophic cancellation.
  // Instead of computing var = E[X²] - E[X]² (which loses precision when
  // E[X]² ≈ E[X²], i.e. in dark or uniform regions), we shift all pixel
  // values by the center pixel: r = ref_val - ref_center, s = src_val - src_center.
  // Then var_ref = E[r²] - E[r]² is well-conditioned because E[r] ≈ 0.
  float sum_w = 0.0f;
  float sum_r = 0.0f;    // Σ w·(ref - ref_center)
  float sum_s = 0.0f;    // Σ w·(src - src_center)
  float sum_rr = 0.0f;   // Σ w·(ref - ref_center)²
  float sum_ss = 0.0f;   // Σ w·(src - src_center)²
  float sum_rs = 0.0f;   // Σ w·(ref - ref_center)·(src - src_center)

  for (int iy = -half_patch; iy <= half_patch; iy++) {
    for (int ix = -half_patch; ix <= half_patch; ix++) {
      // Strided sampling: sample at stride*ix, stride*iy in pixel space
      float fdx = stride * (float)ix;
      float fdy = stride * (float)iy;

      float rx = (float)cx + fdx;
      float ry = (float)cy + fdy;
      float ref_val = read_imagef(ref_img, samp, (float2)(rx + 0.5f, ry + 0.5f)).x;

      // Warp to source using the Jacobian (already per-pixel, scale by stride)
      float sx = src_cx + dfdx_x * fdx + dfdy_x * fdy;
      float sy = src_cy + dfdx_y * fdx + dfdy_y * fdy;
      float src_val = read_imagef(src_img, samp, (float2)(sx + 0.5f, sy + 0.5f)).x;

      // Bilateral weight: spatial × color.
      // Intensities and sigma_color are both in normalized [0,1] units.
      // Use physical pixel distance (fdx, fdy), not loop index (ix, iy),
      // so that the spatial weight decays correctly at stride > 1.
      float dist2 = fdx * fdx + fdy * fdy;
      float r = ref_val - ref_center;
      float w = exp(dist2 * spatial_factor + r * r * color_factor);
      float s = src_val - src_center;

      sum_r  += w * r;
      sum_s  += w * s;
      sum_rr += w * r * r;
      sum_ss += w * s * s;
      sum_rs += w * r * s;
      sum_w  += w;
    }
  }

  if (sum_w < 1e-6f) return -1.0f;

  float inv_w = 1.0f / sum_w;
  float mean_r = sum_r * inv_w;
  float mean_s = sum_s * inv_w;
  // var = E[(x-c)²] - E[x-c]².  Since c = center pixel, E[x-c] ≈ 0,
  // so E[x-c]² ≈ 0 and the subtraction is benign.
  float var_ref = sum_rr * inv_w - mean_r * mean_r;
  float var_src = sum_ss * inv_w - mean_s * mean_s;

  // Reference PatchMatch uses kMinVar = 1e-5 with [0,255] pixel values.
  // Our pixels are [0,1], so equivalent threshold is 1e-5 / 255² ≈ 1.5e-10.
  // With center-shifted accumulation this is numerically stable.
  if (var_ref < 1.5e-10f || var_src < 1.5e-10f) return -1.0f;

  float covar = sum_rs * inv_w - mean_r * mean_s;
  return covar / sqrt(var_ref * var_src);
}

float compute_ncc(
    read_only image2d_t ref_img,
    read_only image2d_t src_img,
    __global const Camera *ref_cam,
    __global const Camera *src_cam,
    int cx, int cy,
    PlaneHypothesis plane,
    int half_patch,
    float sigma_spatial,
    float sigma_color
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  float ref_center = read_imagef(ref_img, samp, (float2)(cx + 0.5f, cy + 0.5f)).x;

  float depth = depth_from_plane((float)cx, (float)cy, plane, ref_cam);
  if (depth <= 0.0f) return -1.0f;

  // Back-project center to world
  float3 center_world = backproject((float)cx, (float)cy, depth, ref_cam);

  // Project to source
  float3 src_proj = project(center_world, src_cam);
  if (src_proj.z < 1e-6f) return -1.0f;
  float src_cx = src_proj.x / src_proj.z;
  float src_cy = src_proj.y / src_proj.z;

  // Jacobian via finite differences for patch warping
  float depth_dx = depth_from_plane((float)(cx+1), (float)cy, plane, ref_cam);
  float depth_dy = depth_from_plane((float)cx, (float)(cy+1), plane, ref_cam);
  if (depth_dx <= 0.0f || depth_dy <= 0.0f) return -1.0f;

  float3 world_dx = backproject((float)(cx+1), (float)cy, depth_dx, ref_cam);
  float3 world_dy = backproject((float)cx, (float)(cy+1), depth_dy, ref_cam);

  float3 proj_dx = project(world_dx, src_cam);
  float3 proj_dy = project(world_dy, src_cam);
  if (proj_dx.z < 1e-6f || proj_dy.z < 1e-6f) return -1.0f;

  float dfdx_x = proj_dx.x / proj_dx.z - src_cx;
  float dfdx_y = proj_dx.y / proj_dx.z - src_cy;
  float dfdy_x = proj_dy.x / proj_dy.z - src_cx;
  float dfdy_y = proj_dy.y / proj_dy.z - src_cy;

  float spatial_factor = -1.0f / (2.0f * sigma_spatial * sigma_spatial);
  float color_factor = -1.0f / (2.0f * sigma_color * sigma_color);

  float src_center = read_imagef(src_img, samp, (float2)(src_cx + 0.5f, src_cy + 0.5f)).x;

  // Multi-scale: try progressively larger effective radii via strided
  // sampling.  Same number of samples at each scale, so cost is O(1)
  // per additional scale attempted.  Only escalate when variance is too
  // low (textureless patch).
  //
  // stride=1.0 → original patch size
  // stride=1.5 → 1.5x radius, same grid points
  // stride=2.0 → 2x radius, etc.
  // stride=2.5 → 2.5x radius
  // stride=3.0 → 3x radius (last resort)
  float strides[5] = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f};

  // Track the best NCC across all scales.  Previously this returned the
  // LAST stride's NCC, which could be worse than an earlier stride —
  // e.g. stride 1.0 gives 0.06 on a dark patch but stride 3.0 gives
  // -0.03 due to noise.  Returning the best prevents false negatives.
  float best_ncc = -1.0f;
  bool base_textureless = false;
  for (int s = 0; s < 5; s++) {
    float ncc = compute_ncc_at_stride(
        ref_img, src_img, ref_center, src_center,
        src_cx, src_cy, cx, cy, half_patch, strides[s],
        dfdx_x, dfdx_y, dfdy_x, dfdy_y,
        spatial_factor, color_factor);
    // If the base stride (1.0) has no texture, larger strides may pick
    // up signal from a different depth surface across a discontinuity.
    // Cap their NCC so the cost function falls back to Census instead.
    if (s == 0 && ncc < -0.5f) base_textureless = true;
    if (base_textureless && ncc > 0.0f) ncc = 0.0f;
    if (ncc > best_ncc) best_ncc = ncc;
    if (best_ncc > 0.1f) return best_ncc;  // good enough, done
  }

  return best_ncc;  // best across all scales
}

// =====================================================================
// Census-transform matching cost between reference and source patches.
//
// The Census transform compares each pixel in a window to the center pixel,
// producing a binary descriptor.  The Hamming distance between the reference
// and (warped) source Census descriptors gives a non-parametric matching
// cost that is robust to:
//   - Low-texture / uniform surfaces (asphalt, walls, sky)
//   - Global illumination changes
//   - Radiometric differences between views
//
// Unlike NCC, Census ALWAYS produces a valid cost because it does not
// depend on patch variance — even a perfectly uniform patch produces a
// well-defined (all-zero) descriptor.
//
// We use a 5x5 Census window (25 comparisons → 25 bits in a uint).
// The result is normalised to [0, 1] where 0 = perfect match.
// =====================================================================
float compute_census_cost(
    read_only image2d_t ref_img,
    read_only image2d_t src_img,
    __global const Camera *ref_cam,
    __global const Camera *src_cam,
    int cx, int cy,
    PlaneHypothesis plane
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  float depth = depth_from_plane((float)cx, (float)cy, plane, ref_cam);
  if (depth <= 0.0f) return 1.0f;

  // Back-project center to world and project to source
  float3 center_world = backproject((float)cx, (float)cy, depth, ref_cam);
  float3 src_proj = project(center_world, src_cam);
  if (src_proj.z < 1e-6f) return 1.0f;
  float src_cx = src_proj.x / src_proj.z;
  float src_cy = src_proj.y / src_proj.z;

  // Jacobian for patch warping (same as NCC)
  float depth_dx = depth_from_plane((float)(cx+1), (float)cy, plane, ref_cam);
  float depth_dy = depth_from_plane((float)cx, (float)(cy+1), plane, ref_cam);
  if (depth_dx <= 0.0f || depth_dy <= 0.0f) return 1.0f;

  float3 world_dx = backproject((float)(cx+1), (float)cy, depth_dx, ref_cam);
  float3 world_dy = backproject((float)cx, (float)(cy+1), depth_dy, ref_cam);
  float3 proj_dx = project(world_dx, src_cam);
  float3 proj_dy = project(world_dy, src_cam);
  if (proj_dx.z < 1e-6f || proj_dy.z < 1e-6f) return 1.0f;

  float dfdx_x = proj_dx.x / proj_dx.z - src_cx;
  float dfdx_y = proj_dx.y / proj_dx.z - src_cy;
  float dfdy_x = proj_dy.x / proj_dy.z - src_cx;
  float dfdy_y = proj_dy.y / proj_dy.z - src_cy;

  // Center pixel values
  float ref_center = read_imagef(ref_img, samp, (float2)(cx + 0.5f, cy + 0.5f)).x;
  float src_center = read_imagef(src_img, samp, (float2)(src_cx + 0.5f, src_cy + 0.5f)).x;

  // Build 5x5 Census descriptors (25 bits each)
  uint ref_census = 0u;
  uint src_census = 0u;
  int bit = 0;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) continue;  // skip center

      float rx = (float)(cx + dx);
      float ry = (float)(cy + dy);
      float ref_val = read_imagef(ref_img, samp, (float2)(rx + 0.5f, ry + 0.5f)).x;

      float sx = src_cx + dfdx_x * dx + dfdy_x * dy;
      float sy = src_cy + dfdx_y * dx + dfdy_y * dy;
      float src_val = read_imagef(src_img, samp, (float2)(sx + 0.5f, sy + 0.5f)).x;

      if (ref_val > ref_center) ref_census |= (1u << bit);
      if (src_val > src_center) src_census |= (1u << bit);
      bit++;
    }
  }

  // Hamming distance
  uint xor_val = ref_census ^ src_census;
  int hamming = popcount(xor_val);

  // Minimum contrast: if both descriptors have fewer than 3 set bits,
  // the patches are essentially uniform.  A Hamming distance of 0
  // between two all-zero descriptors is meaningless — it holds for
  // *any* depth.  Return worst-case cost to prevent false matches.
  if (popcount(ref_census) < 3 && popcount(src_census) < 3) return 1.0f;

  // Normalise to [0, 1]:  24 is max Hamming distance for 5x5-1 = 24 bits
  return (float)hamming / 24.0f;
})CL"
R"CL(

// =====================================================================
// Combined matching cost: (1-w)*NCC_cost + w*Census_cost
//
// When NCC fails (returns -1 due to low variance), we fall back entirely
// to the Census cost, which always works.  This is critical for
// textureless surfaces like asphalt, painted walls, etc.
// =====================================================================
float compute_combined_cost(
    read_only image2d_t ref_img,
    read_only image2d_t src_img,
    __global const Camera *ref_cam,
    __global const Camera *src_cam,
    int cx, int cy,
    PlaneHypothesis plane,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    float census_weight
) {
  float ncc = compute_ncc(ref_img, src_img, ref_cam, src_cam,
                          cx, cy, plane, half_patch,
                          sigma_spatial, sigma_color);

  if (ncc <= -0.5f) {
    // NCC completely failed (textureless / degenerate): Census only.
    // Census always produces a valid cost because it compares relative
    // pixel orderings, not absolute values — even a uniform patch gives
    // a well-defined (all-zero) descriptor.
    if (census_weight > 1e-6f) {
      float census = compute_census_cost(ref_img, src_img, ref_cam, src_cam,
                                         cx, cy, plane);
      return census;
    }
    return 2.0f;  // Census disabled and NCC failed
  }

  float ncc_cost = 1.0f - ncc;  // [0, 2], good match → near 0

  if (census_weight < 1e-6f) return ncc_cost;  // Census disabled

  // Adaptive blending: Census contributes only when NCC quality is poor.
  // On textured surfaces (ncc >= 0.9), Census is skipped entirely,
  // preserving NCC's sub-pixel precision and avoiding overhead.
  // On dark/low-texture surfaces (ncc ~= 0.4), Census is fully engaged
  // at the user-specified census_weight, providing robust discrimination.
  //
  // smoothstep(0.1, 0.6, ncc_cost) gives:
  //   ncc >= 0.9 (ncc_cost <= 0.1) → t = 0 → pure NCC
  //   ncc ~= 0.4 (ncc_cost ~= 0.6) → t = 1 → full census_weight blend
  float t = smoothstep(0.1f, 0.6f, ncc_cost);
  float blend = census_weight * t;

  if (blend < 1e-4f) return ncc_cost;  // fast path: skip Census entirely

  float census = compute_census_cost(ref_img, src_img, ref_cam, src_cam,
                                     cx, cy, plane);
  return (1.0f - blend) * ncc_cost + blend * census;
}

// =====================================================================
// Multi-scale NCC using pre-computed warp parameters.
//
// Takes the already-computed source projection and Jacobian, avoiding
// redundant depth_from_plane / backproject / project calls.
// Reduced from 5 to 3 scales — Census handles textureless regions,
// so intermediate strides (1.5, 2.5) are redundant.
// =====================================================================
float compute_ncc_multiscale(
    read_only image2d_t ref_img,
    read_only image2d_t src_img,
    float ref_center,
    float src_center,
    float src_cx, float src_cy,
    int cx, int cy,
    int half_patch,
    float dfdx_x, float dfdx_y,
    float dfdy_x, float dfdy_y,
    float spatial_factor,
    float color_factor
) {
  float strides[3] = {1.0f, 2.0f, 3.0f};
  float best_ncc = -1.0f;
  bool base_textureless = false;
  for (int s = 0; s < 3; s++) {
    float ncc = compute_ncc_at_stride(
        ref_img, src_img, ref_center, src_center,
        src_cx, src_cy, cx, cy, half_patch, strides[s],
        dfdx_x, dfdx_y, dfdy_x, dfdy_y,
        spatial_factor, color_factor);
    if (s == 0 && ncc < -0.5f) base_textureless = true;
    if (base_textureless && ncc > 0.0f) ncc = 0.0f;
    if (ncc > best_ncc) best_ncc = ncc;
    if (best_ncc > 0.1f) return best_ncc;
  }
  return best_ncc;
})CL"
R"CL(

// =====================================================================
// Census-transform cost using pre-computed warp parameters.
//
// Takes the already-computed source projection and Jacobian, avoiding
// redundant depth_from_plane / backproject / project calls.
// Returns normalised Hamming distance in [0, 1] where 0 = perfect match.
// =====================================================================
float compute_census_inner(
    read_only image2d_t ref_img,
    read_only image2d_t src_img,
    float ref_center,
    float src_center,
    int cx, int cy,
    float src_cx, float src_cy,
    float dfdx_x, float dfdx_y,
    float dfdy_x, float dfdy_y
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  uint ref_census = 0u;
  uint src_census = 0u;
  int bit = 0;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) continue;

      float rx = (float)(cx + dx);
      float ry = (float)(cy + dy);
      float ref_val = read_imagef(ref_img, samp, (float2)(rx + 0.5f, ry + 0.5f)).x;

      float sx = src_cx + dfdx_x * dx + dfdy_x * dy;
      float sy = src_cy + dfdx_y * dx + dfdy_y * dy;
      float src_val = read_imagef(src_img, samp, (float2)(sx + 0.5f, sy + 0.5f)).x;

      if (ref_val > ref_center) ref_census |= (1u << bit);
      if (src_val > src_center) src_census |= (1u << bit);
      bit++;
    }
  }

  uint xor_val = ref_census ^ src_census;
  int hamming = popcount(xor_val);

  // Minimum contrast: reject uniform-vs-uniform matches (see
  // compute_census_cost for rationale).
  if (popcount(ref_census) < 3 && popcount(src_census) < 3) return 1.0f;

  return (float)hamming / 24.0f;
}

// =====================================================================
// Combined NCC+Census cost for a single source image, using
// pre-computed reference-side geometry (world points).
//
// This is the core optimisation: the reference camera geometry
// (depth_from_plane, backproject at center/dx/dy) is computed ONCE
// in compute_cost_vector and shared across all source images.
// Each call here only does the source-specific projection (3 project
// calls) plus NCC/Census — eliminating ~75% of redundant geometry.
// =====================================================================
float compute_single_cost(
    read_only image2d_t ref_img,
    read_only image2d_t src_img,
    __global const Camera *src_cam,
    float3 center_world,
    float3 world_dx,
    float3 world_dy,
    float ref_center,
    int cx, int cy,
    int half_patch,
    float spatial_factor,
    float color_factor,
    float census_weight
) {
  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;

  // Project to source (only per-source work)
  float3 src_proj = project(center_world, src_cam);
  if (src_proj.z < 1e-6f) return 2.0f;
  float src_cx = src_proj.x / src_proj.z;
  float src_cy = src_proj.y / src_proj.z;

  // Jacobian (source-specific)
  float3 proj_dx = project(world_dx, src_cam);
  float3 proj_dy = project(world_dy, src_cam);
  if (proj_dx.z < 1e-6f || proj_dy.z < 1e-6f) return 2.0f;

  float dfdx_x = proj_dx.x / proj_dx.z - src_cx;
  float dfdx_y = proj_dx.y / proj_dx.z - src_cy;
  float dfdy_x = proj_dy.x / proj_dy.z - src_cx;
  float dfdy_y = proj_dy.y / proj_dy.z - src_cy;

  float src_center = read_imagef(src_img, samp,
      (float2)(src_cx + 0.5f, src_cy + 0.5f)).x;

  // NCC (multi-scale, using pre-computed warp)
  float ncc = compute_ncc_multiscale(ref_img, src_img,
      ref_center, src_center, src_cx, src_cy, cx, cy, half_patch,
      dfdx_x, dfdx_y, dfdy_x, dfdy_y, spatial_factor, color_factor);

  if (ncc <= -0.5f) {
    if (census_weight > 1e-6f) {
      return compute_census_inner(ref_img, src_img, ref_center, src_center,
          cx, cy, src_cx, src_cy, dfdx_x, dfdx_y, dfdy_x, dfdy_y);
    }
    return 2.0f;
  }

  float ncc_cost = 1.0f - ncc;
  if (census_weight < 1e-6f) return ncc_cost;

  float t = smoothstep(0.1f, 0.6f, ncc_cost);
  float blend = census_weight * t;
  if (blend < 1e-4f) return ncc_cost;

  float census = compute_census_inner(ref_img, src_img, ref_center, src_center,
      cx, cy, src_cx, src_cy, dfdx_x, dfdx_y, dfdy_x, dfdy_y);
  return (1.0f - blend) * ncc_cost + blend * census;
}

// =====================================================================
// Compute per-view cost vector: one cost per source image.
// Returns the number of valid costs written.
//
// OPTIMISED: reference-side geometry (depth_from_plane, backproject)
// is computed ONCE and shared across all source images, eliminating
// redundant geometry calls per plane hypothesis evaluation.
//
// OpenCL 1.2 does not allow dynamic indexing of image2d_t parameters,
// so all MAX_SOURCES source images are passed individually and
// dispatched via a switch statement.
// =====================================================================
int compute_cost_vector(
    read_only image2d_t ref_img,
    __global const Camera *cameras,
    int num_images,
    int x, int y,
    PlaneHypothesis plane,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    float census_weight,
    float *out_costs,
    read_only image2d_t src_img0,
    read_only image2d_t src_img1,
    read_only image2d_t src_img2,
    read_only image2d_t src_img3,
    read_only image2d_t src_img4,
    read_only image2d_t src_img5,
    read_only image2d_t src_img6,
    read_only image2d_t src_img7,
    read_only image2d_t src_img8,
    read_only image2d_t src_img9,
    read_only image2d_t src_img10,
    read_only image2d_t src_img11,
    read_only image2d_t src_img12,
    read_only image2d_t src_img13,
    read_only image2d_t src_img14,
    read_only image2d_t src_img15
) {
  int n_src = min(num_images - 1, MAX_SOURCES);

  // ---- Reference geometry: computed ONCE for all source images ----
  float depth = depth_from_plane((float)x, (float)y, plane, &cameras[0]);
  if (depth <= 0.0f) {
    for (int i = 0; i < n_src; i++) out_costs[i] = 2.0f;
    return n_src;
  }

  float depth_dx = depth_from_plane((float)(x+1), (float)y, plane, &cameras[0]);
  float depth_dy = depth_from_plane((float)x, (float)(y+1), plane, &cameras[0]);
  if (depth_dx <= 0.0f || depth_dy <= 0.0f) {
    for (int i = 0; i < n_src; i++) out_costs[i] = 2.0f;
    return n_src;
  }

  float3 center_world = backproject((float)x, (float)y, depth, &cameras[0]);
  float3 world_dx = backproject((float)(x+1), (float)y, depth_dx, &cameras[0]);
  float3 world_dy = backproject((float)x, (float)(y+1), depth_dy, &cameras[0]);

  const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE |
                         CLK_FILTER_LINEAR;
  float ref_center = read_imagef(ref_img, samp, (float2)(x + 0.5f, y + 0.5f)).x;

  float spatial_factor = -1.0f / (2.0f * sigma_spatial * sigma_spatial);
  float color_factor = -1.0f / (2.0f * sigma_color * sigma_color);

  // ---- Per-source cost: dispatch via switch (OpenCL can't index image2d_t) ----
  // Macro to avoid repeating the compute_single_cost call for each case.
  #define COST_CASE(I, IMG) \
    case I: out_costs[I] = compute_single_cost(ref_img, IMG, &cameras[I+1], \
        center_world, world_dx, world_dy, ref_center, \
        x, y, half_patch, spatial_factor, color_factor, census_weight); break;

  for (int i = 0; i < n_src; i++) {
    switch (i) {
      COST_CASE( 0, src_img0)
      COST_CASE( 1, src_img1)
      COST_CASE( 2, src_img2)
      COST_CASE( 3, src_img3)
      COST_CASE( 4, src_img4)
      COST_CASE( 5, src_img5)
      COST_CASE( 6, src_img6)
      COST_CASE( 7, src_img7)
      COST_CASE( 8, src_img8)
      COST_CASE( 9, src_img9)
      COST_CASE(10, src_img10)
      COST_CASE(11, src_img11)
      COST_CASE(12, src_img12)
      COST_CASE(13, src_img13)
      COST_CASE(14, src_img14)
      COST_CASE(15, src_img15)
    }
  }
  #undef COST_CASE
  return n_src;
})CL"
R"CL(

// =====================================================================
// Compute initial multi-view cost and selected views bitmask.
// Evaluates all source images, picks top-k, returns mean cost and
// stores a bitmask of which views were selected.
// =====================================================================
float compute_initial_cost_and_views(
    read_only image2d_t ref_img,
    __global const Camera *cameras,
    int num_images,
    int x, int y,
    PlaneHypothesis plane,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight,
    uint *out_selected,
    read_only image2d_t src_img0,
    read_only image2d_t src_img1,
    read_only image2d_t src_img2,
    read_only image2d_t src_img3,
    read_only image2d_t src_img4,
    read_only image2d_t src_img5,
    read_only image2d_t src_img6,
    read_only image2d_t src_img7,
    read_only image2d_t src_img8,
    read_only image2d_t src_img9,
    read_only image2d_t src_img10,
    read_only image2d_t src_img11,
    read_only image2d_t src_img12,
    read_only image2d_t src_img13,
    read_only image2d_t src_img14,
    read_only image2d_t src_img15
) {
  float cost_vector[MAX_SOURCES];
  float cost_copy[MAX_SOURCES];
  int n_src = min(num_images - 1, MAX_SOURCES);
  for (int i = 0; i < MAX_SOURCES; i++) {
    cost_vector[i] = 2.0f;
    cost_copy[i] = 2.0f;
  }

  compute_cost_vector(ref_img, cameras, num_images, x, y, plane,
                      half_patch, sigma_spatial, sigma_color, census_weight,
                      cost_vector,
                      src_img0, src_img1, src_img2, src_img3,
                      src_img4, src_img5, src_img6, src_img7,
                      src_img8, src_img9, src_img10, src_img11,
                      src_img12, src_img13, src_img14, src_img15);

  int num_valid = 0;
  for (int i = 0; i < n_src; i++) {
    cost_copy[i] = cost_vector[i];
    if (cost_vector[i] < 2.0f) num_valid++;
  }

  // Sort the cost_vector ascending (insertion sort)
  for (int i = 1; i < n_src; i++) {
    float tmp = cost_vector[i];
    int j = i;
    for (; j >= 1 && tmp < cost_vector[j-1]; j--)
      cost_vector[j] = cost_vector[j-1];
    cost_vector[j] = tmp;
  }

  *out_selected = 0u;
  int k = min(min(top_k, num_valid), n_src);
  if (k <= 0) return 2.0f;

  float total = 0.0f;
  for (int i = 0; i < k; i++) total += cost_vector[i];
  float threshold = cost_vector[k - 1];

  for (int i = 0; i < n_src; i++) {
    if (cost_copy[i] <= threshold) {
      *out_selected |= (1u << i);
    }
  }
  return total / (float)k;
}

// =====================================================================
// Convert PDF to CDF in-place.
// =====================================================================
void pdf_to_cdf(float *probs, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += probs[i];
  if (sum < 1e-12f) return;
  float inv = 1.0f / sum;
  float cum = 0.0f;
  for (int i = 0; i < n; i++) {
    cum += probs[i] * inv;
    probs[i] = cum;
  }
}

// =====================================================================
// Bilateral-gated smoothness cost.
//
// Measures how well a plane hypothesis agrees with its spatial
// neighbours in depth and normal orientation.  A bilateral gate on
// relative depth difference preserves discontinuities: neighbours
// at very different depths contribute near-zero weight.
//
// Samples the 4-connected + 4 diagonal neighbours (up to 8).
// Returns a cost in [0, 1] where 0 = perfectly smooth.
// =====================================================================
float compute_smoothness_cost(
    __global const PlaneHypothesis *planes,
    __global const Camera *cam,
    PlaneHypothesis cur_plane,
    int x, int y, int width, int height
) {
  float cur_depth = depth_from_plane((float)x, (float)y, cur_plane, cam);
  if (cur_depth <= 0.0f) return 0.0f;

  float3 cur_n = normalize((float3)(cur_plane.x, cur_plane.y, cur_plane.z));

  // Bilateral sigma: 2% relative depth.  Neighbours beyond this
  // relative distance are treated as belonging to a different surface.
  const float sigma_rel = 0.02f;

  // 4-connected offsets only.  Diagonal neighbours have the *same*
  // checkerboard parity, so reading them during a red/black pass is a
  // data race (they're being written concurrently by other work-items).
  const int dx[4] = {-1,  1,  0,  0};
  const int dy[4] = { 0,  0, -1,  1};

  float total_weight = 0.0f;
  float total_cost = 0.0f;

  for (int k = 0; k < 4; k++) {
    int nx = x + dx[k];
    int ny = y + dy[k];
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

    int nidx = ny * width + nx;
    PlaneHypothesis nb = planes[nidx];
    float nb_depth = depth_from_plane((float)nx, (float)ny, nb, cam);
    if (nb_depth <= 0.0f) continue;

    // Bilateral gate: exponential falloff on relative depth difference
    float rel_diff = fabs(cur_depth - nb_depth) / cur_depth;
    float w = exp(-rel_diff * rel_diff / (2.0f * sigma_rel * sigma_rel));

    // Normal disagreement: 1 - dot(n_cur, n_nb), in [0, 2]
    float3 nb_n = normalize((float3)(nb.x, nb.y, nb.z));
    float normal_cost = 1.0f - dot(cur_n, nb_n);  // 0 = same, 2 = opposite

    // Depth surface deviation: does the neighbour's depth agree with
    // what our plane predicts at the neighbour's pixel position?
    float predicted = depth_from_plane((float)nx, (float)ny, cur_plane, cam);
    float depth_dev = (predicted > 0.0f)
        ? fabs(nb_depth - predicted) / cur_depth
        : 0.0f;

    // Combined: 50% normal, 50% depth deviation, both in [0, ~1]
    float cost = 0.5f * clamp(normal_cost, 0.0f, 1.0f)
               + 0.5f * clamp(depth_dev * 10.0f, 0.0f, 1.0f);

    total_cost += w * cost;
    total_weight += w;
  }

  if (total_weight < 1e-6f) return 0.0f;
  return total_cost / total_weight;
}

// =====================================================================
// Sample a previous-scale depth map stored in a flat global buffer.
// prev_depths layout: [MAX_SOURCES * width * height] floats.
// Uses nearest-neighbor sampling (avoids blurring depth discontinuities).
// =====================================================================
float sample_prev_depth(
    __global const float *prev_depths,
    int src_idx,
    float u, float v,
    int width, int height
) {
  int ix = clamp((int)(u + 0.5f), 0, width - 1);
  int iy = clamp((int)(v + 0.5f), 0, height - 1);
  return prev_depths[src_idx * width * height + iy * width + ix];
}

// =====================================================================
// Geometric consistency cost (matches reference ComputeGeomConsistencyCost).
//
// For a plane hypothesis at pixel (px, py) in the reference camera:
//   1. Compute depth from plane, backproject to 3D world point.
//   2. Project into source camera → (src_x, src_y).
//   3. Look up previous-scale depth at (src_x, src_y) in source view.
//   4. Backproject source pixel with that depth to 3D.
//   5. Re-project to reference camera → (back_x, back_y).
//   6. Return 2D reprojection error ||p - back||.
//
// If the hypothesis is geometrically consistent with the previous depth,
// the forward-backward roundtrip returns to the original pixel (cost ≈ 0).
// =====================================================================
float compute_geom_consistency_cost(
    __global const Camera *ref_cam,
    __global const Camera *src_cam,
    PlaneHypothesis plane,
    int px, int py,
    __global const float *prev_depths,
    int src_idx,
    int pd_width, int pd_height
) {
  const float max_cost = 5.0f;

  // 1. Depth from plane at reference pixel
  float depth = depth_from_plane((float)px, (float)py, plane, ref_cam);
  if (depth <= 0.0f) return max_cost;

  // 2. Backproject to world
  float3 world_pt = backproject((float)px, (float)py, depth, ref_cam);

  // 3. Project to source camera
  float3 src_proj = project(world_pt, src_cam);
  if (src_proj.z < 1e-6f) return max_cost;
  float src_x = src_proj.x / src_proj.z;
  float src_y = src_proj.y / src_proj.z;

  // Bounds check
  if (src_x < 0.0f || src_x >= (float)pd_width ||
      src_y < 0.0f || src_y >= (float)pd_height)
    return max_cost;

  // 4. Look up previous depth at source pixel
  float src_depth = sample_prev_depth(prev_depths, src_idx,
                                      src_x, src_y, pd_width, pd_height);
  if (src_depth <= 0.0f) return max_cost;

  // 5. Backproject source pixel with previous depth to world
  float3 src_world = backproject(src_x, src_y, src_depth, src_cam);

  // 6. Re-project to reference camera
  float3 ref_proj = project(src_world, ref_cam);
  if (ref_proj.z < 1e-6f) return max_cost;
  float back_x = ref_proj.x / ref_proj.z;
  float back_y = ref_proj.y / ref_proj.z;

  // 7. 2D reprojection error
  float dx = (float)px - back_x;
  float dy = (float)py - back_y;
  return min(max_cost, sqrt(dx * dx + dy * dy));
}

// =====================================================================
// Compute view-weighted aggregate cost from a per-view cost vector.
// =====================================================================
float compute_weighted_cost(const float *cost_vec, const float *view_weights,
                            float weight_norm, int n_src) {
  if (weight_norm < 1e-6f) return 2.0f;
  float total = 0.0f;
  for (int i = 0; i < n_src; i++) {
    if (view_weights[i] > 0.0f) {
      total += view_weights[i] * cost_vec[i];
    }
  }
  return total / weight_norm;
}

// =====================================================================
// View-weighted cost with geometric consistency.
//
// For source views that have previous-scale depth maps (indicated by
// prev_depth_mask), the geometric consistency cost is added to the
// photometric cost before view-weighting, following the reference:
//   cost += view_weight * (photo + geom_weight * geom)
// =====================================================================
float compute_weighted_cost_geom(
    const float *cost_vec,
    const float *view_weights,
    float weight_norm,
    int n_src,
    __global const Camera *cameras,
    PlaneHypothesis plane,
    int x, int y,
    __global const float *prev_depths,
    uint prev_depth_mask,
    int pd_width, int pd_height,
    float geom_weight
) {
  if (weight_norm < 1e-6f) return 2.0f;
  float total = 0.0f;
  for (int i = 0; i < n_src; i++) {
    if (view_weights[i] > 0.0f) {
      float photo = cost_vec[i];
      float geom = 0.0f;
      if (geom_weight > 0.0f && (prev_depth_mask & (1u << i))) {
        geom = compute_geom_consistency_cost(
            &cameras[0], &cameras[i + 1],
            plane, x, y,
            prev_depths, i, pd_width, pd_height);
      }
      total += view_weights[i] * (photo + geom_weight * geom);
    }
  }
  return total / weight_norm;
})CL"
R"CL(

// =====================================================================
// Kernel: Random initialisation of plane hypotheses.
//
// Generates random depth + normal for each pixel, computes initial
// cost, and stores the per-pixel selected-views bitmask.
// =====================================================================
__kernel void acmmp_random_init(
    __global PlaneHypothesis *planes,
    __global float *costs,
    __global uint2 *rand_states,
    __global uint *selected_views,
    __global const Camera *cameras,
    read_only image2d_t ref_img,
    read_only image2d_t src_img0,
    read_only image2d_t src_img1,
    read_only image2d_t src_img2,
    read_only image2d_t src_img3,
    read_only image2d_t src_img4,
    read_only image2d_t src_img5,
    read_only image2d_t src_img6,
    read_only image2d_t src_img7,
    read_only image2d_t src_img8,
    read_only image2d_t src_img9,
    read_only image2d_t src_img10,
    read_only image2d_t src_img11,
    read_only image2d_t src_img12,
    read_only image2d_t src_img13,
    read_only image2d_t src_img14,
    read_only image2d_t src_img15,
    int width, int height,
    float depth_min, float depth_max,
    int num_images,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  int idx = y * width + x;

  // Initialise PRNG state from pixel coordinates
  uint2 state = (uint2)(idx * 1099087573u + 2654435769u,
                        idx * 2246822519u + 13u);
  uint key = (uint)(idx + 7919);

  // Random depth in log space
  float log_d_min = log(max(depth_min, 1e-6f));
  float log_d_max = log(max(depth_max, 1e-5f));
  float depth = exp(log_d_min + rand_float(&state, key) * (log_d_max - log_d_min));

  // Random normal (hemisphere facing camera)
  float3 normal;
  normal.x = rand_float(&state, key + 3u) * 2.0f - 1.0f;
  normal.y = rand_float(&state, key + 5u) * 2.0f - 1.0f;
  normal.z = -1.0f;
  normal = normalize(normal);

  PlaneHypothesis plane = plane_from_depth_normal(
      (float)x, (float)y, depth, normal, &cameras[0]);

  planes[idx] = plane;
  rand_states[idx] = state;

  // Compute initial cost and selected views
  uint sel = 0u;
  costs[idx] = compute_initial_cost_and_views(
      ref_img, cameras, num_images,
      x, y, plane, half_patch, sigma_spatial, sigma_color, top_k,
      census_weight, &sel,
      src_img0, src_img1, src_img2, src_img3,
      src_img4, src_img5, src_img6, src_img7,
      src_img8, src_img9, src_img10, src_img11,
      src_img12, src_img13, src_img14, src_img15);
  selected_views[idx] = sel;
}

// =====================================================================
// Kernel: Planar-prior re-initialisation (cost-matched).
//
// Re-seeds masked pixels with the SfM Delaunay prior plane (±1% depth
// perturbation) and computes the SAME total cost as acmmp_patchmatch
// (photometric + smoothness + geometric consistency).  This ensures the
// comparison against the stored cost is fair — the prior can only replace
// an existing hypothesis when it genuinely matches better.
//
// Unlike the original implementation which used compute_initial_cost_and_views
// (photometric-only, fresh top-k view selection), this version:
//   1. Reuses the existing per-pixel selected_views for cost computation.
//   2. Includes smoothness and geometric consistency terms.
//   3. Only updates selected_views if the prior hypothesis is accepted.
// =====================================================================
__kernel void acmmp_prior_reinit(
    __global PlaneHypothesis *planes,
    __global float *costs,
    __global uint2 *rand_states,
    __global uint *selected_views,
    __global const Camera *cameras,
    read_only image2d_t ref_img,
    read_only image2d_t src_img0,
    read_only image2d_t src_img1,
    read_only image2d_t src_img2,
    read_only image2d_t src_img3,
    read_only image2d_t src_img4,
    read_only image2d_t src_img5,
    read_only image2d_t src_img6,
    read_only image2d_t src_img7,
    read_only image2d_t src_img8,
    read_only image2d_t src_img9,
    read_only image2d_t src_img10,
    read_only image2d_t src_img11,
    read_only image2d_t src_img12,
    read_only image2d_t src_img13,
    read_only image2d_t src_img14,
    read_only image2d_t src_img15,
    int width, int height,
    float depth_min, float depth_max,
    int num_images,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight,
    __global const PlaneHypothesis *prior_planes,
    __global const uint *plane_masks,
    float smooth_weight,
    __global const float *prev_depths,
    uint prev_depth_mask,
    float geom_weight
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  int idx = y * width + x;

  // Only consider masked pixels (inside a Delaunay triangle).
  if (plane_masks[idx] == 0u) return;

  // Recover prior depth at this pixel.
  float depth_prior = depth_from_plane((float)x, (float)y,
                                       prior_planes[idx], &cameras[0]);
  if (depth_prior <= 0.0f || depth_prior < depth_min || depth_prior > depth_max)
    return;

  // Perturb prior depth by ±1% (reference: curand_uniform * 0.02 * d + 0.99 * d).
  uint2 state = rand_states[idx];
  uint key = (uint)(idx + 31337u);
  float depth_perturbed = depth_prior * 0.99f
      + rand_float(&state, key) * depth_prior * 0.02f;
  depth_perturbed = clamp(depth_perturbed, depth_min, depth_max);

  // Use prior normal directly.
  float3 prior_n = normalize((float3)(prior_planes[idx].x,
                                      prior_planes[idx].y,
                                      prior_planes[idx].z));

  PlaneHypothesis plane = plane_from_depth_normal(
      (float)x, (float)y, depth_perturbed, prior_n, &cameras[0]);

  // ---- Reuse the existing per-pixel view selection for cost ----
  int n_src = min(num_images - 1, MAX_SOURCES);
  uint cur_sel = selected_views[idx];

  // Derive uniform weights from the existing selection bitmask.
  float view_weights[MAX_SOURCES];
  float weight_norm = 0.0f;
  for (int i = 0; i < MAX_SOURCES; i++) {
    if (i < n_src && (cur_sel & (1u << i))) {
      view_weights[i] = 1.0f;
      weight_norm += 1.0f;
    } else {
      view_weights[i] = 0.0f;
    }
  }

  // If no views are selected yet (e.g. first iteration), fall back to
  // compute_initial_cost_and_views to bootstrap.
  if (weight_norm < 1e-6f) {
    uint sel = 0u;
    float cost = compute_initial_cost_and_views(
        ref_img, cameras, num_images,
        x, y, plane, half_patch, sigma_spatial, sigma_color, top_k,
        census_weight, &sel,
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);

    if (smooth_weight > 1e-6f) {
      cost += smooth_weight * compute_smoothness_cost(planes, &cameras[0],
          plane, x, y, width, height);
    }

    if (cost < costs[idx]) {
      planes[idx] = plane;
      costs[idx] = cost;
      selected_views[idx] = sel;
    }
    rand_states[idx] = state;
    return;
  }

  // ---- Compute the same total cost as acmmp_patchmatch ----
  float cost_vec[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) cost_vec[i] = 2.0f;
  compute_cost_vector(ref_img, cameras, num_images, x, y, plane,
      half_patch, sigma_spatial, sigma_color, census_weight,
      cost_vec,
      src_img0, src_img1, src_img2, src_img3,
      src_img4, src_img5, src_img6, src_img7,
      src_img8, src_img9, src_img10, src_img11,
      src_img12, src_img13, src_img14, src_img15);

  float cost = compute_weighted_cost_geom(cost_vec, view_weights, weight_norm,
      n_src, cameras, plane, x, y,
      prev_depths, prev_depth_mask, width, height, geom_weight);

  if (smooth_weight > 1e-6f) {
    cost += smooth_weight * compute_smoothness_cost(planes, &cameras[0],
        plane, x, y, width, height);
  }

  // Only replace the existing hypothesis if the prior-based plane
  // actually matches better under the SAME cost function.
  if (cost < costs[idx]) {
    planes[idx] = plane;
    costs[idx] = cost;
    // Keep existing view selection — prior only changes depth/normal.
  }
  rand_states[idx] = state;
}

// =====================================================================
// Kernel: Adaptive Checkerboard PatchMatch iteration.
//
// Implements the PatchMatch propagation from Xu et al. (TPAMI 2022):
//   1. Adaptive checkerboard sampling: 8 candidate positions
//      (4 near + 4 far) with long-range search.
//   2. Multi-hypothesis joint view selection: per-pixel view weights
//      via CDF-based sampling informed by neighbor view selections.
//   3. View-weighted cost aggregation: accept best candidate.
//   4. Multi-hypothesis refinement with 5 diverse plane candidates.
//
// color_flag: 0 = red pixels (x+y even), 1 = black pixels (x+y odd).
// =====================================================================
__kernel void acmmp_patchmatch(
    __global PlaneHypothesis *planes,
    __global float *costs,
    __global uint2 *rand_states,
    __global uint *selected_views,
    __global const Camera *cameras,
    read_only image2d_t ref_img,
    read_only image2d_t src_img0,
    read_only image2d_t src_img1,
    read_only image2d_t src_img2,
    read_only image2d_t src_img3,
    read_only image2d_t src_img4,
    read_only image2d_t src_img5,
    read_only image2d_t src_img6,
    read_only image2d_t src_img7,
    read_only image2d_t src_img8,
    read_only image2d_t src_img9,
    read_only image2d_t src_img10,
    read_only image2d_t src_img11,
    read_only image2d_t src_img12,
    read_only image2d_t src_img13,
    read_only image2d_t src_img14,
    read_only image2d_t src_img15,
    int width, int height,
    float depth_min, float depth_max,
    int num_images,
    int half_patch,
    float sigma_spatial,
    float sigma_color,
    int top_k,
    float census_weight,
    int color_flag,
    int iteration,
    float smooth_weight,
    __global const float *prev_depths,
    uint prev_depth_mask,
    float geom_weight,
    __global const PlaneHypothesis *prior_planes,
    __global const uint *plane_masks
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;
  if (((x + y) & 1) != color_flag) return;

  int idx = y * width + x;
  uint2 state = rand_states[idx];
  uint key = (uint)(idx + iteration * 7919 + 42);
  int n_src = min(num_images - 1, MAX_SOURCES);
  int npix = width * height;

  // =================================================================
  // Step 1: Identify 8 candidate neighbor positions (adaptive search)
  // =================================================================
  // Layout: 0=near_up, 1=far_up, 2=near_down, 3=far_down,
  //         4=near_left, 5=far_left, 6=near_right, 7=far_right
  int positions[8];
  bool flags[8];
  float cost_array[8 * MAX_SOURCES];  // cost_array[cand * MAX_SOURCES + view]
  for (int i = 0; i < 8; i++) { positions[i] = idx; flags[i] = false; }
  for (int i = 0; i < 8 * MAX_SOURCES; i++) cost_array[i] = 2.0f;

  // --- Far up (start at y-3, search every 2 rows up to 23 rows) ---
  if (y > 2) {
    int base = (y - 3) * width + x;
    float best_c = costs[base]; int best_p = base;
    for (int i = 1; i < 11; i++) {
      int yy = y - 3 - 2 * i;
      if (yy < 0) break;
      int pp = yy * width + x;
      if (costs[pp] < best_c) { best_c = costs[pp]; best_p = pp; }
    }
    positions[1] = best_p; flags[1] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[1 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }

  // --- Far down ---
  if (y < height - 3) {
    int base = (y + 3) * width + x;
    float best_c = costs[base]; int best_p = base;
    for (int i = 1; i < 11; i++) {
      int yy = y + 3 + 2 * i;
      if (yy >= height) break;
      int pp = yy * width + x;
      if (costs[pp] < best_c) { best_c = costs[pp]; best_p = pp; }
    }
    positions[3] = best_p; flags[3] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[3 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }

  // --- Far left ---
  if (x > 2) {
    int base = y * width + (x - 3);
    float best_c = costs[base]; int best_p = base;
    for (int i = 1; i < 11; i++) {
      int xx = x - 3 - 2 * i;
      if (xx < 0) break;
      int pp = y * width + xx;
      if (costs[pp] < best_c) { best_c = costs[pp]; best_p = pp; }
    }
    positions[5] = best_p; flags[5] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[5 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }

  // --- Far right ---
  if (x < width - 3) {
    int base = y * width + (x + 3);
    float best_c = costs[base]; int best_p = base;
    for (int i = 1; i < 11; i++) {
      int xx = x + 3 + 2 * i;
      if (xx >= width) break;
      int pp = y * width + xx;
      if (costs[pp] < best_c) { best_c = costs[pp]; best_p = pp; }
    }
    positions[7] = best_p; flags[7] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[7 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }

  // --- Near up (with diagonal V-shape search) ---
  if (y > 0) {
    int base = (y - 1) * width + x;
    float best_c = costs[base]; int best_p = base;
    for (int i = 0; i < 3; i++) {
      if (y > 1 + i && x > i) {
        int pp = (y - 2 - i) * width + (x - i - 1);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
      if (y > 1 + i && x < width - 1 - i) {
        int pp = (y - 2 - i) * width + (x + i + 1);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
    }
    positions[0] = best_p; flags[0] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[0 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }
)CL"
R"CL(

  // --- Near down ---
  if (y < height - 1) {
    int base = (y + 1) * width + x;
    float best_c = costs[base]; int best_p = base;
    for (int i = 0; i < 3; i++) {
      if (y < height - 2 - i && x > i) {
        int pp = (y + 2 + i) * width + (x - i - 1);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
      if (y < height - 2 - i && x < width - 1 - i) {
        int pp = (y + 2 + i) * width + (x + i + 1);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
    }
    positions[2] = best_p; flags[2] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[2 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }

  // --- Near left ---
  if (x > 0) {
    int base = y * width + (x - 1);
    float best_c = costs[base]; int best_p = base;
    for (int i = 0; i < 3; i++) {
      if (x > 1 + i && y > i) {
        int pp = (y - i - 1) * width + (x - 2 - i);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
      if (x > 1 + i && y < height - 1 - i) {
        int pp = (y + i + 1) * width + (x - 2 - i);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
    }
    positions[4] = best_p; flags[4] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[4 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }

  // --- Near right ---
  if (x < width - 1) {
    int base = y * width + (x + 1);
    float best_c = costs[base]; int best_p = base;
    for (int i = 0; i < 3; i++) {
      if (x < width - 2 - i && y > i) {
        int pp = (y - i - 1) * width + (x + 2 + i);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
      if (x < width - 2 - i && y < height - 1 - i) {
        int pp = (y + i + 1) * width + (x + 2 + i);
        if (pp >= 0 && pp < npix && costs[pp] < best_c)
          { best_c = costs[pp]; best_p = pp; }
      }
    }
    positions[6] = best_p; flags[6] = true;
    compute_cost_vector(ref_img, cameras, num_images, x, y,
        planes[best_p], half_patch, sigma_spatial, sigma_color, census_weight,
        &cost_array[6 * MAX_SOURCES],
        src_img0, src_img1, src_img2, src_img3,
        src_img4, src_img5, src_img6, src_img7,
        src_img8, src_img9, src_img10, src_img11,
        src_img12, src_img13, src_img14, src_img15);
  }

  // =================================================================
  // Step 2: Multi-hypothesis Joint View Selection
  // =================================================================

  // 2a. Build view selection priors from 4 direct neighbors
  float view_selection_priors[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) view_selection_priors[i] = 0.0f;
  int neighbor_pos[4] = {idx - width, idx + width, idx - 1, idx + 1};
  for (int i = 0; i < 4; i++) {
    int np = neighbor_pos[i];
    // Map neighbor to the near candidate flag index: up=0, down=2, left=4, right=6
    bool neighbor_valid = flags[2 * i];
    if (neighbor_valid && np >= 0 && np < npix) {
      uint sv = selected_views[np];
      for (int j = 0; j < n_src; j++) {
        if (sv & (1u << j))
          view_selection_priors[j] = 0.9f;
        else
          view_selection_priors[j] = 0.1f;
      }
    }
  }

  // 2b. Compute sampling probabilities from cost_array
  float cost_threshold = 0.8f * exp((float)(iteration * iteration) / (-90.0f));
  float sampling_probs[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) sampling_probs[i] = 0.0f;
  for (int i = 0; i < n_src; i++) {
    float count = 0.0f;
    int count_false = 0;
    float tmpw = 0.0f;
    for (int j = 0; j < 8; j++) {
      float cv = cost_array[j * MAX_SOURCES + i];
      if (flags[j] && cv < cost_threshold) {
        tmpw += exp(cv * cv / (-0.18f));
        count += 1.0f;
      }
      if (flags[j] && cv > 1.2f) {
        count_false++;
      }
    }
    if (count > 2.0f && count_false < 3)
      sampling_probs[i] = tmpw / count;
    else if (count_false < 3)
      sampling_probs[i] = exp(cost_threshold * cost_threshold / (-0.32f));
    sampling_probs[i] *= view_selection_priors[i];
  }

  // 2c. Convert to CDF and draw 15 weighted samples
  pdf_to_cdf(sampling_probs, n_src);

  float view_weights[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) view_weights[i] = 0.0f;
  for (int sample = 0; sample < 15; sample++) {
    float r = rand_float(&state, key + sample * 997u) - 1e-7f;
    for (int j = 0; j < n_src; j++) {
      if (sampling_probs[j] > r) {
        view_weights[j] += 1.0f;
        break;
      }
    }
  }

  uint new_selected = 0u;
  float weight_norm = 0.0f;
  for (int i = 0; i < n_src; i++) {
)CL"
R"CL(
    if (view_weights[i] > 0.0f) {
      new_selected |= (1u << i);
      weight_norm += view_weights[i];
    }
  }

  // =================================================================
  // Step 3: Compute view-weighted final costs for each candidate
  // =================================================================
  float final_costs[8];
  for (int i = 0; i < 8; i++) {
    if (flags[i]) {
      float photo = compute_weighted_cost_geom(&cost_array[i * MAX_SOURCES],
          view_weights, weight_norm, n_src,
          cameras, planes[positions[i]], x, y,
          prev_depths, prev_depth_mask, width, height, geom_weight);
      if (smooth_weight > 1e-6f) {
        float smooth = compute_smoothness_cost(planes, &cameras[0],
            planes[positions[i]], x, y, width, height);
        final_costs[i] = photo + smooth_weight * smooth;
      } else {
        final_costs[i] = photo;
      }
    } else {
      final_costs[i] = 2.0f;
    }
  }

  // Find best candidate
  int best_cand = 0;
  float best_cand_cost = final_costs[0];
  for (int i = 1; i < 8; i++) {
    if (final_costs[i] < best_cand_cost) {
      best_cand_cost = final_costs[i];
      best_cand = i;
    }
  }

  // Compute current pixel's cost with new view weights
  float cost_vec_now[MAX_SOURCES];
  for (int i = 0; i < MAX_SOURCES; i++) cost_vec_now[i] = 2.0f;
  compute_cost_vector(ref_img, cameras, num_images, x, y,
      planes[idx], half_patch, sigma_spatial, sigma_color, census_weight,
      cost_vec_now,
      src_img0, src_img1, src_img2, src_img3,
      src_img4, src_img5, src_img6, src_img7,
      src_img8, src_img9, src_img10, src_img11,
      src_img12, src_img13, src_img14, src_img15);
  float cost_now = compute_weighted_cost_geom(cost_vec_now, view_weights,
      weight_norm, n_src,
      cameras, planes[idx], x, y,
      prev_depths, prev_depth_mask, width, height, geom_weight);
  if (smooth_weight > 1e-6f) {
    cost_now += smooth_weight * compute_smoothness_cost(planes, &cameras[0],
        planes[idx], x, y, width, height);
  }
  costs[idx] = cost_now;

  PlaneHypothesis plane_now = planes[idx];
  float depth_now = depth_from_plane((float)x, (float)y, plane_now, &cameras[0]);

  // =================================================================
  // Planar prior setup (hypothesis generation only)
  //
  // The prior provides diverse starting hypotheses near the Delaunay
  // surface (step 4), but acceptance is always pure photometric cost.
  // =================================================================
  bool have_prior = (plane_masks[idx] > 0u);
  float depth_prior = 0.0f;
  float3 prior_n = (float3)(0.0f, 0.0f, -1.0f);
  float depth_sigma = (depth_max - depth_min) / 32.0f;
  float angle_sigma_rad = M_PI_F * (5.0f / 180.0f);  // for hypothesis generation

  if (have_prior) {
    depth_prior = depth_from_plane((float)x, (float)y, prior_planes[idx], &cameras[0]);
    prior_n = normalize((float3)(prior_planes[idx].x, prior_planes[idx].y, prior_planes[idx].z));
  }

  // =================================================================
  // Step 3b: Accept best candidate (pure photometric cost)
  // =================================================================
  {
    if (flags[best_cand] && best_cand_cost < cost_now) {
      float db = depth_from_plane((float)x, (float)y,
                                  planes[positions[best_cand]], &cameras[0]);
      if (db >= depth_min && db <= depth_max) {
        depth_now = db;
        plane_now = planes[positions[best_cand]];
        cost_now = best_cand_cost;
        selected_views[idx] = new_selected;
      }
    }
  }

  // =================================================================
  // Step 4: Multi-hypothesis refinement (5 diverse candidates)
  //
  // From Xu et al.: test 5 combinations of depth and normal:
  //   0: (random_depth,     current_normal)
  //   1: (current_depth,    random_normal)
  //   2: (random_depth,     random_normal)
  //   3: (current_depth,    perturbed_normal)
  //   4: (perturbed_depth,  current_normal)
  //
  // =================================================================
  {
    float perturbation = 0.02f;
    float3 cur_normal = normalize((float3)(plane_now.x, plane_now.y, plane_now.z));

    float depth_rand, depth_perturbed;
    float3 normal_rand, normal_perturbed;

    // Generate random depth + normal.
    // When a prior exists, draw from a narrow Gaussian around the prior
    // (ACMMP: depth in [prior - 3*sigma, prior + 3*sigma], normal
    // perturbed from prior by angle_sigma = 5 degrees).
    if (have_prior && depth_prior > 0.0f) {
      depth_rand = depth_prior - 3.0f * depth_sigma
                 + rand_float(&state, key + 503u) * 6.0f * depth_sigma;
      depth_rand = clamp(depth_rand, depth_min, depth_max);
      float pa1 = (rand_float(&state, key + 504u) - 0.5f) * angle_sigma_rad;
      float pa2 = (rand_float(&state, key + 505u) - 0.5f) * angle_sigma_rad;
      normal_rand.x = prior_n.x + pa1;
      normal_rand.y = prior_n.y + pa2;
      normal_rand.z = prior_n.z;
      normal_rand = normalize(normal_rand);
      if (normal_rand.z > 0.0f) normal_rand.z = -normal_rand.z;
    } else {
      depth_rand = exp(log(max(depth_min, 1e-6f))
                   + rand_float(&state, key + 503u)
                   * (log(max(depth_max, 1e-5f)) - log(max(depth_min, 1e-6f))));
      normal_rand.x = rand_float(&state, key + 504u) * 2.0f - 1.0f;
      normal_rand.y = rand_float(&state, key + 505u) * 2.0f - 1.0f;
      normal_rand.z = -1.0f;
      normal_rand = normalize(normal_rand);
    }

    // Perturbed depth and normal (small perturbation of current)
    {
      float d_min_p = (1.0f - perturbation) * depth_now;
      float d_max_p = (1.0f + perturbation) * depth_now;
      depth_perturbed = d_min_p + rand_float(&state, key + 510u) * (d_max_p - d_min_p);
      depth_perturbed = clamp(depth_perturbed, depth_min, depth_max);
    }
    {
      float pa1 = (rand_float(&state, key + 520u) - 0.5f) * perturbation * M_PI_F;
      float pa2 = (rand_float(&state, key + 521u) - 0.5f) * perturbation * M_PI_F;
      normal_perturbed.x = cur_normal.x + pa1;
      normal_perturbed.y = cur_normal.y + pa2;
      normal_perturbed.z = cur_normal.z;
      normal_perturbed = normalize(normal_perturbed);
      if (normal_perturbed.z > 0.0f) normal_perturbed.z = -normal_perturbed.z;
    }

    // 5 candidate hypotheses
    float  ref_depths[5]  = {depth_rand, depth_now, depth_rand,
                             depth_now, depth_perturbed};
    float3 ref_normals[5];
    ref_normals[0] = cur_normal;
    ref_normals[1] = normal_rand;
    ref_normals[2] = normal_rand;
    ref_normals[3] = normal_perturbed;
    ref_normals[4] = cur_normal;

    for (int h = 0; h < 5; h++) {
      PlaneHypothesis cand = plane_from_depth_normal(
          (float)x, (float)y, ref_depths[h], ref_normals[h], &cameras[0]);

      float cv[MAX_SOURCES];
      for (int ci = 0; ci < MAX_SOURCES; ci++) cv[ci] = 2.0f;
      compute_cost_vector(ref_img, cameras, num_images, x, y, cand,
          half_patch, sigma_spatial, sigma_color, census_weight,
          cv,
          src_img0, src_img1, src_img2, src_img3,
          src_img4, src_img5, src_img6, src_img7,
          src_img8, src_img9, src_img10, src_img11,
          src_img12, src_img13, src_img14, src_img15);
      float temp_cost = compute_weighted_cost_geom(cv, view_weights,
          weight_norm, n_src,
          cameras, cand, x, y,
          prev_depths, prev_depth_mask, width, height, geom_weight);
      if (smooth_weight > 1e-6f) {
        temp_cost += smooth_weight * compute_smoothness_cost(planes, &cameras[0],
            cand, x, y, width, height);
      }

      float depth_before = depth_from_plane((float)x, (float)y, cand, &cameras[0]);
      if (depth_before < depth_min || depth_before > depth_max) continue;

      if (temp_cost < cost_now) {
        depth_now = depth_before;
        plane_now = cand;
        cost_now = temp_cost;
      }
    }
  }

  planes[idx] = plane_now;
  costs[idx] = cost_now;
  rand_states[idx] = state;
})CL"
R"CL(

// =====================================================================
// Kernel: Upsample plane hypotheses from a coarser resolution.
//
// For valid coarse pixels, the plane is recomputed at the fine pixel
// position and the cost is set to 2.0 so PatchMatch re-evaluates it
// at the fine resolution (coarse-level NCC costs are artificially low
// due to less high-frequency detail).
//
// For invalid coarse pixels (depth <= 0), a random plane is generated
// instead — otherwise those pixels would start with garbage data.
// =====================================================================
__kernel void acmmp_upsample(
    __global PlaneHypothesis *dst_planes,
    __global float *dst_costs,
    __global const PlaneHypothesis *src_planes,
    __global const float *src_costs,
    __global const Camera *cameras,
    __global uint2 *rand_states,
    int dst_width, int dst_height,
    int src_width, int src_height,
    float depth_min, float depth_max
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= dst_width || y >= dst_height) return;

  int dst_idx = y * dst_width + x;

  // Always initialise PRNG state for every pixel so that subsequent
  // PatchMatch iterations have well-defined random numbers.  Without
  // this, valid-upsampled pixels would keep whatever garbage was in
  // the (newly allocated) rand_states buffer.
  uint2 state = (uint2)(dst_idx * 1099087573u + 2654435769u,
                        dst_idx * 2246822519u + 13u);
  rand_states[dst_idx] = state;

  // Find corresponding coarse pixel (nearest-neighbor)
  float scale_x = (float)src_width / (float)dst_width;
  float scale_y = (float)src_height / (float)dst_height;
  int sx = (int)((float)x * scale_x);
  int sy = (int)((float)y * scale_y);
  sx = clamp(sx, 0, src_width - 1);
  sy = clamp(sy, 0, src_height - 1);
  int src_idx = sy * src_width + sx;

  PlaneHypothesis plane = src_planes[src_idx];

  // Recompute depth at the high-res pixel position
  float depth = depth_from_plane((float)x, (float)y, plane, &cameras[0]);

  if (depth > 0.0f) {
    // Valid coarse pixel: upsample the plane
    float3 normal = normalize((float3)(plane.x, plane.y, plane.z));
    dst_planes[dst_idx] = plane_from_depth_normal(
        (float)x, (float)y, depth, normal, &cameras[0]);
  } else {
    // Invalid coarse pixel: random initialisation
    uint key = (uint)(dst_idx + 7919);
    float log_d_min = log(max(depth_min, 1e-6f));
    float log_d_max = log(max(depth_max, 1e-5f));
    float rand_depth = exp(log_d_min + rand_float(&state, key) * (log_d_max - log_d_min));
    float3 normal;
    normal.x = rand_float(&state, key + 3u) * 2.0f - 1.0f;
    normal.y = rand_float(&state, key + 5u) * 2.0f - 1.0f;
    normal.z = -1.0f;
    normal = normalize(normal);
    dst_planes[dst_idx] = plane_from_depth_normal(
        (float)x, (float)y, rand_depth, normal, &cameras[0]);
    rand_states[dst_idx] = state;
  }

  // Always force PatchMatch to re-evaluate at the fine resolution.
  // Coarse-level NCC costs are artificially low and would prevent
  // PatchMatch from replacing wrong hypotheses.
  dst_costs[dst_idx] = 2.0f;
}

// =====================================================================
// Helper: insertion sort for small float arrays.
// =====================================================================
void sort_small_f(float *d, int n) {
  for (int i = 1; i < n; i++) {
    float tmp = d[i];
    int j;
    for (j = i; j >= 1 && tmp < d[j - 1]; j--)
      d[j] = d[j - 1];
    d[j] = tmp;
  }
}

// =====================================================================
// Kernel: Red-black checkerboard median filter (21 neighbours).
//
// After each PatchMatch red-black iteration, smooth depth estimates by
// replacing each pixel's depth with the median of up to 21 neighbours
// sampled in a cross + diagonal checkerboard pattern (all neighbours
// are of the opposite colour, so there are no read-write conflicts
// within a single colour pass).
//
// Ported from CheckerboardFilter in Xu et al. (TPAMI 2022).
// color_flag: 0 = red (x+y even), 1 = black (x+y odd).
// =====================================================================
__kernel void acmmp_checkerboard_filter(
    __global PlaneHypothesis *planes,
    __global float *costs,
    __global const Camera *cameras,
    int width, int height,
    int color_flag
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;
  if (((x + y) & 1) != color_flag) return;

  int center = y * width + x;

  // Skip pixels with very low cost (already well converged).
  if (costs[center] < 0.001f) return;

  float filter_buf[21];
  int idx = 0;

  // Center pixel depth.
  filter_buf[idx++] = depth_from_plane((float)x, (float)y,
                                       planes[center], &cameras[0]);

  // --- Vertical axis (stride 1, 3, 5 in y) ---
  if (y > 0) {
    int nb = center - width;
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y - 1),
                                         planes[nb], &cameras[0]);
  }
  if (y > 2) {
    int nb = center - 3 * width;
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y - 3),
                                         planes[nb], &cameras[0]);
  }
  if (y > 4) {
    int nb = center - 5 * width;
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y - 5),
                                         planes[nb], &cameras[0]);
  }
  if (y < height - 1) {
    int nb = center + width;
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y + 1),
                                         planes[nb], &cameras[0]);
  }
  if (y < height - 3) {
    int nb = center + 3 * width;
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y + 3),
                                         planes[nb], &cameras[0]);
  }
  if (y < height - 5) {
    int nb = center + 5 * width;
    filter_buf[idx++] = depth_from_plane((float)x, (float)(y + 5),
                                         planes[nb], &cameras[0]);
  }

  // --- Horizontal axis (stride 1, 3, 5 in x) ---
  if (x > 0) {
    int nb = center - 1;
    filter_buf[idx++] = depth_from_plane((float)(x - 1), (float)y,
                                         planes[nb], &cameras[0]);
  }
  if (x > 2) {
    int nb = center - 3;
    filter_buf[idx++] = depth_from_plane((float)(x - 3), (float)y,
                                         planes[nb], &cameras[0]);
  }
  if (x > 4) {
    int nb = center - 5;
    filter_buf[idx++] = depth_from_plane((float)(x - 5), (float)y,
                                         planes[nb], &cameras[0]);
  }
  if (x < width - 1) {
    int nb = center + 1;
    filter_buf[idx++] = depth_from_plane((float)(x + 1), (float)y,
                                         planes[nb], &cameras[0]);
  }
  if (x < width - 3) {
    int nb = center + 3;
    filter_buf[idx++] = depth_from_plane((float)(x + 3), (float)y,
                                         planes[nb], &cameras[0]);
  }
  if (x < width - 5) {
    int nb = center + 5;
    filter_buf[idx++] = depth_from_plane((float)(x + 5), (float)y,
                                         planes[nb], &cameras[0]);
  }

  // --- Diagonal-ish neighbours (checkerboard L-shaped offsets) ---
  if (y > 0 && x < width - 2) {
    int nb = center - width + 2;
    filter_buf[idx++] = depth_from_plane((float)(x + 2), (float)(y - 1),
                                         planes[nb], &cameras[0]);
  }
  if (y < height - 1 && x < width - 2) {
    int nb = center + width + 2;
    filter_buf[idx++] = depth_from_plane((float)(x + 2), (float)(y + 1),
                                         planes[nb], &cameras[0]);
  }
  if (y > 0 && x > 1) {
    int nb = center - width - 2;
    filter_buf[idx++] = depth_from_plane((float)(x - 2), (float)(y - 1),
                                         planes[nb], &cameras[0]);
  }
  if (y < height - 1 && x > 1) {
    int nb = center + width - 2;
    filter_buf[idx++] = depth_from_plane((float)(x - 2), (float)(y + 1),
                                         planes[nb], &cameras[0]);
  }
  if (x > 0 && y > 2) {
    int nb = center - 1 - 2 * width;
    filter_buf[idx++] = depth_from_plane((float)(x - 1), (float)(y - 2),
                                         planes[nb], &cameras[0]);
  }
  if (x < width - 1 && y > 2) {
    int nb = center + 1 - 2 * width;
    filter_buf[idx++] = depth_from_plane((float)(x + 1), (float)(y - 2),
                                         planes[nb], &cameras[0]);
  }
  if (x > 0 && y < height - 2) {
    int nb = center - 1 + 2 * width;
    filter_buf[idx++] = depth_from_plane((float)(x - 1), (float)(y + 2),
                                         planes[nb], &cameras[0]);
  }
  if (x < width - 1 && y < height - 2) {
    int nb = center + 1 + 2 * width;
    filter_buf[idx++] = depth_from_plane((float)(x + 1), (float)(y + 2),
                                         planes[nb], &cameras[0]);
  }

  // Bilateral depth gate: reject neighbors whose depth differs by more
  // than 5% from the center pixel.  At foreground-background transitions
  // this prevents the median from mixing depths from different surfaces,
  // which would otherwise crystallize wrong-depth blobs.
  {
    float percent = 0.15f;  // 1.5% depth difference allowed
    float center_d = filter_buf[0];
    if (center_d > 0.0f) {
      float depth_lo = center_d * (1 - percent);
      float depth_hi = center_d * (1 + percent);
      int new_idx = 1;  // always keep center
      for (int i = 1; i < idx; i++) {
        if (filter_buf[i] > depth_lo && filter_buf[i] < depth_hi) {
          filter_buf[new_idx++] = filter_buf[i];
        }
      }
      idx = new_idx;
    }
  }

  // Insertion-sort and pick median.
  sort_small_f(filter_buf, idx);
  int mid = idx / 2;
  float median_depth;
  if (idx % 2 == 0) {
    median_depth = (filter_buf[mid - 1] + filter_buf[mid]) * 0.5f;
  } else {
    median_depth = filter_buf[mid];
  }

  // Reconstruct plane with center's normal but the median depth.
  if (median_depth > 0.0f) {
    float3 normal = normalize((float3)(planes[center].x,
                                       planes[center].y,
                                       planes[center].z));
    planes[center] = plane_from_depth_normal(
        (float)x, (float)y, median_depth, normal, &cameras[0]);
  }
}
)CL";

// =====================================================================
// Depthmap cleaning kernel: cross-view depth consistency check.
//
// For each pixel in the reference depth, backproject to 3D, project
// into each neighbor view, and check depth agreement.  Pixels without
// enough consistent neighbor views are zeroed out.
//
// The neighbor depth maps are stored as a flat buffer:
//   clean_depths[view_idx * width * height + y * width + x]
// where view_idx 0 = reference, 1..N-1 = neighbors.
// =====================================================================
inline const char* kCleanKernelSource = R"CL(

#define MAX_CLEAN_SOURCES 16

typedef struct {
  float K[9];
  float R[9];
  float t[3];
  int width;
  int height;
} Camera;

// Nearest-neighbor sampler for the reference depth (pixel-aligned reads).
__constant sampler_t samp_nn = CLK_NORMALIZED_COORDS_FALSE
                             | CLK_ADDRESS_CLAMP_TO_EDGE
                             | CLK_FILTER_NEAREST;

// Bilinear sampler for neighbor depth (sub-pixel reprojection).
__constant sampler_t samp_lin = CLK_NORMALIZED_COORDS_FALSE
                              | CLK_ADDRESS_CLAMP_TO_EDGE
                              | CLK_FILTER_LINEAR;

// Read neighbor depth from image2d_t by source index (0-based).
// Uses bilinear sampling with +0.5 pixel-center offset.
float read_src_depth(int src_idx, float u, float v,
                     read_only image2d_t s0,  read_only image2d_t s1,
                     read_only image2d_t s2,  read_only image2d_t s3,
                     read_only image2d_t s4,  read_only image2d_t s5,
                     read_only image2d_t s6,  read_only image2d_t s7,
                     read_only image2d_t s8,  read_only image2d_t s9,
                     read_only image2d_t s10, read_only image2d_t s11,
                     read_only image2d_t s12, read_only image2d_t s13,
                     read_only image2d_t s14, read_only image2d_t s15) {
  float2 coord = (float2)(u + 0.5f, v + 0.5f);
  switch (src_idx) {
    case  0: return read_imagef(s0,  samp_lin, coord).x;
    case  1: return read_imagef(s1,  samp_lin, coord).x;
    case  2: return read_imagef(s2,  samp_lin, coord).x;
    case  3: return read_imagef(s3,  samp_lin, coord).x;
    case  4: return read_imagef(s4,  samp_lin, coord).x;
    case  5: return read_imagef(s5,  samp_lin, coord).x;
    case  6: return read_imagef(s6,  samp_lin, coord).x;
    case  7: return read_imagef(s7,  samp_lin, coord).x;
    case  8: return read_imagef(s8,  samp_lin, coord).x;
    case  9: return read_imagef(s9,  samp_lin, coord).x;
    case 10: return read_imagef(s10, samp_lin, coord).x;
    case 11: return read_imagef(s11, samp_lin, coord).x;
    case 12: return read_imagef(s12, samp_lin, coord).x;
    case 13: return read_imagef(s13, samp_lin, coord).x;
    case 14: return read_imagef(s14, samp_lin, coord).x;
    default: return read_imagef(s15, samp_lin, coord).x;
  }
}

__kernel void acmmp_clean_depthmap(
    read_only image2d_t ref_depth_img,
    read_only image2d_t src0,  read_only image2d_t src1,
    read_only image2d_t src2,  read_only image2d_t src3,
    read_only image2d_t src4,  read_only image2d_t src5,
    read_only image2d_t src6,  read_only image2d_t src7,
    read_only image2d_t src8,  read_only image2d_t src9,
    read_only image2d_t src10, read_only image2d_t src11,
    read_only image2d_t src12, read_only image2d_t src13,
    read_only image2d_t src14, read_only image2d_t src15,
    __global const Camera *cameras,
    __global float *clean_depth,
    int width, int height,
    int num_views,
    float same_depth_threshold,
    int min_consistent_views,
    float carving_threshold,
    int max_carved_views
) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  int idx = y * width + x;

  float ref_depth = read_imagef(ref_depth_img, samp_nn, (int2)(x, y)).x;
  if (!(ref_depth > 0.0f)) {
    clean_depth[idx] = 0.0f;
    return;
  }

  Camera ref_cam = cameras[0];
  float fx = ref_cam.K[0], cx = ref_cam.K[2];
  float fy = ref_cam.K[4], cy = ref_cam.K[5];

  float cam_x = (x - cx) / fx * ref_depth;
  float cam_y = (y - cy) / fy * ref_depth;
  float cam_z = ref_depth;

  float pcam_x = cam_x - ref_cam.t[0];
  float pcam_y = cam_y - ref_cam.t[1];
  float pcam_z = cam_z - ref_cam.t[2];

  float wx = ref_cam.R[0]*pcam_x + ref_cam.R[3]*pcam_y + ref_cam.R[6]*pcam_z;
  float wy = ref_cam.R[1]*pcam_x + ref_cam.R[4]*pcam_y + ref_cam.R[7]*pcam_z;
  float wz = ref_cam.R[2]*pcam_x + ref_cam.R[5]*pcam_y + ref_cam.R[8]*pcam_z;

  int consistent = 1;
  int carved = 0;
  int n_src = min(num_views - 1, MAX_CLEAN_SOURCES);

  for (int v = 0; v < n_src; ++v) {
    Camera src_cam = cameras[v + 1];

    float sx = src_cam.R[0]*wx + src_cam.R[1]*wy + src_cam.R[2]*wz + src_cam.t[0];
    float sy = src_cam.R[3]*wx + src_cam.R[4]*wy + src_cam.R[5]*wz + src_cam.t[1];
    float sz = src_cam.R[6]*wx + src_cam.R[7]*wy + src_cam.R[8]*wz + src_cam.t[2];

    if (sz <= 1e-6f) continue;

    float su = src_cam.K[0] * sx / sz + src_cam.K[2];
    float sv_coord = src_cam.K[4] * sy / sz + src_cam.K[5];

    // Bounds check (1px margin for bilinear tap).
    if (su < 0.0f || sv_coord < 0.0f ||
        su >= (float)(width - 1) || sv_coord >= (float)(height - 1)) continue;

    // Hardware bilinear depth read via texture cache.
    float src_depth = read_src_depth(v, su, sv_coord,
        src0, src1, src2, src3, src4, src5, src6, src7,
        src8, src9, src10, src11, src12, src13, src14, src15);

    if (!(src_depth > 0.0f)) continue;

    // Space carving: if neighbor's ray passes THROUGH the reference point
    // (neighbor sees further away), the reference point is in free space
    // from this view — a strong signal that it is a floater.
    if (src_depth > sz * (1.0f + carving_threshold)) {
      carved++;
      continue;
    }

    // Forward depth check.
    if (fabs(src_depth - sz) >= sz * same_depth_threshold) continue;

    // === Backward geometric consistency check ===
    float src_fx = src_cam.K[0], src_cx = src_cam.K[2];
    float src_fy = src_cam.K[4], src_cy = src_cam.K[5];

    float src_cam_x = (su - src_cx) / src_fx * src_depth;
    float src_cam_y = (sv_coord - src_cy) / src_fy * src_depth;
    float src_cam_z = src_depth;

    float spc_x = src_cam_x - src_cam.t[0];
    float spc_y = src_cam_y - src_cam.t[1];
    float spc_z = src_cam_z - src_cam.t[2];

    float wx2 = src_cam.R[0]*spc_x + src_cam.R[3]*spc_y + src_cam.R[6]*spc_z;
    float wy2 = src_cam.R[1]*spc_x + src_cam.R[4]*spc_y + src_cam.R[7]*spc_z;
    float wz2 = src_cam.R[2]*spc_x + src_cam.R[5]*spc_y + src_cam.R[8]*spc_z;

    float rx = ref_cam.R[0]*wx2 + ref_cam.R[1]*wy2 + ref_cam.R[2]*wz2 + ref_cam.t[0];
    float ry = ref_cam.R[3]*wx2 + ref_cam.R[4]*wy2 + ref_cam.R[5]*wz2 + ref_cam.t[1];
    float rz = ref_cam.R[6]*wx2 + ref_cam.R[7]*wy2 + ref_cam.R[8]*wz2 + ref_cam.t[2];

    if (rz <= 1e-6f) continue;

    float ru = fx * rx / rz + cx;
    float rv = fy * ry / rz + cy;

    if (fabs(ru - (float)x) + fabs(rv - (float)y) > 2.0f) continue;
    if (fabs(rz - ref_depth) >= ref_depth * same_depth_threshold) continue;

    consistent++;
  }

  int keep = (consistent >= min_consistent_views) && (carved < max_carved_views);
  clean_depth[idx] = keep ? ref_depth : 0.0f;
}

)CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
