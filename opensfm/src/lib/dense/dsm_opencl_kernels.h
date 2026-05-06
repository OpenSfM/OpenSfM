#pragma once

// OpenCL kernels for DSM (Digital Surface Model) rasterization.
//
// Two-pass depthmap rasterization:
//   Pass 1: back-project each clean depthmap pixel to world XY, scatter
//           weighted Z onto a 2D grid (atomic_add on fixed-point accumulators).
//   Pass 2: re-scatter rejecting outliers that deviate from the pass-1 mean.
//
// Followed by an optional edge-preserving bilateral filter.

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline const char* kDSMKernelSource = R"CL(

// Fixed-point scale for weighted Z accumulation.
// 8192 gives ~0.12 mm precision.  Max accumulated value per cell before
// overflow: ~260 000 (z * w * FP_SCALE * N_views < 2^31).
#define FP_SCALE 8192

// ----------------------------------------------------------------
// Clear three int32 accumulation grids to zero.
// ----------------------------------------------------------------
__kernel void dsm_clear_grid(
    __global int* grid_a,
    __global int* grid_b,
    __global int* grid_c,
    const int num_cells)
{
    const int idx = get_global_id(0);
    if (idx >= num_cells) return;
    grid_a[idx] = 0;
    grid_b[idx] = 0;
    grid_c[idx] = 0;
}

// ----------------------------------------------------------------
// Back-project every depthmap pixel to world space and scatter a
// confidence-and-normal-weighted Z value onto the DSM grid.
//
// cam_params layout (21 floats, row-major):
//   [0..8]   K_inv  (3x3 inverse intrinsics)
//   [9..17]  R_inv  (3x3 = R^T, camera-to-world rotation)
//   [18..20] t      (translation: X_cam = R * X_world + t)
// ----------------------------------------------------------------
__kernel void dsm_backproject_scatter(
    __global const float* depth,
    __global const float* normal,
    __global const float* confidence,
    __global const float* cam_params,
    __global volatile int* sum_zw,
    __global volatile int* sum_w,
    __global volatile int* count,
    const float origin_x,
    const float origin_y,
    const float inv_gsd,
    const int grid_w,
    const int grid_h,
    const int img_w,
    const int img_h)
{
    const int idx = get_global_id(0);
    if (idx >= img_w * img_h) return;

    const float d = depth[idx];
    if (d <= 0.0f) return;

    const int u = idx % img_w;
    const int v = idx / img_w;

    // K_inv (row-major)
    const float ki00 = cam_params[0], ki01 = cam_params[1], ki02 = cam_params[2];
    const float ki10 = cam_params[3], ki11 = cam_params[4], ki12 = cam_params[5];
    const float ki20 = cam_params[6], ki21 = cam_params[7], ki22 = cam_params[8];
    // R_inv (row-major)
    const float ri00 = cam_params[9],  ri01 = cam_params[10], ri02 = cam_params[11];
    const float ri10 = cam_params[12], ri11 = cam_params[13], ri12 = cam_params[14];
    const float ri20 = cam_params[15], ri21 = cam_params[16], ri22 = cam_params[17];
    // Translation
    const float tx = cam_params[18], ty = cam_params[19], tz = cam_params[20];

    // Back-project: cam_pt = K_inv * [u, v, 1] * depth
    const float fu = (float)u;
    const float fv = (float)v;
    const float cx = (ki00 * fu + ki01 * fv + ki02) * d;
    const float cy = (ki10 * fu + ki11 * fv + ki12) * d;
    const float cz = (ki20 * fu + ki21 * fv + ki22) * d;

    // World = R_inv * (cam_pt - t)
    const float bx = cx - tx, by = cy - ty, bz = cz - tz;
    const float wx = ri00 * bx + ri01 * by + ri02 * bz;
    const float wy = ri10 * bx + ri11 * by + ri12 * bz;
    const float wz = ri20 * bx + ri21 * by + ri22 * bz;

    // Grid cell
    const int gx = (int)floor((wx - origin_x) * inv_gsd);
    const int gy = (int)floor((wy - origin_y) * inv_gsd);
    if (gx < 0 || gx >= grid_w || gy < 0 || gy >= grid_h) return;

    // Normal Z in world frame (up-facing weight)
    const int nbase = idx * 3;
    const float cnx = normal[nbase];
    const float cny = normal[nbase + 1];
    const float cnz = normal[nbase + 2];
    const float wnz = ri20 * cnx + ri21 * cny + ri22 * cnz;

    // Hard cutoff: discard pixels whose surface is too steep (walls).
    // 0.3 ≈ 73° from vertical — keeps pitched roofs, rejects walls.
    if (wnz < 0.35f) return;
    const float w = confidence[idx] * wnz;

    // Fixed-point accumulation
    const int cell = gy * grid_w + gx;
    atomic_add(&sum_zw[cell], (int)(wz * w * (float)FP_SCALE));
    atomic_add(&sum_w[cell],  (int)(w * (float)FP_SCALE));
    atomic_add(&count[cell], 1);
}

// ----------------------------------------------------------------
// Finalize: weighted mean Z = sum_zw / sum_w.  NaN where no data
// or insufficient observations.
// ----------------------------------------------------------------
__kernel void dsm_finalize(
    __global const int* sum_zw,
    __global const int* sum_w,
    __global const int* count,
    __global float* out_z,
    const int num_cells,
    const int min_count,
    const int min_weight)
{
    const int idx = get_global_id(0);
    if (idx >= num_cells) return;
    const int sw = sum_w[idx];
    const int c  = count[idx];
    out_z[idx] = (c < min_count || sw < min_weight)
        ? NAN
        : ((float)sum_zw[idx] / (float)sw);
}

// ----------------------------------------------------------------
// Pass-2 scatter: back-project with one-sided outlier rejection
// and exponential Z-bias (softmax).
//
// The bias exp(z_bias * (wz - mean) / threshold) strongly upweights
// above-mean observations and suppresses below-mean ones, yielding
// an approximate upper percentile without histograms or extra memory.
//   z_bias = 0   → plain weighted mean (no bias)
//   z_bias ≈ 2.5 → ~P90 behaviour
//   z_bias → ∞   → approaches max
// ----------------------------------------------------------------
__kernel void dsm_reject_scatter(
    __global const float* depth,
    __global const float* normal,
    __global const float* confidence,
    __global const float* cam_params,
    __global const float* mean_z,
    __global volatile int* sum_zw,
    __global volatile int* sum_w,
    __global volatile int* count,
    const float origin_x,
    const float origin_y,
    const float inv_gsd,
    const int grid_w,
    const int grid_h,
    const int img_w,
    const int img_h,
    const float reject_threshold,
    const float z_bias)
{
    const int idx = get_global_id(0);
    if (idx >= img_w * img_h) return;

    const float d = depth[idx];
    if (d <= 0.0f) return;

    const int u = idx % img_w;
    const int v = idx / img_w;

    const float ki00 = cam_params[0], ki01 = cam_params[1], ki02 = cam_params[2];
    const float ki10 = cam_params[3], ki11 = cam_params[4], ki12 = cam_params[5];
    const float ki20 = cam_params[6], ki21 = cam_params[7], ki22 = cam_params[8];
    const float ri00 = cam_params[9],  ri01 = cam_params[10], ri02 = cam_params[11];
    const float ri10 = cam_params[12], ri11 = cam_params[13], ri12 = cam_params[14];
    const float ri20 = cam_params[15], ri21 = cam_params[16], ri22 = cam_params[17];
    const float tx = cam_params[18], ty = cam_params[19], tz = cam_params[20];

    const float fu = (float)u;
    const float fv = (float)v;
    const float cx = (ki00 * fu + ki01 * fv + ki02) * d;
    const float cy = (ki10 * fu + ki11 * fv + ki12) * d;
    const float cz = (ki20 * fu + ki21 * fv + ki22) * d;

    const float bx = cx - tx, by = cy - ty, bz = cz - tz;
    const float wx = ri00 * bx + ri01 * by + ri02 * bz;
    const float wy = ri10 * bx + ri11 * by + ri12 * bz;
    const float wz = ri20 * bx + ri21 * by + ri22 * bz;

    const int gx = (int)floor((wx - origin_x) * inv_gsd);
    const int gy = (int)floor((wy - origin_y) * inv_gsd);
    if (gx < 0 || gx >= grid_w || gy < 0 || gy >= grid_h) return;

    // Outlier rejection against pass-1 mean (one-sided: only reject below)
    const int cell = gy * grid_w + gx;
    const float mz = mean_z[cell];
    if (isnan(mz) || (mz - wz) > reject_threshold) return;

    const int nbase = idx * 3;
    const float cnx = normal[nbase];
    const float cny = normal[nbase + 1];
    const float cnz = normal[nbase + 2];
    const float wnz = ri20 * cnx + ri21 * cny + ri22 * cnz;

    if (wnz < 0.3f) return;

    float w = confidence[idx] * wnz;

    // Exponential bias toward higher Z (approximate upper percentile).
    if (z_bias > 0.0f) {
        const float delta = (wz - mz) / reject_threshold;
        float bias = exp(z_bias * delta);
        bias = fmin(bias, 5.0f);    // cap to prevent int32 overflow
        w *= bias;
    }

    // Use a reduced fixed-point scale for pass 2.  The FP scale cancels
    // in the weighted-mean ratio (sum_zw / sum_w), so lower precision here
    // does not affect the final result.  This keeps the accumulators well
    // within int32 range even with the exponential bias.
    //   Worst case per pixel: Z=200 * w=5 * 128 = 128 000
    //   INT_MAX / 128 000 ≈ 16 700 contributions per cell → safe.
    #define FP_SCALE_P2 128
    atomic_add(&sum_zw[cell], (int)(wz * w * (float)FP_SCALE_P2));
    atomic_add(&sum_w[cell],  (int)(w * (float)FP_SCALE_P2));
    atomic_add(&count[cell], 1);
    #undef FP_SCALE_P2
}

// ----------------------------------------------------------------
// Edge-preserving bilateral filter on the DSM grid (float).
// 2D dispatch: global_id(0) = x, global_id(1) = y.
// ----------------------------------------------------------------
__kernel void dsm_bilateral(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    const int radius,
    const float inv_2sigma_s_sq,
    const float inv_2sigma_r_sq)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float center = input[y * width + x];
    if (isnan(center)) {
        output[y * width + x] = NAN;
        return;
    }

    float sum_val = 0.0f;
    float sum_wt  = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            const float nval = input[ny * width + nx];
            if (isnan(nval)) continue;

            const float sd = (float)(dx * dx + dy * dy);
            const float rd = center - nval;
            const float wt = exp(-sd * inv_2sigma_s_sq
                                 - rd * rd * inv_2sigma_r_sq);
            sum_val += nval * wt;
            sum_wt  += wt;
        }
    }

    output[y * width + x] = (sum_wt > 0.0f) ? (sum_val / sum_wt) : NAN;
}

)CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
