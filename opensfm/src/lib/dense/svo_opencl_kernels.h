#pragma once

// OpenCL kernel source for SVO (Sparse Voxel Octree) TSDF integration.
//
// The integration kernel parallelises over depthmap pixels.  Each work-item
// back-projects its pixel, ray-marches through the truncation band, and
// atomically accumulates TSDF / normal / color into a GPU open-addressing
// hash table.
//
// The hash table uses a packed (x,y) 32-bit key with a separate 32-bit z key
// and a "ready" flag to handle the multi-word-key race condition safely.
// Values are accumulated as fixed-point integers and converted to floating
// point on final download.

#ifdef OPENSFM_HAVE_OPENCL

namespace dense {

inline const char* kSVOKernelSource =
R"CL(

// =====================================================================
// Constants
// =====================================================================
#define EMPTY_KEY 0xFFFFFFFFu
#define FP_SCALE  32768
#define MAX_PROBES 256
#define WEIGHT_SCALE 128

// =====================================================================
// GPU hash table slot — must match host-side GPUVoxelSlot exactly.
// 12 x int32 = 48 bytes.
// =====================================================================
typedef struct {
    uint key_ab;      // packed (x+32768)<<16 | (y+32768)
    int  key_c;       // z coordinate
    int  ready;       // 1 after key_c is written
    int  count;       // observation count
    int  sum_tsdf;    // fixed-point TSDF accumulator
    int  sum_nx;      // fixed-point normal.x accumulator
    int  sum_ny;
    int  sum_nz;
    int  sum_r;       // color accumulator (raw uint8 scale)
    int  sum_g;
    int  sum_b;
    int  sum_weight;   // accumulated confidence weight (scale WEIGHT_SCALE)
} VoxelSlot;

// =====================================================================
// Per-frame camera parameters — must match host-side SVOCameraGPU.
// All matrices are stored in **row-major** order.
// =====================================================================
typedef struct {
    float Kinv[9];    // inverse intrinsics
    float Rinv[9];    // R^T  (cam → world rotation)
    float R[9];       // world → cam rotation
    float t[3];       // translation  (world → cam)
    float cam_pos[3]; // camera centre in world = -R^T * t
    float _pad[3];
} SVOCamera;

// =====================================================================
// Hash helpers
// =====================================================================
uint pack_xy(int kx, int ky) {
    return ((uint)(kx + 32768) << 16) | (uint)(ky + 32768);
}

uint voxel_hash(int kx, int ky, int kz) {
    uint h = 2166136261u;
    h ^= (uint)kx; h *= 16777619u;
    h ^= (uint)ky; h *= 16777619u;
    h ^= (uint)kz; h *= 16777619u;
    return h;
}

// =====================================================================
// Atomically insert-or-accumulate a voxel contribution.
//
// All value sums (tsdf, normal, color) are PRE-MULTIPLIED by weight
// before accumulation so that Download can compute weighted averages
// by dividing by sum_weight.
//
// Protocol
//   1. CAS on key_ab to claim an empty slot.
//   2. For a freshly claimed slot, write key_c with atomic_xchg, then
//      set ready=1 with atomic_xchg + mem_fence.
//   3. For a slot that already has our key_ab, spin briefly on ready,
//      then verify key_c.  On match, accumulate; otherwise probe.
//   4. Accumulation uses atomic_add on every field.
// =====================================================================
void hash_accumulate(__global VoxelSlot* table, uint mask,
                     int kx, int ky, int kz,
                     int tsdf_fp,
                     int nx_fp, int ny_fp, int nz_fp,
                     int cr, int cg, int cb, int weight)
{
    uint my_ab = pack_xy(kx, ky);
    if (my_ab == EMPTY_KEY) return;  // (32767,32767) cannot be stored

    // Pre-multiply values by weight for weighted averaging.
    int w_tsdf = tsdf_fp * weight;
    int w_nx   = nx_fp   * weight;
    int w_ny   = ny_fp   * weight;
    int w_nz   = nz_fp   * weight;
    int w_cr   = cr      * weight;
    int w_cg   = cg      * weight;
    int w_cb   = cb      * weight;

    uint h = voxel_hash(kx, ky, kz) & mask;

    for (int probe = 0; probe < MAX_PROBES; probe++) {
        uint slot = (h + (uint)probe) & mask;

        uint old = atomic_cmpxchg(&table[slot].key_ab, EMPTY_KEY, my_ab);

        if (old == EMPTY_KEY) {
            // Freshly claimed — write z, signal ready, accumulate.
            atomic_xchg(&table[slot].key_c, kz);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_xchg(&table[slot].ready, 1);
            mem_fence(CLK_GLOBAL_MEM_FENCE);

            atomic_add(&table[slot].count,    1);
            atomic_add(&table[slot].sum_weight, weight);
            atomic_add(&table[slot].sum_tsdf, w_tsdf);
            atomic_add(&table[slot].sum_nx,   w_nx);
            atomic_add(&table[slot].sum_ny,   w_ny);
            atomic_add(&table[slot].sum_nz,   w_nz);
            atomic_add(&table[slot].sum_r,    w_cr);
            atomic_add(&table[slot].sum_g,    w_cg);
            atomic_add(&table[slot].sum_b,    w_cb);
            return;
        }

        if (old == my_ab) {
            // Same (x,y) — wait for ready, then verify z.
            int ready_val = 0;
            for (int s = 0; s < 512; s++) {
                ready_val = atomic_add(&table[slot].ready, 0);
                if (ready_val != 0) break;
            }
            if (ready_val == 0) continue;  // timed-out — probe next

            int stored_z = atomic_add(&table[slot].key_c, 0);
            if (stored_z == kz) {
                atomic_add(&table[slot].count,    1);
                atomic_add(&table[slot].sum_weight, weight);
                atomic_add(&table[slot].sum_tsdf, w_tsdf);
                atomic_add(&table[slot].sum_nx,   w_nx);
                atomic_add(&table[slot].sum_ny,   w_ny);
                atomic_add(&table[slot].sum_nz,   w_nz);
                atomic_add(&table[slot].sum_r,    w_cr);
                atomic_add(&table[slot].sum_g,    w_cg);
                atomic_add(&table[slot].sum_b,    w_cb);
                return;
            }
            // Different z — linear probe.
        }
        // Different key_ab — linear probe.
    }
    // Dropped (max probes exceeded — table too full).
}

// =====================================================================
// Clear the hash table (one work-item per slot).
// =====================================================================
__kernel void svo_clear_table(__global VoxelSlot* table, uint capacity) {
    uint i = get_global_id(0);
    if (i >= capacity) return;
    table[i].key_ab   = EMPTY_KEY;
    table[i].key_c    = 0;
    table[i].ready    = 0;
    table[i].count    = 0;
    table[i].sum_tsdf = 0;
    table[i].sum_nx   = 0;
    table[i].sum_ny   = 0;
    table[i].sum_nz   = 0;
    table[i].sum_r    = 0;
    table[i].sum_g    = 0;
    table[i].sum_b    = 0;
    table[i].sum_weight = 0;
}


#define COS_THETA_MIN 0.15f   // ~81° — skip pixels at more grazing angles

// =====================================================================
// Integrate one depthmap into the hash table.
// Global work size: (cols_padded, rows_padded).
// =====================================================================
__kernel void svo_integrate(
    __global VoxelSlot*    table,
    uint                   capacity_mask,
    __global const float*  depth,
    __global const float*  normal_buf,
    __global const uchar*  color_buf,
    __global const uchar*  mask_buf,
    __global const float*  weight_buf,
    int                    has_normal,
    int                    has_color,
    int                    has_mask,
    int                    has_weight,
    __constant SVOCamera*  cam,
    int                    rows,
    int                    cols,
    float                  trunc,
    float                  voxel_size,
    float                  inv_voxel_size,
    int                    bbox_min_x,
    int                    bbox_min_y,
    int                    bbox_min_z,
    int                    bbox_max_x,
    int                    bbox_max_y,
    int                    bbox_max_z,
    int                    has_bbox)
{
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= cols || r >= rows) return;

    int idx = r * cols + c;

    if (has_mask && mask_buf[idx] == 0) return;

    float d = depth[idx];
    if (d <= 0.0f) return;

    // ---- Back-project pixel to camera frame ----
    float uc = (float)c, vr = (float)r;
    float3 p_cam;
    p_cam.x = d * (cam->Kinv[0]*uc + cam->Kinv[1]*vr + cam->Kinv[2]);
    p_cam.y = d * (cam->Kinv[3]*uc + cam->Kinv[4]*vr + cam->Kinv[5]);
    p_cam.z = d * (cam->Kinv[6]*uc + cam->Kinv[7]*vr + cam->Kinv[8]);

    // ---- Transform to world: Rinv * (p_cam - t) ----
    float3 diff = (float3)(
        p_cam.x - cam->t[0],
        p_cam.y - cam->t[1],
        p_cam.z - cam->t[2]);
    float3 p_world;
    p_world.x = cam->Rinv[0]*diff.x + cam->Rinv[1]*diff.y + cam->Rinv[2]*diff.z;
    p_world.y = cam->Rinv[3]*diff.x + cam->Rinv[4]*diff.y + cam->Rinv[5]*diff.z;
    p_world.z = cam->Rinv[6]*diff.x + cam->Rinv[7]*diff.y + cam->Rinv[8]*diff.z;

    // ---- Ray from camera centre to surface point ----
    float3 cp = (float3)(cam->cam_pos[0], cam->cam_pos[1], cam->cam_pos[2]);
    float3 ray = p_world - cp;
    float ray_len = length(ray);
    if (ray_len < 1e-8f) return;
    // Guard against float32 precision loss: when ray_len is so large
    // that tt += voxel_size is a no-op (below the ULP), the ray-march
    // loop becomes infinite.  FLT_EPSILON ~ 1.19e-7.
    if (ray_len * 1.2e-7f >= voxel_size) return;
    ray /= ray_len;

    float t_start = fmax(0.0f, ray_len - trunc);
    float t_end   = ray_len + trunc;

    // ---- Per-pixel normal in world frame ----
    float3 n_w = (float3)(0.0f, 0.0f, 0.0f);
    int nx_fp = 0, ny_fp = 0, nz_fp = 0;
    if (has_normal) {
        float3 n_cam = (float3)(
            normal_buf[idx*3], normal_buf[idx*3+1], normal_buf[idx*3+2]);
        float nlen = length(n_cam);
        if (nlen > 1e-6f) {
            n_cam /= nlen;
            n_w.x = cam->Rinv[0]*n_cam.x + cam->Rinv[1]*n_cam.y + cam->Rinv[2]*n_cam.z;
            n_w.y = cam->Rinv[3]*n_cam.x + cam->Rinv[4]*n_cam.y + cam->Rinv[5]*n_cam.z;
            n_w.z = cam->Rinv[6]*n_cam.x + cam->Rinv[7]*n_cam.y + cam->Rinv[8]*n_cam.z;
            nx_fp = (int)(n_w.x * (float)FP_SCALE);
            ny_fp = (int)(n_w.y * (float)FP_SCALE);
            nz_fp = (int)(n_w.z * (float)FP_SCALE);

            // cos(angle between ray and surface normal).
            // Normal should point toward camera (negative dot with ray).
            float cos_theta = fabs(dot(ray, n_w));
            if (cos_theta < COS_THETA_MIN) return;  // grazing — skip pixel
        }
    }

    // ---- Per-pixel color ----
    int cr = 0, cg = 0, cb = 0;
    if (has_color) {
        cr = (int)color_buf[idx*3];
        cg = (int)color_buf[idx*3+1];
        cb = (int)color_buf[idx*3+2];
    }

    // ---- Per-pixel weight: confidence (cos theta is too noisy)----
    float w_float = 1.0f;
    if (has_weight) {
        w_float *= weight_buf[idx];
    }
    int w_int = clamp((int)(w_float * (float)WEIGHT_SCALE),
                      1, WEIGHT_SCALE * 4);

    // ---- Ray-march through the truncation band ----
    for (float tt = t_start; tt <= t_end; tt += voxel_size) {
        float3 sample_pt = cp + tt * ray;

        // World → voxel coordinate
        int kx = (int)floor(sample_pt.x * inv_voxel_size);
        int ky = (int)floor(sample_pt.y * inv_voxel_size);
        int kz = (int)floor(sample_pt.z * inv_voxel_size);

        // Skip out-of-range coordinates (16-bit packing limit).
        if (kx < -32768 || kx > 32766 || ky < -32768 || ky > 32766)
            continue;

        // Skip voxels outside the bounding box (sub-volume clipping).
        if (has_bbox &&
            (kx < bbox_min_x || kx > bbox_max_x ||
             ky < bbox_min_y || ky > bbox_max_y ||
             kz < bbox_min_z || kz > bbox_max_z)) continue;

        // Voxel centre in world coordinates.
        float3 center = (float3)(
            ((float)kx + 0.5f) * voxel_size,
            ((float)ky + 0.5f) * voxel_size,
            ((float)kz + 0.5f) * voxel_size);

        // Project voxel centre into camera frame → get its depth.
        float3 p_cam_v;
        p_cam_v.x = cam->R[0]*center.x + cam->R[1]*center.y + cam->R[2]*center.z + cam->t[0];
        p_cam_v.y = cam->R[3]*center.x + cam->R[4]*center.y + cam->R[5]*center.z + cam->t[1];
        p_cam_v.z = cam->R[6]*center.x + cam->R[7]*center.y + cam->R[8]*center.z + cam->t[2];

        if (p_cam_v.z <= 0.0f) continue;

        float sdf = d - p_cam_v.z;
        if (sdf < -trunc) continue;

        float tsdf = fmin(sdf / trunc, 1.0f);
        int tsdf_fp = (int)(tsdf * (float)FP_SCALE);

        hash_accumulate(table, capacity_mask, kx, ky, kz,
                        tsdf_fp, nx_fp, ny_fp, nz_fp, cr, cg, cb, w_int);
    }
})CL"
R"CL(

// =====================================================================
// Lightweight counting hash table (12 bytes/slot vs 48 for integration).
// Used in a dry-run pass to count unique voxels before allocating the
// full integration table with the right capacity.
// =====================================================================
typedef struct {
    uint key_ab;   // packed (x+32768)<<16 | (y+32768), EMPTY = 0xFFFFFFFF
    int  key_c;    // z coordinate
    int  ready;    // initialisation flag
} CountSlot;

// Insert a voxel key into the counting table.  Increments *counter
// exactly once per unique (kx, ky, kz) coordinate.
void count_hash_insert(__global CountSlot* table, uint mask,
                       __global uint* counter,
                       int kx, int ky, int kz)
{
    uint my_ab = pack_xy(kx, ky);
    if (my_ab == EMPTY_KEY) return;

    uint h = voxel_hash(kx, ky, kz) & mask;

    for (int probe = 0; probe < MAX_PROBES; probe++) {
        uint slot = (h + (uint)probe) & mask;

        uint old = atomic_cmpxchg(&table[slot].key_ab, EMPTY_KEY, my_ab);

        if (old == EMPTY_KEY) {
            // Freshly claimed slot — write z, signal ready, count.
            atomic_xchg(&table[slot].key_c, kz);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_xchg(&table[slot].ready, 1);
            atomic_add(counter, 1u);
            return;
        }

        if (old == my_ab) {
            // Same (x,y) — wait for ready, then verify z.
            int ready_val = 0;
            for (int s = 0; s < 512; s++) {
                ready_val = atomic_add(&table[slot].ready, 0);
                if (ready_val != 0) break;
            }
            if (ready_val == 0) continue;

            int stored_z = atomic_add(&table[slot].key_c, 0);
            if (stored_z == kz) return;  // already counted
        }
        // Different key — linear probe.
    }
    // Dropped (max probes exceeded).
}

// Clear the counting table.
__kernel void svo_clear_count_table(__global CountSlot* table, uint capacity) {
    uint i = get_global_id(0);
    if (i >= capacity) return;
    table[i].key_ab = EMPTY_KEY;
    table[i].key_c  = 0;
    table[i].ready  = 0;
}

// =====================================================================
// Count unique voxels touched by one depthmap (dry-run ray-march).
// Same back-projection and ray-march as svo_integrate but only inserts
// keys into the lightweight counting table — no TSDF/color/normal.
// =====================================================================
__kernel void svo_count(
    __global CountSlot*    table,
    uint                   capacity_mask,
    __global uint*         counter,
    __global const float*  depth,
    __global const uchar*  mask_buf,
    int                    has_mask,
    __constant SVOCamera*  cam,
    int                    rows,
    int                    cols,
    float                  trunc,
    float                  voxel_size,
    float                  inv_voxel_size,
    int                    bbox_min_x,
    int                    bbox_min_y,
    int                    bbox_min_z,
    int                    bbox_max_x,
    int                    bbox_max_y,
    int                    bbox_max_z,
    int                    has_bbox)
{
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= cols || r >= rows) return;

    int idx = r * cols + c;

    if (has_mask && mask_buf[idx] == 0) return;

    float d = depth[idx];
    if (d <= 0.0f) return;

    // ---- Back-project pixel to camera frame ----
    float uc = (float)c, vr = (float)r;
    float3 p_cam;
    p_cam.x = d * (cam->Kinv[0]*uc + cam->Kinv[1]*vr + cam->Kinv[2]);
    p_cam.y = d * (cam->Kinv[3]*uc + cam->Kinv[4]*vr + cam->Kinv[5]);
    p_cam.z = d * (cam->Kinv[6]*uc + cam->Kinv[7]*vr + cam->Kinv[8]);

    // ---- Transform to world: Rinv * (p_cam - t) ----
    float3 diff = (float3)(
        p_cam.x - cam->t[0],
        p_cam.y - cam->t[1],
        p_cam.z - cam->t[2]);
    float3 p_world;
    p_world.x = cam->Rinv[0]*diff.x + cam->Rinv[1]*diff.y + cam->Rinv[2]*diff.z;
    p_world.y = cam->Rinv[3]*diff.x + cam->Rinv[4]*diff.y + cam->Rinv[5]*diff.z;
    p_world.z = cam->Rinv[6]*diff.x + cam->Rinv[7]*diff.y + cam->Rinv[8]*diff.z;

    // ---- Ray from camera centre to surface point ----
    float3 cp = (float3)(cam->cam_pos[0], cam->cam_pos[1], cam->cam_pos[2]);
    float3 ray = p_world - cp;
    float ray_len = length(ray);
    if (ray_len < 1e-8f) return;
    if (ray_len * 1.2e-7f >= voxel_size) return;
    ray /= ray_len;

    float t_start = fmax(0.0f, ray_len - trunc);
    float t_end   = ray_len + trunc;

    // ---- Ray-march through the truncation band ----
    for (float tt = t_start; tt <= t_end; tt += voxel_size) {
        float3 sample_pt = cp + tt * ray;

        int kx = (int)floor(sample_pt.x * inv_voxel_size);
        int ky = (int)floor(sample_pt.y * inv_voxel_size);
        int kz = (int)floor(sample_pt.z * inv_voxel_size);

        if (kx < -32768 || kx > 32766 || ky < -32768 || ky > 32766)
            continue;

        // Skip voxels outside the bounding box (sub-volume clipping).
        if (has_bbox &&
            (kx < bbox_min_x || kx > bbox_max_x ||
             ky < bbox_min_y || ky > bbox_max_y ||
             kz < bbox_min_z || kz > bbox_max_z)) continue;

        // Voxel centre in world coordinates.
        float3 center = (float3)(
            ((float)kx + 0.5f) * voxel_size,
            ((float)ky + 0.5f) * voxel_size,
            ((float)kz + 0.5f) * voxel_size);

        // Project voxel centre into camera frame.
        float3 p_cam_v;
        p_cam_v.x = cam->R[0]*center.x + cam->R[1]*center.y + cam->R[2]*center.z + cam->t[0];
        p_cam_v.y = cam->R[3]*center.x + cam->R[4]*center.y + cam->R[5]*center.z + cam->t[1];
        p_cam_v.z = cam->R[6]*center.x + cam->R[7]*center.y + cam->R[8]*center.z + cam->t[2];

        if (p_cam_v.z <= 0.0f) continue;

        float sdf = d - p_cam_v.z;
        if (sdf < -trunc) continue;

        count_hash_insert(table, capacity_mask, counter, kx, ky, kz);
    }
}

// =====================================================================
// GPU-side surface point extraction.
//
// One work-item per hash table slot.  For each occupied slot with
// sufficient weight, check the three positive-axis neighbours
// (+X, +Y, +Z) for a TSDF sign change.  When found, linearly
// interpolate position, normal, and color and write to the output
// arrays via an atomic counter.
//
// This replaces the expensive Download→ImportVoxels→ExtractPoints
// CPU pipeline and avoids transferring the entire hash table.
// =====================================================================

// Look up a voxel in the integration hash table by its integer
// coordinate.  Returns the slot index if found, 0xFFFFFFFF otherwise.
uint hash_lookup(__global const VoxelSlot* table, uint mask,
                 int kx, int ky, int kz)
{
    uint my_ab = pack_xy(kx, ky);
    if (my_ab == EMPTY_KEY) return 0xFFFFFFFF;

    uint h = voxel_hash(kx, ky, kz) & mask;

    for (int probe = 0; probe < MAX_PROBES; probe++) {
        uint slot_idx = (h + (uint)probe) & mask;
        uint stored_ab = table[slot_idx].key_ab;

        if (stored_ab == EMPTY_KEY) return 0xFFFFFFFF;

        if (stored_ab == my_ab) {
            if (table[slot_idx].ready == 0) continue;
            if (table[slot_idx].key_c == kz) return slot_idx;
        }
    }
    return 0xFFFFFFFF;
}

__kernel void svo_extract_points(
    __global const VoxelSlot* table,
    uint                      capacity_mask,
    uint                      capacity,
    float                     min_weight_scaled,
    float                     voxel_size,
    __global float*           out_points,
    __global float*           out_normals,
    __global uchar*           out_colors,
    __global uint*            out_counter,
    uint                      max_output)
{
    uint i = get_global_id(0);
    if (i >= capacity) return;

    // Read slot A.
    uint key_ab_a = table[i].key_ab;
    if (key_ab_a == EMPTY_KEY) return;
    int count_a = table[i].count;
    if (count_a == 0) return;

    float sw_a = (float)table[i].sum_weight;
    if (sw_a < min_weight_scaled) return;

    int kx = (int)((key_ab_a >> 16) & 0xFFFF) - 32768;
    int ky = (int)(key_ab_a & 0xFFFF) - 32768;
    int kz = table[i].key_c;

    float inv_sw_a     = 1.0f / sw_a;
    float inv_fp_sw_a  = inv_sw_a / (float)FP_SCALE;
    float tsdf_a       = (float)table[i].sum_tsdf * inv_fp_sw_a;

    // Pre-decode voxel A normal & color (reused for all 3 axes).
    float na_x = (float)table[i].sum_nx * inv_fp_sw_a;
    float na_y = (float)table[i].sum_ny * inv_fp_sw_a;
    float na_z = (float)table[i].sum_nz * inv_fp_sw_a;
    float ra   = (float)table[i].sum_r  * inv_sw_a;
    float ga   = (float)table[i].sum_g  * inv_sw_a;
    float ba   = (float)table[i].sum_b  * inv_sw_a;

    // Check +X, +Y, +Z neighbours.
    int off_x[3] = {1, 0, 0};
    int off_y[3] = {0, 1, 0};
    int off_z[3] = {0, 0, 1};

    for (int d = 0; d < 3; d++) {
        int nx = kx + off_x[d];
        int ny = ky + off_y[d];
        int nz = kz + off_z[d];

        uint nb_idx = hash_lookup(table, capacity_mask, nx, ny, nz);
        if (nb_idx == 0xFFFFFFFF) continue;

        float sw_b = (float)table[nb_idx].sum_weight;
        if (sw_b < min_weight_scaled) continue;

        float inv_sw_b    = 1.0f / sw_b;
        float inv_fp_sw_b = inv_sw_b / (float)FP_SCALE;
        float tsdf_b      = (float)table[nb_idx].sum_tsdf * inv_fp_sw_b;

        // Sign change?
        if ((tsdf_a > 0.0f) == (tsdf_b > 0.0f)) continue;
        if (tsdf_a == 0.0f || tsdf_b == 0.0f) continue;

        // Interpolation factor: tsdf_a + t*(tsdf_b - tsdf_a) = 0.
        float t = tsdf_a / (tsdf_a - tsdf_b);
        t = clamp(t, 0.0f, 1.0f);

        // Interpolated position.
        float pa_x = ((float)kx + 0.5f) * voxel_size;
        float pa_y = ((float)ky + 0.5f) * voxel_size;
        float pa_z = ((float)kz + 0.5f) * voxel_size;
        float pb_x = ((float)nx + 0.5f) * voxel_size;
        float pb_y = ((float)ny + 0.5f) * voxel_size;
        float pb_z = ((float)nz + 0.5f) * voxel_size;

        float px = pa_x + t * (pb_x - pa_x);
        float py = pa_y + t * (pb_y - pa_y);
        float pz = pa_z + t * (pb_z - pa_z);

        // Interpolated normal.
        float nb_x = (float)table[nb_idx].sum_nx * inv_fp_sw_b;
        float nb_y = (float)table[nb_idx].sum_ny * inv_fp_sw_b;
        float nb_z = (float)table[nb_idx].sum_nz * inv_fp_sw_b;
        float inx = na_x + t * (nb_x - na_x);
        float iny = na_y + t * (nb_y - na_y);
        float inz = na_z + t * (nb_z - na_z);
        float nlen = sqrt(inx*inx + iny*iny + inz*inz);
        if (nlen > 1e-6f) { inx /= nlen; iny /= nlen; inz /= nlen; }

        // Interpolated color.
        float rb = (float)table[nb_idx].sum_r * inv_sw_b;
        float gb = (float)table[nb_idx].sum_g * inv_sw_b;
        float bb = (float)table[nb_idx].sum_b * inv_sw_b;

        // Atomic output index.
        uint out_idx = atomic_add(out_counter, 1u);
        if (out_idx >= max_output) continue;

        out_points[out_idx * 3 + 0] = px;
        out_points[out_idx * 3 + 1] = py;
        out_points[out_idx * 3 + 2] = pz;
        out_normals[out_idx * 3 + 0] = inx;
        out_normals[out_idx * 3 + 1] = iny;
        out_normals[out_idx * 3 + 2] = inz;
        out_colors[out_idx * 3 + 0] = (uchar)clamp(ra + t * (rb - ra), 0.0f, 255.0f);
        out_colors[out_idx * 3 + 1] = (uchar)clamp(ga + t * (gb - ga), 0.0f, 255.0f);
        out_colors[out_idx * 3 + 2] = (uchar)clamp(ba + t * (bb - ba), 0.0f, 255.0f);
    }
})CL"
R"CL(

)CL";

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
