#pragma once

// OpenCL kernel source for brute-force descriptor matching, embedded as a C++
// raw string.  Follows the same pattern as dense/opencl_kernels.h.
//
// The kernel computes L2 distances between two sets of descriptors and tracks
// the 2 nearest neighbours per query row for Lowe's ratio test.
//
// Optimised for HAHOG descriptors (128-dimensional float32) via float4
// vectorised loads and shared-memory tiling.

#ifdef OPENSFM_HAVE_OPENCL

namespace features {

inline const char* kBruteForceMatchKernelSource = R"CL(

// DESC_DIM is defined at compile time (e.g. -DDESC_DIM=128).
// TILE_SIZE is the number of descriptors loaded into local memory per tile.

// DESC_DIM_VEC4 = DESC_DIM / 4  (number of float4 elements per descriptor).
#define DESC_DIM_VEC4 (DESC_DIM / 4)

// =====================================================================
// brute_force_knn2
//
// For each query descriptor (row in f1), find the 2 nearest neighbours
// in the reference set (f2) using squared L2 distance.
//
// 1D dispatch: global_size >= N1 (rounded up to work-group multiple).
// Excess work-items participate in barriers but skip computation/writes.
//
// Output:
//   best_idx[i]   — index in f2 of the nearest neighbour of f1[i]
//   best_dist[i]  — squared L2 distance to nearest neighbour
//   second_dist[i] — squared L2 distance to second nearest neighbour
// =====================================================================
__kernel void brute_force_knn2(
    __global const float* f1,      // [N1 x DESC_DIM]
    __global const float* f2,      // [N2 x DESC_DIM]
    __global int*   best_idx,      // [N1]
    __global float* best_dist,     // [N1]
    __global float* second_dist,   // [N1]
    const int N1,
    const int N2)
{
    const int i = get_global_id(0);  // query index
    // NOTE: do NOT return early for i >= N1 — all work-items in the
    // work-group must hit the same barriers to avoid GPU hangs.
    const int valid = (i < N1);

    // Load query descriptor into private registers (float4 for coalescing).
    float4 q[DESC_DIM_VEC4];
    if (valid) {
        __global const float4* q_ptr =
            (__global const float4*)(f1 + (long)i * DESC_DIM);
        for (int d = 0; d < DESC_DIM_VEC4; ++d) {
            q[d] = q_ptr[d];
        }
    }

    // Tile f2 through local memory.
    __local float4 tile4[TILE_SIZE * DESC_DIM_VEC4];
    __local float* tile = (__local float*)tile4;

    float bd  = INFINITY;   // best distance
    float sd  = INFINITY;   // second-best distance
    int   bi  = -1;         // best index

    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    for (int tile_start = 0; tile_start < N2; tile_start += TILE_SIZE) {
        // Cooperatively load a tile of TILE_SIZE descriptors from f2.
        // ALL work-items (including excess ones) participate in the load
        // and the barriers.
        barrier(CLK_LOCAL_MEM_FENCE);
        const int tile_elems = TILE_SIZE * DESC_DIM;
        for (int t = lid; t < tile_elems; t += local_size) {
            int row = t / DESC_DIM;
            int col = t % DESC_DIM;
            int j = tile_start + row;
            tile[t] = (j < N2) ? f2[(long)j * DESC_DIM + col] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute distances to each descriptor in the tile.
        if (valid) {
            int tile_count = min(TILE_SIZE, N2 - tile_start);
            for (int t = 0; t < tile_count; ++t) {
                __local const float4* r_ptr = tile4 + t * DESC_DIM_VEC4;

                float4 acc = (float4)(0.0f);
                for (int d = 0; d < DESC_DIM_VEC4; ++d) {
                    float4 diff = q[d] - r_ptr[d];
                    acc += diff * diff;
                }
                float dist = acc.x + acc.y + acc.z + acc.w;

                int j = tile_start + t;
                if (dist < bd) {
                    sd = bd;
                    bd = dist;
                    bi = j;
                } else if (dist < sd) {
                    sd = dist;
                }
            }
        }
    }

    if (valid) {
        best_idx[i]     = bi;
        best_dist[i]    = bd;
        second_dist[i]  = sd;
    }
}

)CL";

// =====================================================================
// Hamming-distance brute-force matching kernel for binarised descriptors.
//
// Each descriptor is N_WORDS × uint32 (e.g. 4 words = 128 bits).
// Distance = sum of popcount(a[k] ^ b[k]) over k.
//
// Uses local-memory tiling (HAMMING_TILE_SIZE descriptors per tile)
// to avoid redundant global memory reads — same pattern as the L2
// kernel.  Binary descriptors are tiny (16 bytes for 128 bits) so
// tiles are large (256 × 16 bytes = 4 KB), giving excellent reuse.
// =====================================================================
inline const char* kHammingMatchKernelSource = R"CL(

// N_WORDS and HAMMING_TILE_SIZE are defined at compile time.

__kernel void brute_force_hamming_knn2(
    __global const uint* f1,       // [N1 * N_WORDS]
    __global const uint* f2,       // [N2 * N_WORDS]
    __global int*   best_idx,      // [N1]
    __global int*   best_dist,     // [N1]
    __global int*   second_dist,   // [N1]
    const int N1,
    const int N2)
{
    const int i = get_global_id(0);
    // Do NOT return early — all work-items must hit the same barriers.
    const int valid = (i < N1);

    // Load query descriptor into private registers.
    uint q[N_WORDS];
    if (valid) {
        for (int k = 0; k < N_WORDS; ++k)
            q[k] = f1[(long)i * N_WORDS + k];
    }

    // Tile f2 through local memory.
    __local uint tile[HAMMING_TILE_SIZE * N_WORDS];

    int bd = N_WORDS * 32 + 1;
    int sd = N_WORDS * 32 + 1;
    int bi = -1;

    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    for (int tile_start = 0; tile_start < N2; tile_start += HAMMING_TILE_SIZE) {
        // Cooperatively load a tile of descriptors from f2.
        barrier(CLK_LOCAL_MEM_FENCE);
        const int tile_elems = HAMMING_TILE_SIZE * N_WORDS;
        for (int t = lid; t < tile_elems; t += local_size) {
            int row = t / N_WORDS;
            int col = t % N_WORDS;
            int j = tile_start + row;
            tile[t] = (j < N2) ? f2[(long)j * N_WORDS + col] : 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute Hamming distances to each descriptor in the tile.
        if (valid) {
            int tile_count = min((int)HAMMING_TILE_SIZE, N2 - tile_start);
            for (int t = 0; t < tile_count; ++t) {
                int dist = 0;
                for (int k = 0; k < N_WORDS; ++k)
                    dist += popcount(q[k] ^ tile[t * N_WORDS + k]);

                int j = tile_start + t;
                if (dist < bd) {
                    sd = bd;
                    bd = dist;
                    bi = j;
                } else if (dist < sd) {
                    sd = dist;
                }
            }
        }
    }

    if (valid) {
        best_idx[i]    = bi;
        best_dist[i]   = bd;
        second_dist[i] = sd;
    }
}

)CL";

// =====================================================================
// Batched Hamming KNN2 — processes ALL pairs in a single kernel launch.
//
// Each work-group reads its pair's f2 slice from a per-wg info buffer,
// so all work-items in a work-group share the same f2 region and
// cooperative tiling works exactly as in the single-pair kernel.
//
// wg_info layout: 4 ints per work-group:
//   [0] f1_off   — offset of this wg's first query in all_f1
//   [1] n1_valid — valid queries in this wg (≤ local_size)
//   [2] f2_off   — offset of this pair's f2 block in all_f2
//   [3] n2       — reference count for this pair
// =====================================================================
inline const char* kHammingMatchBatchedKernelSource = R"CL(

__kernel void brute_force_hamming_knn2_batched(
    __global const uint* all_f1,
    __global const uint* all_f2,
    __global int*   best_idx,
    __global int*   best_dist,
    __global int*   second_dist,
    __global const int* wg_info,
    const int total_n1)
{
    const int wg  = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    const int f1_off   = wg_info[wg * 4 + 0];
    const int n1_valid = wg_info[wg * 4 + 1];
    const int f2_off   = wg_info[wg * 4 + 2];
    const int n2       = wg_info[wg * 4 + 3];

    const int valid = (lid < n1_valid);
    const int out_idx = f1_off + lid;

    uint q[N_WORDS];
    if (valid) {
        for (int k = 0; k < N_WORDS; ++k)
            q[k] = all_f1[(long)(f1_off + lid) * N_WORDS + k];
    }

    __local uint tile[HAMMING_TILE_SIZE * N_WORDS];

    int bd = N_WORDS * 32 + 1;
    int sd = N_WORDS * 32 + 1;
    int bi = -1;

    for (int tile_start = 0; tile_start < n2; tile_start += HAMMING_TILE_SIZE) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const int tile_elems = HAMMING_TILE_SIZE * N_WORDS;
        for (int t = lid; t < tile_elems; t += local_size) {
            int row = t / N_WORDS;
            int col = t % N_WORDS;
            int j = tile_start + row;
            tile[t] = (j < n2)
                ? all_f2[(long)(f2_off + j) * N_WORDS + col] : 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (valid) {
            int tile_count = min((int)HAMMING_TILE_SIZE, n2 - tile_start);
            for (int t = 0; t < tile_count; ++t) {
                int dist = 0;
                for (int k = 0; k < N_WORDS; ++k)
                    dist += popcount(q[k] ^ tile[t * N_WORDS + k]);

                int j = tile_start + t;
                if (dist < bd) {
                    sd = bd;
                    bd = dist;
                    bi = j;
                } else if (dist < sd) {
                    sd = dist;
                }
            }
        }
    }

    if (valid) {
        best_idx[out_idx]     = bi;
        best_dist[out_idx]    = bd;
        second_dist[out_idx]  = sd;
    }
}

)CL";

}  // namespace features

#endif  // OPENSFM_HAVE_OPENCL
