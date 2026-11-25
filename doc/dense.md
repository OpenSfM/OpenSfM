# Dense Reconstruction & 2D Maps

After a sparse reconstruction (camera poses + sparse points), OpenSfM can compute a **dense** representation of the scene and derive 2D map products from it:

- a **dense point cloud** (`fused.ply`, optionally `fused.las` / `fused.laz`),
- a **triangle mesh** (`mesh.ply`, Surface Nets),
- a **Digital Surface Model** raster (`dsm.tif`) and an **orthophoto** (`ortho.tif`),
- streaming **octree tiles** for the web/point-cloud viewer (`point_cloud/`).

![Dense Reconstruction](images/dense.png)
![DSM and Orthophoto](images/dsm_ortho.png)

> The math behind depthmap backprojection, plane-induced homographies and undistortion is in [dense_matching.md](dense_matching.md). This page is the practical how-to.

---

## Prerequisites

1. A sparse reconstruction (`reconstruction.json`), i.e. the core pipeline up to `reconstruct`.
2. An **undistorted** reconstruction: run `undistort` first. The dense stages operate exclusively on `undistorted/` (perspective images, no radial distortion; spherical panoramas are split into several perspective views — see [dense_matching.md](dense_matching.md#undistortion)).
3. A **GPU with OpenCL**: depthmap estimation and fusion run on the GPU.

All dense artefacts live under `undistorted/depthmaps/`.

---

## The four-stage pipeline

Dense reconstruction is split into four commands that hand off through disk, so each can run in its own process (bounded memory, clean teardown) — or even on a different machine sharing the dataset directory. Each later stage fails fast if an earlier stage's artefacts are missing.

| # | Command | Reads | Writes |
| - | ------- | ----- | ------ |
| 1 | `dense_clustering`  | undistorted reconstruction + tracks | clusters, super-points, neighbours, depth ranges, cluster bounding boxes |
| — | `dense_equalize` (optional) | undistorted reconstruction + tracks | `equalization.json` (per-image gain, white balance, vignetting) |
| 2 | `compute_depthmaps` | stage-1 artefacts + undistorted images | per-shot `*.raw.npz` → `*.clean.npz` depthmaps |
| 3 | `fuse_depthmaps`    | cleaned depthmaps + stage-1 artefacts | per-cluster `fused_batch_*.ply` (+ `mesh_batch_*.ply`, DSM/ortho tiles) |
| 4 | `dense_merging`     | per-cluster batches | `fused.ply`, `mesh.ply`, `dsm.tif`, `ortho.tif`, `fused.las/.laz`, `point_cloud/` |

```bash
bin/opensfm undistort         path/to/dataset
bin/opensfm dense_clustering  path/to/dataset
bin/opensfm dense_equalize    path/to/dataset   # optional: seam-free colours
bin/opensfm compute_depthmaps path/to/dataset
bin/opensfm fuse_depthmaps    path/to/dataset
bin/opensfm dense_merging     path/to/dataset   # add --georeferenced for geo-located LAS/LAZ + DSM/ortho
```

### Stage 1 — `dense_clustering`

Groups covisible shots into small clusters and prepares the per-shot context the GPU stages need:

- **Clusters** — shots that see common surface are grouped (bounded by `depthmap_cluster_max_size`) so fusion can process the scene in disjoint, memory-bounded chunks.
- **Super-points** — sparse track points fused per cluster, used to seed depth ranges and priors.
- **Neighbours** — for each shot, the best matching views for stereo (gated by baseline angle, `depthmap_neighbor_min_angle` / `_max_angle`) are stored as `neighbors_best.json` / `neighbors_all.json`.
- **Depth ranges** — a per-shot `[min, max]` depth used to bound the PatchMatch search.

Outputs: `clusters.json`, `clusters_points.json`, `neighbors_best.json`, `neighbors_all.json`, `depth_ranges.json`, `cluster_bboxes.json`.

### Stage 2 — `compute_depthmaps`

Estimates a depth + normal map for every shot, then cleans it.

- **Estimation** — a PatchMatch-based multi-view stereo (ACMMP-style: adaptive checkerboard propagation with multi-hypothesis sampling) running on the GPU via OpenCL. It is **multi-scale** (`depthmap_hierarchy_levels`), uses a bilateral-NCC matching cost with a Census-transform fallback in low-texture regions (`depthmap_use_census`), and can add a geometric-consistency term across neighbouring views (`depthmap_geom_consistency_weight`).
- **Cleaning** — a consistency + visibility pass keeps only depths confirmed by enough neighbours (`depthmap_min_consistent_views`) and carves away depths that are occluded by other views (`depthmap_carving_threshold`, two-pass by default).

Outputs (per shot, lz4-compressed NumPy): `<image>.raw.npz` then `<image>.clean.npz`. The raw map is deleted once its clean counterpart exists (`depthmap_delete_raw_after_clean`). Set `depthmap_save_debug_ply` to also dump per-shot/per-cluster PLYs for inspection.

See the [Depth Estimation](configuration.md#depth-estimation-patchmatch-opencl) config for all knobs.

### Stage 3 — `fuse_depthmaps`

Fuses the cleaned depthmaps into a surface using a **sparse-voxel-octree (SVO) TSDF** integrator, one cluster at a time over disjoint territories (so two clusters never fuse the same surface twice).

- **Voxel size is automatic**, derived from the data's median per-pixel ground footprint (`depth/focal`); `depthmap_fusion_svo_voxel_level` selects `fine` / `half` / `quarter` of that resolution.
- **Mesh** — when `depthmap_fusion_mesh_enabled` is set (opt-in, **off by default**), a **Surface Nets** (dual-contouring) triangle mesh is extracted from the same TSDF zero-set. Unlike Poisson it never balloons facades, but it leaves holes where the volume is empty.
- **2D tiles** — when `dsm_enabled` (default), each cluster also renders its slice of the DSM and orthophoto.

Outputs per cluster: `fused_batch_*.ply` (xyz + normal + rgb), `mesh_batch_*.ply`, `dsm_ortho_batch_*.npz`. See the [Fusion](configuration.md#fusion) config.

### Stage 4 — `dense_merging`

Merges the per-cluster batches into the final products and exports them:

- **`fused.ply`** — the merged dense point cloud (always topocentric).
- **`mesh.ply`** — the merged Surface Nets mesh, when `depthmap_fusion_mesh_enabled` is set (always topocentric).
- **`dsm.tif` / `ortho.tif`** — the merged DSM and orthophoto GeoTIFFs (per-cluster tiles are feather-blended, `dsm_merge_feather`).
- **`fused.las` / `fused.laz`** — the dense cloud as LAS/LAZ, when `dense_pointcloud_export_las` / `_laz` are enabled.
- **`point_cloud/`** — Potree-style octree tiles for streaming in a web/point-cloud viewer (see [Octree Tiling](configuration.md#octree-tiling)).

Add **`--georeferenced`** to write the LAS/LAZ and DSM/ortho in the output coordinate system instead of the topocentric frame. The point cloud PLY, mesh and octree always stay topocentric. See [Georeferencing & GIS outputs](georeferencing.md) for how the output CRS is chosen and a full GIS walkthrough.

---

## Output reference

All paths are relative to `undistorted/depthmaps/`.

| File | Format | Produced by | Georeferenced? |
| ---- | ------ | ----------- | -------------- |
| `<image>.clean.npz` | lz4 NPZ (depth, normal, score) | `compute_depthmaps` | — |
| `fused.ply` | binary PLY (xyz + normal + rgb) | `dense_merging` | no (topocentric) |
| `mesh.ply` | binary PLY mesh | `dense_merging` | no (topocentric) |
| `fused.las`, `fused.laz` | LAS / LAZ 1.4 | `dense_merging` | with `--georeferenced` |
| `dsm.tif` | single-band Float32 GeoTIFF | `dense_merging` | with `--georeferenced` (else topocentric grid) |
| `ortho.tif` | RGB + alpha GeoTIFF | `dense_merging` | with `--georeferenced` (else topocentric grid) |
| `point_cloud/` | Potree octree tiles | `dense_merging` | no (topocentric) |

> **Legacy note:** older docs/tools refer to `merged.ply`; the current pipeline writes `fused.ply`. `export_geocoords --dense` still reads the legacy `merged.ply` — for georeferenced dense products use `dense_merging --georeferenced`.

---

## 2D maps: DSM and orthophoto

The **DSM** (`dsm.tif`) is a top-down height raster; the **orthophoto** (`ortho.tif`) is a top-down, perspective-corrected color image on the same grid. Both are rendered from the fused surface during stage 3 and finalized in stage 4. The ground sample distance is derived automatically from the fused voxel size.

DSM/ortho quality is shaped by a dedicated config group ([DSM and Orthophoto](configuration.md#dsm-and-orthophoto)): shape-aware hole filling (compact gaps flat-filled, elongated ones edge-aware diffused), a coherence-enhancing shock filter that sharpens edges, robust multi-view color baking with detail injection for the ortho, and feather blending across cluster tiles. Set `dsm_enabled: false` to skip 2D maps entirely.

---

## Performance & disk usage

- **Run stages separately.** The four-command split keeps peak memory bounded; you can stop and resume between stages, and re-run a single stage after changing its config.
- **Disk reclamation.** Intermediate artefacts are removed by default once consumed (`depthmap_delete_raw_after_clean`, `depthmap_delete_fusion_batches`, `depthmap_fusion_mesh_delete_batches`). Set the relevant flag to `false` to keep them for debugging. See [Dense disk reclamation and export](configuration.md#dense-disk-reclamation-and-export).
- **Image size.** `depthmap_max_image_size` caps the processed resolution; `undistorted_image_max_size` caps the undistorted images.

---

## Visualizing the results

- Point cloud / mesh: open `fused.ply` or `mesh.ply` in [MeshLab](http://www.meshlab.net/) or any PLY viewer.
- DSM / ortho: open the GeoTIFFs in QGIS or any GIS tool.
- Web viewer: serve the `point_cloud/` octree (Potree-style tiles).
- Rerun: `export_rerun` for an interactive 3D session.

## See also

- [Pipeline commands](using.md#dense-reconstruction-dense_clustering-compute_depthmaps-fuse_depthmaps-dense_merging) — command reference.
- [Configuration reference](configuration.md) — Depth Estimation, Fusion, DSM and Orthophoto, Octree Tiling.
- [Georeferencing & GIS outputs](georeferencing.md) — output CRS, GCPs, and a large-scale aerial walkthrough.
- [Dense matching notes](dense_matching.md) — the underlying math.
