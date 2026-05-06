# pyre-strict
"""Four-phase cluster-based depthmap computation pipeline (PatchMatch).

Phase 1 – Raw depthmaps (GPU, per cluster):
  Each cluster of nearby views runs multi-scale PatchMatch with
  geometric consistency on GPU.  One thread-pool per device, each
  pool limited to _MAX_PARALLEL_PER_DEVICE concurrent clusters.
  Clusters are distributed round-robin with GPUs filled first.

Phase 2 – Cleaning (GPU, N-hop neighbors):
  Each view's raw depthmap is cleaned using an expanded neighbor set
  (N hops in the neighbor graph) so cleaning sees far more context
  than the small GPU clusters.

Phase 3 – Fusion (CPU, per cluster, N-hop neighbors):
  Cleaned depthmaps are fused per cluster into batch PLYs, each using
  N-hop expanded neighbors for the fusion neighbor set.

Phase 4 – Merge:
  All batch PLYs are concatenated into the final ``fused.ply``.

Processing order is determined by a tracks-graph coverage traversal so
spatial neighbours are close in sequence.
"""

import gc
import logging
import os
import queue
import threading
import time
import json

import igraph as ig
import leidenalg
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from opensfm import context, io, log, pydense, pypointcloud, pymap, pysfm, tracking, types
from opensfm.dataset import UndistortedDataSet

logger: logging.Logger = logging.getLogger(__name__)

# Maximum concurrent clusters running on a single device.
_MAX_PARALLEL_PER_DEVICE: int = 4


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════


def compute_depthmaps(
    data: UndistortedDataSet,
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> None:
    """Four-phase pipeline: raw (GPU clusters) → clean (GPU N-hop)
    → fuse (CPU per-cluster N-hop) → merge.
    """
    context.log_memory("compute_depthmaps start")
    config = data.config
    processes: int = config["processes"]
    num_neighbors: int = config["depthmap_num_neighbors"]

    # ─ 1.  Build clusters ────────────────────────────────────────────
    processable = list(set(reconstruction.shots.keys())
                       & set(graph.get_shot_ids()))
    if not processable:
        logger.warning("No processable shots found for depthmap computation")
        return

    # Try to load view clustering
    if os.path.exists(data.clusters_file()) and os.path.isfile(data.clusters_points_file()):
        clusters = data.load_clusters()
        super_points = data.load_clusters_points()
        logger.info(
            f"Loaded {len(clusters)} clusters from {data.clusters_file()}")
    else:
        clusters, sp_coords, sp_vis, _, covis = cluster_views(
            processable,
            graph,
            reconstruction,
            config["depthmap_cluster_max_size"],
            fuse_knn=15,
            fuse_radius_factor=0.5,
        )

        if config["depthmap_save_debug_ply"]:
            _save_cluster_debug_ply(
                data, clusters, reconstruction,
                sp_coords, sp_vis)

        super_points = covis.super_points
        data.save_clusters(clusters)
        data.save_clusters_points(super_points)
        logger.info(
            f"Cluster assignments saved to {data.clusters_file()}")

    # ── 1b. Per-cluster bounding boxes ─────────────────────────────────
    cluster_bboxes: List[Tuple[NDArray, NDArray]] = [
        _compute_cluster_bbox(cl, graph, reconstruction) for cl in clusters
    ]
    context.log_memory("compute cluster bounding boxes done")

    # ── 1c. Global neighbor selection (C++, multithreaded) ────────────
    #   SelectNeighbors uses the super-point visibility from the
    #   covisibility graph directly — no Python common-tracks dict needed.
    nbr_result: pysfm.NeighborResult = pysfm.select_neighbors(
        graph,
        reconstruction.map,
        super_points,
        processable,
        num_neighbors,
    )
    context.log_memory("C++ neighbor selection done")

    # Convert C++ string-based neighbor lists to pymap.Shot lists
    # (downstream expects List[pymap.Shot]).
    # Access the pybind maps once (each access copies the full C++ map).
    best_nbr_map = nbr_result.best_neighbors
    all_nbr_map = nbr_result.all_neighbors
    del nbr_result
    shots_view = reconstruction.shots
    best_neighbors: Dict[str, List[pymap.Shot]] = {}
    all_neighbors: Dict[str, List[pymap.Shot]] = {}
    for sid in processable:
        best_neighbors[sid] = [
            shots_view[nid] for nid in best_nbr_map.get(sid, [])]
        all_neighbors[sid] = [
            shots_view[nid] for nid in all_nbr_map.get(sid, [])]

    # Pre-compute depth ranges.
    depth_ranges: Dict[str, Tuple[float, float]] = {}
    for shot_id in processable:
        shot = reconstruction.shots[shot_id]
        mind, maxd = compute_depth_range(graph, reconstruction, shot, config)
        depth_ranges[shot_id] = (mind, maxd)

    logger.info("Per-cluster track graphs built; neighbors updated")

    context.log_memory("clustering done")

    # ── 3.  Discover OpenCL devices ─────────────────────────────────────
    num_devices = 0
    if pydense.DepthmapClusterEstimator.is_available():
        num_devices = pydense.DepthmapClusterEstimator.num_devices()
    if num_devices == 0:
        logger.warning("No OpenCL devices found — cannot compute depthmaps")
        return

    ignore_intel_device = config.get("opencl_ignore_intel_device", True)

    # Build ordered device list: GPUs only
    gpu_devs: List[int] = []
    for di in range(num_devices):
        dname = pydense.DepthmapClusterEstimator.device_name(di)
        if "Intel(R)" in dname and ignore_intel_device:
            logger.info(
                f"  Device {di}: {dname} (ignored by config, skipping)"
            )
            continue
        dmem = pydense.DepthmapClusterEstimator.device_memory_bytes(di)
        if pydense.DepthmapClusterEstimator.device_is_gpu(di):
            logger.info(
                f"  Device {di}: {dname} ({dmem / (1024**3):.1f} GiB)"
            )
            gpu_devs.append(di)
    device_order = gpu_devs

    total_slots = _MAX_PARALLEL_PER_DEVICE * len(gpu_devs)
    logger.info(
        f"{num_devices} device(s), {_MAX_PARALLEL_PER_DEVICE} GPU slots/device → {total_slots} total parallel cluster(s)"
    )

    # ── 4.  Phase 1: Per-cluster raw depthmaps ────────────────────────
    #   Shared work queue consumed independently by per-device workers.
    #   Each GPU runs _MAX_PARALLEL_PER_DEVICE threads that pull clusters on-demand
    processed_shots: Set[str] = set()
    errors: List[Tuple[int, Exception]] = []
    results_lock = threading.Lock()

    work_q: queue.Queue[Tuple[int, List[str]]] = queue.Queue()
    for ci, cluster_shots in enumerate(clusters):
        work_q.put((ci, cluster_shots))

    def _device_loop(device_idx: int) -> None:
        """Pull clusters from the shared queue until empty."""
        log.setup()
        while True:
            try:
                batch_num, cluster_shots = work_q.get_nowait()
            except queue.Empty:
                return
            try:
                if all(data.raw_depthmap_exists(s) or data.clean_depthmap_exists(s) for s in cluster_shots):
                    logger.info(
                        f"Cluster {batch_num}: all {len(cluster_shots)} raw depthmaps exist, skipping"
                    )
                    with results_lock:
                        processed_shots.update(cluster_shots)
                    continue
                context.log_memory(
                    f"raw cluster {batch_num} start (dev {device_idx})"
                )
                ret = _run_cluster_raw(
                    data, graph, reconstruction, best_neighbors,
                    cluster_shots, depth_ranges, batch_num,
                    device_idx=device_idx,
                )
                context.log_memory(f"raw cluster {batch_num} end")
                with results_lock:
                    processed_shots.update(ret)
                logger.info(
                    f"Cluster {batch_num} raw done on dev {device_idx} ({len(processed_shots)}/{len(processable)} views)"
                )
            except Exception as exc:
                logger.error(
                    f"Cluster batch {batch_num} failed", exc_info=True,
                )
                with results_lock:
                    errors.append((batch_num, exc))

    # Launch per-device threads.
    threads: List[threading.Thread] = []
    for di in device_order:
        per_device = _MAX_PARALLEL_PER_DEVICE
        for wi in range(per_device):
            t = threading.Thread(
                target=_device_loop, args=(di,),
                name=f"dev{di}-w{wi}", daemon=True,
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    if errors:
        logger.warning(f"{len(errors)} cluster(s) failed in raw phase:")
        for batch_num, exc in errors:
            logger.warning(f"  batch {batch_num}: {exc}")

    context.log_memory("phase 1 raw depthmaps done")

    # ── 5.  Phase 2: GPU cleaning with N-hop neighbors ────────────────
    #   Views are loaded from disk in batches, cleaned on GPU, and
    #   saved back.  Parallel workers overlap I/O with GPU compute.
    _clean_views_batched_gpu(
        data, reconstruction, clusters, all_neighbors, processable, config,
        device_order=device_order,
    )

    context.log_memory("phase 2 cleaning done")

    # Remove raw depthmaps — no longer needed after cleaning.
    for sid in processable:
        data.io_handler.rm_if_exist(data.depthmap_file(sid, "raw.npz"))
    logger.info("Removed raw depthmaps")

    # ── 6.  Phase 3: Per-cluster fusion with N-hop neighbors ──────────
    #   Each cluster loads its cleaned depthmaps from disk on-demand.
    #   Clusters run in parallel to overlap I/O with CPU fusion.
    _fuse_per_cluster(
        data, all_neighbors, clusters, cluster_bboxes,
        processable, config, reconstruction, depth_ranges,
    )

    # ── 7.  Phase 4: Merge batch PLYs into final fused.ply ────────────
    context.log_memory("phase 3 fusion done")
    _merge_fusion_batches(data, list(range(len(clusters))))

    context.log_memory("phase 4 merge done")

    # Build octree tiles for the viewer.
    _export_octree_tiles(data, config)

    context.log_memory("compute_depthmaps end")


# ═══════════════════════════════════════════════════════════════════════
#  View clustering for geometric consistency
# ═══════════════════════════════════════════════════════════════════════


def cluster_views(
    processable: List[str],
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    max_cluster_size: int,
    fuse_knn: int = 15,
    fuse_radius_factor: float = 0.5,
) -> Tuple[List[List[str]], NDArray, List[Set[str]], List[List[str]], "pysfm.CovisibilityGraph"]:
    """Cluster shots by shared super-point visibility (Leiden/CPM).

    Steps 1–3 (super-point fusion, kNN, pairwise covisibility) are
    delegated to C++ via ``pysfm.build_covisibility_graph`` for speed
    and low memory usage.  Step 4 (Leiden partitioning + small-cluster
    absorption) remains in Python.

    Returns ``(clusters, sp_coords, sp_vis, sp_track_ids, covis)``
    where *covis* is the C++ ``CovisibilityGraph`` object (needed for
    downstream neighbor selection).
    """
    N = len(processable)

    # ── C++ super-point fusion + covisibility graph (multithreaded) ──
    covis: pysfm.CovisibilityGraph = pysfm.build_covisibility_graph(
        tracks_manager,
        reconstruction.map,
        processable,
        fuse_knn,
        fuse_radius_factor,
    )

    # Unpack super-points for downstream debug PLY and other consumers.
    sp_coords_list: List[NDArray] = []
    sp_vis_list: List[Set[str]] = []
    sp_track_ids_list: List[List[str]] = []
    for sp in covis.super_points:
        sp_coords_list.append(np.asarray(sp.coord, dtype=np.float64))
        sp_vis_list.append(set(sp.vis))
        sp_track_ids_list.append(list(sp.tracks))
    sp_coords_arr = (
        np.array(sp_coords_list)
        if sp_coords_list
        else np.empty((0, 3), dtype=np.float64)
    )

    logger.info(
        f"C++ covisibility: {len(covis.super_points)} super-points, "
        f"{len(covis.edges)} edges"
    )

    if N <= max_cluster_size:
        return [list(processable)], sp_coords_arr, sp_vis_list, sp_track_ids_list, covis
    if N <= 1 or max_cluster_size <= 1:
        return (
            [[s] for s in processable],
            sp_coords_arr, sp_vis_list, sp_track_ids_list, covis,
        )

    # ── Leiden partitioning (Python — igraph/leidenalg, C under hood) ─
    edges = covis.edges
    weights = covis.weights

    if not edges:
        logger.warning("No covisibility edges — single cluster")
        return [list(processable)], sp_coords_arr, sp_vis_list, sp_track_ids_list, covis

    ig_graph = ig.Graph(n=N, edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    # Build a quick weight lookup for the absorption step.
    pair_weight: Dict[Tuple[int, int], float] = {}
    for (ei, ej), ew in zip(edges, weights):
        pair_weight[(ei, ej)] = ew
        pair_weight[(ej, ei)] = ew

    mean_w = float(np.mean(weights)) if weights else 1.0
    resolution = mean_w * 0.5

    context.log_memory("starting Leiden partitioning")

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        max_comm_size=int(max_cluster_size),
        n_iterations=-1,
        seed=42,
    )

    part_map: Dict[int, List[str]] = {}
    for i, comm in enumerate(partition.membership):
        part_map.setdefault(comm, []).append(processable[i])
    final = list(part_map.values())
    logger.info(
        f"Leiden partitioning: {N} shots → {len(final)} communities, "
        f"quality {partition.quality():.1f}, resolution {resolution:.1f}"
    )

    # Absorb tiny clusters into best neighbour.
    idx_of: Dict[str, int] = {sid: i for i, sid in enumerate(processable)}
    changed = True
    while changed:
        changed = False
        for ci in range(len(final)):
            if len(final[ci]) >= int(max_cluster_size * 0.75):
                continue
            logger.info(
                f"Cluster {ci} is too small (size {len(final[ci])}), "
                f"absorbing into best neighbour"
            )
            best_ti = -1
            best_sim_val = -1.0
            for ti in range(len(final)):
                if ti == ci:
                    continue
                s = sum(
                    pair_weight.get((idx_of[a], idx_of[b]), 0.0)
                    for a in final[ci]
                    for b in final[ti]
                )
                if s > best_sim_val:
                    best_sim_val = s
                    best_ti = ti
            if best_ti < 0:
                continue
            final[best_ti].extend(final[ci])
            logger.info(
                f"  Absorbed cluster {ci} (size {len(final[ci])}) into cluster {best_ti} (new size {len(final[best_ti])})"
            )
            del final[ci]
            changed = True
            break

    sizes = [len(c) for c in final]
    logger.info(
        f"Super-point clustering: {N} shots → {len(final)} clusters, "
        f"sizes {sizes}",
    )
    return final, sp_coords_arr, sp_vis_list, sp_track_ids_list, covis


def _save_cluster_debug_ply(
    data: UndistortedDataSet,
    clusters: List[List[str]],
    reconstruction: types.Reconstruction,
    sp_coords: NDArray,
    sp_vis: List[Set[str]],
    frustum_step: float = 0.005,
) -> None:
    """Write one debug PLY per cluster with colored camera frustums,
    white neighbor frustums, and super-points.

    Each super-point is colored by the cluster(s) it is observed from.
    A super-point observed by cameras in cluster *i* gets cluster *i*'s
    color.  If observed by multiple clusters it gets a blended color.
    """
    # Build shot → cluster index lookup.
    shot_cluster: Dict[str, int] = {}
    for ci, cl in enumerate(clusters):
        for sid in cl:
            shot_cluster[sid] = ci

    # Pre-assign each super-point a dominant cluster (majority vote)
    # and its blended color.
    sp_cluster: List[int] = []  # dominant cluster per super-point
    sp_color: List[Tuple[int, int, int]] = []
    for vis in sp_vis:
        votes: Dict[int, int] = {}
        for sid in vis:
            ci = shot_cluster.get(sid, -1)
            if ci >= 0:
                votes[ci] = votes.get(ci, 0) + 1
        if not votes:
            sp_cluster.append(-1)
            sp_color.append((128, 128, 128))
            continue
        # Dominant cluster.
        dom = max(votes, key=lambda c: votes[c])
        sp_cluster.append(dom)
        # Blend colors weighted by vote count.
        total = sum(votes.values())
        r = sum(votes[c] * _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)][0]
                for c in votes) // total
        g = sum(votes[c] * _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)][1]
                for c in votes) // total
        b = sum(votes[c] * _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)][2]
                for c in votes) // total
        sp_color.append((r, g, b))

    for ci, cl in enumerate(clusters):
        cr, cg, cb = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
        color = np.array([cr, cg, cb], dtype=np.uint8)

        cl_pts: List[NDArray] = []
        cl_colors: List[NDArray] = []

        # Camera frustum wireframes.
        for sid in cl:
            if sid not in reconstruction.shots:
                continue
            shot = reconstruction.shots[sid]
            frustum_depth = 5
            fpts = _sample_frustum_wireframe(shot, depth=frustum_depth,
                                             step=frustum_step)
            cl_pts.append(fpts)
            cl_colors.append(np.tile(color, (len(fpts), 1)))

        # Super-points whose dominant cluster is this one.
        if len(sp_coords) > 0:
            mask = [i for i, dom in enumerate(sp_cluster) if dom == ci]
            if mask:
                sp_pts = sp_coords[mask].astype(np.float32)
                sp_cols = np.array(
                    [sp_color[i] for i in mask], dtype=np.uint8,
                )
                cl_pts.append(sp_pts)
                cl_colors.append(sp_cols)

        if not cl_pts:
            continue

        points = np.concatenate(cl_pts).astype(np.float32)
        colors = np.concatenate(cl_colors).astype(np.uint8)
        normals = np.zeros_like(points)
        labels = np.zeros(len(points), dtype=np.uint8)
        filename = f"cluster_debug_{ci:04d}.ply"
        data.save_point_cloud(
            points, normals, colors, labels, filename=filename,
        )
        logger.info(
            f"Cluster {ci} debug PLY saved ({len(points)} points, "
            f"{len(cl)} shots)"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Cluster utilities
# ═══════════════════════════════════════════════════════════════════════


def _compute_cluster_bbox(
    cluster_shots: List[str],
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> Tuple[NDArray, NDArray]:
    """Axis-aligned bounding box of 3-D points visible in the cluster.

    Uses the tracks manager to collect every reconstruction point
    observed by at least one shot in the cluster.  Falls back to
    camera positions if no points are found.

    Returns ``(min_coords, max_coords)``, each shape ``(3,)``.
    """
    seen_tracks: Set[str] = set()
    for sid in cluster_shots:
        for tid in graph.get_shot_observations(sid):
            if tid in reconstruction.points:
                seen_tracks.add(tid)

    if seen_tracks:
        coords = np.array(
            [reconstruction.points[tid].coordinates for tid in seen_tracks]
        )
    else:
        coords = np.array(
            [reconstruction.shots[sid].pose.get_origin()
             for sid in cluster_shots]
        )
    return coords.min(axis=0), coords.max(axis=0)


def build_common_tracks(
    cluster_shots: List[str],
    all_neighbors: Dict[str, List[pymap.Shot]],
    sp_vis: List[Set[str]],
    sp_track_ids: List[List[str]],
) -> Dict[str, Dict[str, List[str]]]:
    """Build a per-cluster common-tracks dict from super-point visibility.

    For each super-point/singleton whose visibility intersects the
    expanded camera set, all underlying track IDs are recorded as
    common tracks between every pair of cameras that see it.

    The camera set is limited to the cluster images, so only views
    within the cluster are considered for common tracks.

    Returns a dict of dicts:

        result[im1][im2] = [track_id, ...]
    """

    expanded: Set[str] = set()
    for sid in cluster_shots:
        expanded.add(sid)
        for nbr in all_neighbors.get(sid, []):
            expanded.add(nbr.id)

    res: Dict[str, Dict[str, List[str]]] = {sid: {} for sid in expanded}
    for vis, tids in zip(sp_vis, sp_track_ids):
        # Cameras in the expanded set that see this super-point.
        cams = sorted(vis & expanded)
        nc = len(cams)
        if nc < 2:
            continue
        for ci in range(nc):
            for cj in range(ci + 1, nc):
                s1, s2 = cams[ci], cams[cj]
                res[s1].setdefault(s2, []).extend(tids)
                res[s2].setdefault(s1, []).extend(tids)

    return res


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Per-cluster raw depthmaps (GPU)
# ═══════════════════════════════════════════════════════════════════════


def _setup_cluster_params(
    cluster: Any,
    config: Dict[str, Any],
) -> None:
    """Apply shared PatchMatch parameters to a cluster object."""
    cluster.set_max_iterations(config["depthmap_max_iterations"])
    cluster.set_patch_size(config["depthmap_patch_size"])
    cluster.set_max_image_size(config["depthmap_max_image_size"])
    cluster.set_sigma_spatial(config["depthmap_sigma_spatial"])
    cluster.set_sigma_color(config["depthmap_sigma_color"])
    cluster.set_top_k(config["depthmap_num_matching_views"])
    cluster.set_census_weight(config["depthmap_census_weight"])
    cluster.set_hierarchy_levels(config["depthmap_hierarchy_levels"])
    cluster.set_smooth_weight(config["depthmap_smooth_weight"])
    cluster.set_geom_consistency_weight(
        config["depthmap_geom_consistency_weight"]
    )


def _run_cluster_raw(
    data: UndistortedDataSet,
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    best_neighbors: Dict[str, List[pymap.Shot]],
    cluster_shot_ids: List[str],
    depth_ranges: Dict[str, Tuple[float, float]],
    batch_num: int,
    device_idx: int = 0,
) -> List[str]:
    """Run raw depthmaps (no cleaning, no fusion) for a cluster.

    Saves raw depthmaps to disk immediately and returns the list of
    processed shot IDs.  This avoids accumulating all cluster results
    in memory.
    """
    config = data.config

    if not pydense.DepthmapClusterEstimator.is_available():
        raise RuntimeError(
            "PatchMatch requires OpenCL but no OpenCL device was found."
        )

    logger.info(
        f"Cluster {batch_num} raw ({len(cluster_shot_ids)} views) on device {device_idx}: {', '.join(cluster_shot_ids)}"
    )

    num_neighbors: int = config["depthmap_num_matching_views"]
    max_size: int = config["depthmap_max_image_size"]

    cluster = pydense.DepthmapClusterEstimator()
    _setup_cluster_params(cluster, config)
    cluster.set_device(device_idx)

    # -- Build cluster: add ref views and their source views --
    ref_idx_map: Dict[str, int] = {}
    ref_nbr_ids: Dict[int, List[str]] = {}

    for ref_id in cluster_shot_ids:
        neighbors = best_neighbors[ref_id]
        shot = reconstruction.shots[ref_id]
        min_d, max_d = depth_ranges[ref_id]

        # Load and prepare the reference image (grayscale + color).
        ref_gray = cv2.cvtColor(
            data.load_undistorted_image(shot.id), cv2.COLOR_RGB2GRAY)
        oh, ow = ref_gray.shape
        if ow > max_size or oh > max_size:
            factor = max_size / max(ow, oh)
            w, h = int(ow * factor), int(oh * factor)
        else:
            w, h = ow, oh
        image = scale_down_image(ref_gray, w, h)
        K = shot.camera.get_K_in_pixel_coordinates(w, h)
        R = shot.pose.get_rotation_matrix()
        t = shot.pose.translation

        idx = cluster.begin_ref_view(K, R, t, image, min_d, max_d)
        ref_idx_map[ref_id] = idx

        # SfM planar prior.
        if config["depthmap_sfm_planar_prior"]:
            sfm_points: List[NDArray] = []
            for track in graph.get_shot_observations(shot.id):
                if track in reconstruction.points:
                    sfm_points.append(reconstruction.points[track].coordinates)
            if sfm_points:
                cluster.set_sfm_points(np.array(sfm_points, dtype=np.float64))

        # Add source views.
        nbr_ids: List[str] = [ref_id]
        for s in neighbors[1: num_neighbors + 1]:
            assert s.camera.projection_type == "perspective"
            src_gray = cv2.cvtColor(
                data.load_undistorted_image(s.id), cv2.COLOR_RGB2GRAY)
            soh, sow = src_gray.shape
            if sow > max_size or soh > max_size:
                factor = max_size / max(sow, soh)
                sw, sh = int(sow * factor), int(soh * factor)
            else:
                sw, sh = sow, soh
            src_img = scale_down_image(src_gray, sw, sh)
            sK = s.camera.get_K_in_pixel_coordinates(sw, sh)
            sR = s.pose.get_rotation_matrix()
            st = s.pose.translation
            cluster.add_source_view(sK, sR, st, src_img)
            nbr_ids.append(s.id)

        ref_nbr_ids[idx] = nbr_ids

    context.log_memory("cluster data loaded")

    # -- Register geometric consistency links --
    for ref_id, idx in ref_idx_map.items():
        nbr_ids = ref_nbr_ids[idx]
        for other_id, other_idx in ref_idx_map.items():
            if other_id == ref_id:
                continue
            if other_id in nbr_ids:
                src_pos = nbr_ids.index(other_id) - 1
                if src_pos >= 0:
                    cluster.add_geom_link(idx, src_pos, other_idx)

    # -- Run raw depthmaps only --
    results = cluster.run()
    cluster.clear()

    context.log_memory("cluster raw depthmaps computed")

    # -- Save raw depthmaps to disk in parallel --
    processed: List[str] = []
    save_items = []
    max_cost: float = config["depthmap_max_cost"]
    for i, ref_id in enumerate(cluster_shot_ids):
        depth, normal, cost, confidence = results[i]
        depth = np.asarray(depth, dtype=np.float32)
        normal = np.asarray(normal, dtype=np.float32)
        cost = np.asarray(cost, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)

        # Sanitize NaN/Inf depths: zero out invalid pixels.
        nan_mask = ~np.isfinite(depth)
        n_nan = int(np.count_nonzero(nan_mask))
        if n_nan > 0:
            logger.warning(
                f"Raw depthmap {ref_id}: {n_nan} NaN/Inf pixels zeroed"
            )
            depth[nan_mask] = 0.0
            # Zero out corresponding normals too.
            nan_mask_3 = np.broadcast_to(
                nan_mask[:, :, np.newaxis], normal.shape
            )
            normal[nan_mask_3] = 0.0

        # Zero out textureless / low-confidence pixels
        if max_cost > 0:
            high_cost = cost > max_cost
            n_high = int(np.count_nonzero(high_cost))
            if n_high > 0:
                depth[high_cost] = 0.0
                logger.debug(
                    f"Raw depthmap {ref_id}: {n_high} pixels zeroed "
                    f"(cost > {max_cost:.2f})"
                )

        nghbrs = [n.id for n in best_neighbors.get(ref_id, [])]
        score = cost
        nghbr = np.zeros((0,), dtype=np.int32)
        save_items.append((ref_id, depth, normal, score, nghbr, nghbrs,
                           confidence))
        processed.append(ref_id)

    data.save_raw_depthmaps_parallel(save_items)

    # Export each raw depthmap as a binary PLY.
    if config["depthmap_save_debug_ply"]:
        for ref_id, dep, _nrm, _sc, _ng, _ngs, _conf in save_items:
            _save_depthmap_as_ply(data, reconstruction, ref_id, dep, "raw")

    del save_items
    del cluster
    del results
    gc.collect()

    context.log_memory("cluster raw depthmaps saved to disk")

    logger.info(
        f"Cluster {batch_num}: raw {len(processed)} views saved to disk"
    )
    return processed


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: GPU cleaning with N-hop neighbors
# ═══════════════════════════════════════════════════════════════════════


def _clean_views_batched_gpu(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    clusters: List[List[str]],
    all_neighbors: Dict[str, List[pymap.Shot]],
    processable: List[str],
    config: Dict[str, Any],
    max_neighbors: int = 16,
    device_order: List[int] | None = None,
) -> None:
    """Clean all raw depthmaps on GPU using N-hop expanded neighbors.

    Clusters are distributed across GPU devices via a shared work queue,
    each device running ``_MAX_PARALLEL_PER_DEVICE`` worker threads.
    Each worker creates its own ``GPUDepthmapCleaner`` bound to one device
    so multiple GPUs work in parallel.
    """
    available = [
        sid for sid in processable if data.raw_depthmap_exists(sid)
    ]
    if not available:
        return

    available_set: Set[str] = set(available)
    if device_order is None:
        device_order = [0]

    def _gpu_clean_and_save_cluster(
        cleaner: Any,
        cluster_shots: List[str],
    ) -> int:
        """Load views, GPU-clean, save results, release everything.

        The *cleaner* is reused across batches (kernel compiled once);
        ``clear()`` is called at entry to discard the previous batch's
        views.  All loaded data is released before returning so memory
        stays flat between batches.

        The C++ cleaner has a hard limit of 16 neighbor views per ref
        (kMaxCleanSources).  We select the best ``max_neighbors``
        neighbors per ref from ``all_neighbors`` (already ranked by
        overlap quality), and only load the union of needed views.
        """
        cleaner.clear()

        # Per-ref neighbor set, capped at max_neighbors, filtered to
        # shots that actually have a raw depthmap.
        per_ref_nbrs: Dict[str, List[str]] = {}
        needed: Set[str] = set(cluster_shots)
        for sid in cluster_shots:
            nbrs = [
                n.id for n in all_neighbors.get(sid, [])
                if n.id != sid and n.id in available_set
            ][:max_neighbors]
            per_ref_nbrs[sid] = nbrs
            needed.update(nbrs)

        # Deterministic ordering so cleaner indices are stable.
        ordered_shots = sorted(needed)
        shot_to_idx: Dict[str, int] = {
            sid: i for i, sid in enumerate(ordered_shots)
        }

        logger.info(
            f"GPU clean cluster: {len(cluster_shots)} ref shots, "
            f"{len(ordered_shots) - len(cluster_shots)} neighbors, "
            f"{len(ordered_shots)} total"
        )

        # ── Load raw depthmaps in parallel & feed the GPU cleaner ──
        raw_normals: Dict[str, NDArray] = {}
        raw_confidence: Dict[str, Optional[NDArray]] = {}
        cluster_shots_set = set(cluster_shots)

        loaded = data.load_raw_depthmaps_parallel(ordered_shots)
        for sid, (depth, normal, _, _, _, confidence) in zip(
            ordered_shots, loaded
        ):
            shot = reconstruction.shots[sid]
            h, w = depth.shape[:2]
            K = shot.camera.get_K_in_pixel_coordinates(w, h)
            R = shot.pose.get_rotation_matrix()
            t = shot.pose.translation
            cleaner.add_view(K, R, t, depth)

            if sid in cluster_shots_set:
                raw_normals[sid] = normal
                raw_confidence[sid] = confidence

        del loaded

        # ── Clean each ref view and save to disk immediately ───────
        save_items = []
        for sid in cluster_shots:
            nbr_indices = np.array(
                [shot_to_idx[n] for n in per_ref_nbrs[sid]],
                dtype=np.int32,
            )
            cleaned_depth = cleaner.clean(shot_to_idx[sid], nbr_indices)
            cleaned_arr = np.asarray(cleaned_depth, dtype=np.float32)
            norm = raw_normals.pop(sid)
            conf = raw_confidence.pop(sid, None)
            score = np.zeros_like(cleaned_arr)
            save_items.append((sid, cleaned_arr, norm, score, conf))

        data.save_clean_depthmaps_parallel(save_items)

        # Export each clean depthmap as a binary PLY.
        if config["depthmap_save_debug_ply"]:
            for sid, cleaned, _norm, _sc, _conf in save_items:
                _save_depthmap_as_ply(
                    data, reconstruction, sid, cleaned, "clean")

        # Release everything for this batch.
        del raw_normals
        cleaner.clear()
        gc.collect()
        return len(save_items)

    # -- parallel GPU pipeline via shared work queue -----------------------
    cleaned_count = 0
    clean_errors: List[Tuple[int, Exception]] = []
    clean_lock = threading.Lock()

    work_q: queue.Queue[Tuple[int, List[str]]] = queue.Queue()
    for bi, cluster in enumerate(clusters):
        cluster_valid = list(set(cluster) & available_set)
        if not cluster_valid:
            continue
        if all(data.clean_depthmap_exists(sid) for sid in cluster_valid):
            logger.info(
                f"Clean batch {bi} skipped: all shots already cleaned"
            )
            continue
        work_q.put((bi, cluster_valid))

    total_work = work_q.qsize()
    if total_work == 0:
        logger.info("GPU cleaning: nothing to do")
        return

    context.log_memory("starting GPU cleaning batches")

    def _clean_device_loop(device_idx: int) -> None:
        """Pull clusters from the shared queue until empty."""
        log.setup()
        cleaner = pydense.GPUDepthmapCleaner()
        cleaner.set_same_depth_threshold(
            config["depthmap_same_depth_threshold"]
        )
        cleaner.set_min_consistent_views(
            config["depthmap_min_consistent_views"]
        )
        cleaner.set_carving_threshold(
            config["depthmap_carving_threshold"]
        )
        cleaner.set_max_carved_views(
            config["depthmap_max_carved_views"]
        )
        cleaner.set_device(device_idx)
        try:
            while True:
                try:
                    bi, cluster_valid = work_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    context.log_memory(
                        f"clean batch {bi} start (dev {device_idx})"
                    )
                    n = _gpu_clean_and_save_cluster(cleaner, cluster_valid)
                    context.log_memory(f"clean batch {bi} end")
                    with clean_lock:
                        nonlocal cleaned_count
                        cleaned_count += n
                    logger.info(
                        f"Clean batch {bi} done ({n} views) on dev {device_idx}, "
                        f"{cleaned_count}/{len(available)} total"
                    )
                except Exception as exc:
                    logger.error(
                        f"Clean batch {bi} failed on dev {device_idx}",
                        exc_info=True,
                    )
                    with clean_lock:
                        clean_errors.append((bi, exc))
        finally:
            del cleaner
            gc.collect()

    # Launch one thread per device (the GPU cleaner holds an OpenCL
    # context that cannot be shared across threads on the same device).
    threads: List[threading.Thread] = []
    for di in device_order:
        t = threading.Thread(
            target=_clean_device_loop, args=(di,),
            name=f"clean-dev{di}", daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if clean_errors:
        logger.warning(
            f"{len(clean_errors)} cluster(s) failed in clean phase:"
        )
        for bi, exc in clean_errors:
            logger.warning(f"  batch {bi}: {exc}")

    logger.info(
        f"GPU cleaning done: {cleaned_count} views",
    )
    context.log_memory("GPU cleaning complete")


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3: Per-cluster fusion with N-hop neighbors
# ═══════════════════════════════════════════════════════════════════════


# Distinct colors for up to 12 clusters; cycles for more.
_CLUSTER_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 255, 128),    # spring green
    (255, 0, 128),    # rose
    (128, 255, 0),    # lime
    (0, 128, 255),    # sky blue
]


def _sample_edges_wireframe(
    corners: NDArray,
    edges: List[Tuple[int, int]],
    step: float = 0.01,
) -> NDArray:
    """Sample points along edges defined by pairs of corner indices."""
    pts: List[NDArray] = []
    for i, j in edges:
        length = float(np.linalg.norm(corners[j] - corners[i]))
        n_samples = max(int(np.ceil(length / step)), 2)
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[:, None]
        pts.append(corners[i] + t * (corners[j] - corners[i]))
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def _sample_bbox_wireframe(
    bbox: Tuple[NDArray, NDArray],
    step: float = 0.01,
) -> NDArray:
    """Sample points along the 12 edges of an axis-aligned bounding box."""
    bb_min, bb_max = bbox
    corners = np.array(
        [
            [bb_min[0], bb_min[1], bb_min[2]],
            [bb_max[0], bb_min[1], bb_min[2]],
            [bb_max[0], bb_max[1], bb_min[2]],
            [bb_min[0], bb_max[1], bb_min[2]],
            [bb_min[0], bb_min[1], bb_max[2]],
            [bb_max[0], bb_min[1], bb_max[2]],
            [bb_max[0], bb_max[1], bb_max[2]],
            [bb_min[0], bb_max[1], bb_max[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    return _sample_edges_wireframe(corners, edges, step)


def _sample_frustum_wireframe(
    shot: pymap.Shot,
    depth: float = 0.3,
    step: float = 0.01,
) -> NDArray:
    """Sample points along the 8 edges of a camera frustum.

    The frustum is defined by the camera origin and four far-plane
    corners at the given *depth* along the viewing direction.
    """
    R = shot.pose.get_rotation_matrix()
    t = shot.pose.translation
    origin = (-R.T @ t).astype(np.float32)  # camera center in world

    w = int(shot.camera.width)
    h = int(shot.camera.height)
    K = shot.camera.get_K_in_pixel_coordinates(w, h)
    K_inv = np.linalg.inv(K)

    # Four image corners in pixel coords → normalized camera rays.
    pixel_corners = np.array(
        [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]],
        dtype=np.float64,
    ).T  # (3, 4)
    cam_rays = K_inv @ pixel_corners  # (3, 4)
    # Normalize so Z=1, then scale by depth.
    cam_rays = cam_rays / cam_rays[2:3, :] * depth
    # Transform to world coordinates.
    world_pts = (R.T @ (cam_rays - t[:, np.newaxis])).T.astype(np.float32)

    # corners: 0=origin, 1..4=far plane TL, TR, BR, BL
    corners = np.vstack([origin[np.newaxis, :], world_pts])  # (5, 3)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # origin to far corners
        (1, 2), (2, 3), (3, 4), (4, 1),  # far plane rectangle
    ]
    return _sample_edges_wireframe(corners, edges, step)


# ═══════════════════════════════════════════════════════════════════════
#  Sub-volume splitting for SVO fusion
# ═══════════════════════════════════════════════════════════════════════


class _SubVolume(NamedTuple):
    """Axis-aligned sub-volume for bounded SVO fusion."""
    core_min: NDArray   # (3,) world-space core bbox min
    core_max: NDArray   # (3,) world-space core bbox max
    ext_min: NDArray    # (3,) extended bbox (core + margin)
    ext_max: NDArray    # (3,) extended bbox (core + margin)
    view_ids: Set[str]  # views contributing to this sub-volume


def _augment_cluster_views(
    cluster_shots: List[str],
    all_neighbors: Dict[str, List[Any]],
    fusable_set: Set[str],
    n_augment: int,
) -> List[str]:
    """Expand cluster with top-N neighbor shots per cluster shot.

    Neighbors are already ranked by shared visibility in ``all_neighbors``.
    Returns sorted augmented view list.
    """
    augmented: Set[str] = set(cluster_shots)
    for sid in cluster_shots:
        count = 0
        for nbr in all_neighbors.get(sid, []):
            nid = nbr.id if hasattr(nbr, "id") else nbr
            if nid != sid and nid in fusable_set and nid not in augmented:
                augmented.add(nid)
                count += 1
                if count >= n_augment:
                    break
    return sorted(augmented)


def _prescan_coarse_grid(
    data: UndistortedDataSet,
    view_ids: List[str],
    reconstruction: types.Reconstruction,
    voxel_size: float,
    coarse_factor: int,
    subsample: int = 8,
) -> Dict[Tuple[int, int, int], Set[int]]:
    """CPU pre-scan: subsample cleaned depthmaps, project to world, bucket.

    For each view, loads the cleaned depth, subsamples, back-projects
    to world coordinates, and records which views contribute to each
    coarse grid cell.

    Args:
        data: dataset for loading cleaned depthmaps.
        view_ids: ordered list of shot IDs to scan.
        reconstruction: for camera parameters.
        voxel_size: SVO fine voxel size.
        coarse_factor: coarse cell = coarse_factor * voxel_size.
        subsample: take every Nth pixel in each dimension.

    Returns:
        Grid mapping coarse cell coordinate to set of view indices
        (indices into ``view_ids``).
    """
    coarse_size = voxel_size * coarse_factor
    inv_coarse = 1.0 / coarse_size
    grid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)

    for view_idx, sid in enumerate(view_ids):
        if not data.clean_depthmap_exists(sid):
            continue
        depth, _normal, _score, _conf = data.load_clean_depthmap(sid)
        h, w = depth.shape

        # Subsample pixel coordinates.
        rs = np.arange(0, h, subsample)
        cs = np.arange(0, w, subsample)
        rr, cc = np.meshgrid(rs, cs, indexing="ij")
        rr = rr.ravel()
        cc = cc.ravel()
        d = depth[rr, cc]
        valid = d > 0
        rr, cc, d = rr[valid], cc[valid], d[valid]
        del depth  # release full depth

        if len(d) == 0:
            continue

        shot = reconstruction.shots[sid]
        K = shot.camera.get_K_in_pixel_coordinates(w, h)
        R = shot.pose.get_rotation_matrix()
        t_vec = shot.pose.translation

        Kinv = np.linalg.inv(K)
        pts_px = np.vstack(
            [cc.astype(np.float64), rr.astype(np.float64), np.ones(len(d))]
        )
        pts_cam = Kinv @ pts_px * d
        Rinv = R.T
        pts_world = Rinv @ (pts_cam - t_vec[:, None])

        cx = np.floor(pts_world[0] * inv_coarse).astype(np.int32)
        cy = np.floor(pts_world[1] * inv_coarse).astype(np.int32)
        cz = np.floor(pts_world[2] * inv_coarse).astype(np.int32)

        # Unique coarse cells touched by this view.
        keys = np.column_stack([cx, cy, cz])
        unique_keys = np.unique(keys, axis=0)
        for ukey in unique_keys:
            grid[(int(ukey[0]), int(ukey[1]), int(ukey[2]))].add(view_idx)

    return grid


def _split_into_subvolumes(
    grid: Dict[Tuple[int, int, int], Set[int]],
    view_ids: List[str],
    voxel_size: float,
    coarse_factor: int,
    max_voxels: int,
    trunc_factor: float,
) -> List[_SubVolume]:
    """Split occupied coarse cells into sub-volumes fitting GPU budget.

    Uses recursive KD-tree style median-split along the longest axis.
    Each sub-volume gets a margin of ``trunc_factor * voxel_size`` for
    correct TSDF at boundaries.  Views are assigned to a sub-volume if
    they have coarse cells inside its extended (core + margin) bbox.
    """
    if not grid:
        return []

    coarse_size = voxel_size * coarse_factor
    voxels_per_cell = coarse_factor ** 3
    margin_world = trunc_factor * voxel_size
    margin_cells = int(np.ceil(trunc_factor / coarse_factor))
    max_cells = max(1, max_voxels // voxels_per_cell)

    all_coords = np.array(list(grid.keys()), dtype=np.int32)  # (N, 3)
    all_keys = list(grid.keys())

    def _make_subvolume(cell_indices: NDArray) -> _SubVolume:
        """Build a SubVolume from a set of coarse cell indices."""
        cells = all_coords[cell_indices]
        cmin = cells.min(axis=0)
        cmax = cells.max(axis=0)

        core_min = cmin.astype(np.float64) * coarse_size
        core_max = (cmax + 1).astype(np.float64) * coarse_size
        ext_min = (core_min - margin_world).astype(np.float32)
        ext_max = (core_max + margin_world).astype(np.float32)
        core_min = core_min.astype(np.float32)
        core_max = core_max.astype(np.float32)

        # Gather views from all cells in the extended bbox (core + margin).
        ext_cmin = cmin - margin_cells
        ext_cmax = cmax + margin_cells
        sv_views: Set[str] = set()
        for idx in range(len(all_coords)):
            c = all_coords[idx]
            if np.all(c >= ext_cmin) and np.all(c <= ext_cmax):
                for vi in grid[all_keys[idx]]:
                    sv_views.add(view_ids[vi])

        return _SubVolume(
            core_min=core_min,
            core_max=core_max,
            ext_min=ext_min,
            ext_max=ext_max,
            view_ids=sv_views,
        )

    def _split(cell_indices: NDArray) -> List[_SubVolume]:
        if len(cell_indices) <= max_cells:
            return [_make_subvolume(cell_indices)]

        cells = all_coords[cell_indices]
        spans = cells.max(axis=0) - cells.min(axis=0)
        axis = int(np.argmax(spans))

        if spans[axis] == 0:
            # Can't split further — accept over-budget.
            return [_make_subvolume(cell_indices)]

        median = int(np.median(cells[:, axis]))
        left_mask = cells[:, axis] <= median
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            # All cells on one side; force split at midpoint.
            mid = (cells[:, axis].min() + cells[:, axis].max()) // 2
            left_mask = cells[:, axis] <= mid
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                return [_make_subvolume(cell_indices)]

        return (
            _split(cell_indices[left_mask])
            + _split(cell_indices[right_mask])
        )

    return _split(np.arange(len(all_coords)))


def _fuse_per_cluster(
    data: UndistortedDataSet,
    all_neighbors: Dict[str, List[pymap.Shot]],
    clusters: List[List[str]],
    cluster_bboxes: List[Tuple[NDArray, NDArray]],
    processable: List[str],
    config: Dict[str, Any],
    reconstruction: types.Reconstruction,
    depth_ranges: Dict[str, Tuple[float, float]],
) -> None:
    """Fuse cleaned depthmaps per cluster, each using N-hop expanded
    neighbors, producing a ``fused_batch_XXXX.ply`` per cluster.
    Points outside the cluster's bounding box are discarded.

    Cleaned depthmaps and color images are loaded from disk on-demand
    for each cluster and released after the cluster is fused.  Clusters
    are processed in parallel to overlap I/O with CPU fusion.
    """
    fusable = [
        sid for sid in processable if data.clean_depthmap_exists(sid)
    ]
    if not fusable:
        logger.warning("No cleaned views available for fusion.")
        return

    fusable_set: Set[str] = set(fusable)

    def _fuse_single_cluster(batch_num: int) -> None:
        """Fuse one cluster: load views from disk, run fuser, save PLY."""
        log.setup()
        cluster_shots = clusters[batch_num]

        ordered = sorted(cluster_shots)
        local_idx: Dict[str, int] = {
            sid: i for i, sid in enumerate(ordered)
        }

        use_svo = config.get("depthmap_fusion_backend", "depthmap") == "svo"

        # ── Batch-preload helper ──────────────────────────────────────
        def _prepare_views(
            view_ids: List[str],
        ) -> Dict[str, Tuple[
            NDArray, NDArray, NDArray, NDArray,
            NDArray, NDArray, NDArray, Optional[NDArray],
        ]]:
            """Batch-load cleaned depthmaps + color images in parallel.

            Returns a dict mapping shot id → (K, R, t, depth, normal,
            color, vmask, confidence) for every successfully loaded view.
            """
            loadable = [
                sid for sid in view_ids
                if data.clean_depthmap_exists(sid)
            ]
            if not loadable:
                return {}

            # I/O-heavy: load all clean depthmaps in parallel.
            loaded = data.load_clean_depthmaps_parallel(loadable)

            # Load validity masks in parallel.
            vmask_map = data.load_undistorted_validity_masks_parallel(
                loadable,
            )

            # Load color images in parallel.
            color_map = data.load_undistorted_images_parallel(loadable)

            result: Dict[str, Tuple[
                NDArray, NDArray, NDArray, NDArray,
                NDArray, NDArray, NDArray, Optional[NDArray],
            ]] = {}
            for sid, (depth, normal, _score, confidence) in zip(
                loadable, loaded,
            ):
                h, w = depth.shape[:2]

                shot = reconstruction.shots[sid]
                K = shot.camera.get_K_in_pixel_coordinates(w, h)
                R = shot.pose.get_rotation_matrix()
                t = shot.pose.translation

                combined = depth.ravel() > 0

                if sid in depth_ranges:
                    _, max_d = depth_ranges[sid]
                    extreme = depth.ravel() > max_d
                    n_ext = int(np.count_nonzero(extreme & combined))
                    if n_ext > 0:
                        logger.warning(
                            "Shot %s: masking %d pixels with depth > %.1f",
                            sid, n_ext, max_d,
                        )
                    combined &= ~extreme

                if sid in vmask_map:
                    vmask_raw = vmask_map[sid]
                    mh, mw = vmask_raw.shape[:2]
                    if (mh, mw) != (h, w):
                        vmask_raw = cv2.resize(
                            vmask_raw, (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    combined &= vmask_raw.ravel() > 0

                reject = ~combined.reshape(h, w)
                n_rejected = int(np.count_nonzero(reject & (depth > 0)))
                if n_rejected > 0:
                    logger.info(
                        "Shot %s: combined mask rejected %d pixels",
                        sid, n_rejected,
                    )
                depth[reject] = 0.0

                vmask = (combined.reshape(h, w).astype(np.uint8)) * 255

                color = color_map[sid]
                color = scale_down_image(color, w, h)

                result[sid] = (
                    K, R, t, depth, normal, color, vmask, confidence,
                )

            return result

        cluster_set = set(cluster_shots) & fusable_set

        if use_svo:
            # ── SVO fusion with sub-volume splitting ─────────────────
            voxel_size = config["depthmap_fusion_svo_voxel_size"]
            trunc_factor_cfg = config["depthmap_fusion_svo_trunc_factor"]
            max_voxels = config["depthmap_fusion_svo_max_voxels"]
            coarse_factor = config["depthmap_fusion_svo_coarse_factor"]
            n_augment = config["depthmap_fusion_svo_augment_neighbors"]

            # Step 1: Augment cluster with neighbor views.
            augmented = _augment_cluster_views(
                cluster_shots, all_neighbors, fusable_set, n_augment,
            )
            n_extra = len(augmented) - len(cluster_shots)
            logger.info(
                f"Cluster {batch_num}: {len(cluster_shots)} shots + "
                f"{n_extra} augmented = {len(augmented)} total"
            )

            # Step 2: Pre-scan on CPU (subsampled depth projection).
            grid = _prescan_coarse_grid(
                data, augmented, reconstruction,
                voxel_size, coarse_factor,
            )
            coarse_size = voxel_size * coarse_factor
            logger.info(
                f"Cluster {batch_num}: pre-scan found {len(grid)} "
                f"coarse cells (cell_size={coarse_size:.3f}m)"
            )

            # Step 3: Split into sub-volumes.
            subvolumes = _split_into_subvolumes(
                grid, augmented, voxel_size, coarse_factor,
                max_voxels, trunc_factor_cfg,
            )
            logger.info(
                f"Cluster {batch_num}: split into "
                f"{len(subvolumes)} sub-volume(s)"
            )

            # Step 4: Fuse each sub-volume independently.
            #         If the GPU counting pass reveals more voxels than
            #         the budget, recursively split along the longest
            #         core axis.
            all_points: List[NDArray] = []
            all_normals: List[NDArray] = []
            all_colors: List[NDArray] = []

            # Batch-load all views upfront (parallel I/O).
            view_cache = _prepare_views(augmented)
            logger.info(
                f"Cluster {batch_num}: loaded {len(view_cache)} views"
            )

            def _fuse_subvolume(sv: _SubVolume) -> None:
                """Fuse a single sub-volume, splitting if over budget."""
                fuser = pydense.SVOFuser()
                fuser.set_voxel_size(voxel_size)
                fuser.set_trunc_factor(trunc_factor_cfg)
                fuser.set_min_weight(
                    config["depthmap_fusion_svo_min_weight"]
                )
                fuser.set_device(0)
                fuser.set_bbox(sv.ext_min, sv.ext_max)

                sv_views = sorted(sv.view_ids)
                n_loaded = 0
                for sid in sv_views:
                    result = view_cache.get(sid)
                    if result is None:
                        continue
                    K, R, t, depth, normal, color, vmask, conf = result
                    fuser.add_view(
                        K, R, t, depth, normal, color, vmask,
                        confidence=conf, name=sid,
                    )
                    n_loaded += 1

                logger.info(
                    f"  Sub-volume ({n_loaded} views), core "
                    f"[{sv.core_min[0]:.1f},{sv.core_min[1]:.1f},"
                    f"{sv.core_min[2]:.1f}]-"
                    f"[{sv.core_max[0]:.1f},{sv.core_max[1]:.1f},"
                    f"{sv.core_max[2]:.1f}]"
                )

                if n_loaded == 0:
                    return

                # Count actual voxels on GPU.
                n_voxels = fuser.count_voxels()
                if n_voxels > max_voxels:
                    logger.info(
                        f"  → {n_voxels} voxels > budget "
                        f"{max_voxels}, splitting …"
                    )
                    del fuser
                    # Binary split along longest core axis.
                    spans = sv.core_max - sv.core_min
                    axis = int(np.argmax(spans))
                    mid = (sv.core_min[axis] + sv.core_max[axis]) / 2.0
                    margin = trunc_factor_cfg * voxel_size

                    # Left half.
                    left_core_max = sv.core_max.copy()
                    left_core_max[axis] = mid
                    left_ext_min = (
                        sv.core_min - margin
                    ).astype(np.float32)
                    left_ext_max = (
                        left_core_max + margin
                    ).astype(np.float32)
                    left_views = {
                        sid for sid in sv.view_ids
                        if sid in fusable_set
                    }
                    left_sv = _SubVolume(
                        core_min=sv.core_min.copy(),
                        core_max=left_core_max.astype(np.float32),
                        ext_min=left_ext_min,
                        ext_max=left_ext_max,
                        view_ids=left_views,
                    )

                    # Right half.
                    right_core_min = sv.core_min.copy()
                    right_core_min[axis] = mid
                    right_ext_min = (
                        right_core_min - margin
                    ).astype(np.float32)
                    right_ext_max = (
                        sv.core_max + margin
                    ).astype(np.float32)
                    right_sv = _SubVolume(
                        core_min=right_core_min.astype(np.float32),
                        core_max=sv.core_max.copy(),
                        ext_min=right_ext_min,
                        ext_max=right_ext_max,
                        view_ids=left_views,  # same views
                    )

                    _fuse_subvolume(left_sv)
                    _fuse_subvolume(right_sv)
                    return

                # Voxels within budget — proceed with integration.
                pts_arr, nrm_arr, clr_arr = fuser.fuse()
                pts = np.asarray(pts_arr, dtype=np.float32)
                nrm = np.asarray(nrm_arr, dtype=np.float32)
                clr = np.asarray(clr_arr, dtype=np.uint8)

                if len(pts) > 0:
                    inside_core = np.all(
                        (pts >= sv.core_min) & (pts <= sv.core_max),
                        axis=1,
                    )
                    pts = pts[inside_core]
                    nrm = nrm[inside_core]
                    clr = clr[inside_core]

                if len(pts) > 0:
                    all_points.append(pts)
                    all_normals.append(nrm)
                    all_colors.append(clr)
                    logger.info(
                        f"    → {len(pts)} points extracted"
                    )

                del fuser

            for sv in subvolumes:
                _fuse_subvolume(sv)
                gc.collect()

            view_cache.clear()

            # Concatenate all sub-volume results.
            if all_points:
                points = np.concatenate(all_points)
                normals = np.concatenate(all_normals)
                colors = np.concatenate(all_colors)
            else:
                points = np.empty((0, 3), dtype=np.float32)
                normals = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.uint8)
            logger.info(
                f"Cluster {batch_num}: {len(points)} total fused points "
                f"from {len(subvolumes)} sub-volume(s)"
            )
            context.log_memory("fused cluster data in memory")

        else:
            # ── Depthmap fusion (unchanged) ──────────────────────────
            fuser = pydense.DepthmapFuser()
            fuser.set_min_num_consistent(
                config["depthmap_fusion_min_consistent"]
            )
            fuser.set_max_reproj_error(
                config["depthmap_fusion_max_reproj_error"]
            )
            fuser.set_max_depth_error(
                config["depthmap_fusion_max_depth_error"]
            )
            fuser.set_max_normal_error(
                config["depthmap_fusion_max_normal_error"]
            )
            fuser.set_border_margin(config["depthmap_fusion_border_margin"])
            fuser.set_num_threads(config["depthmap_fusion_num_threads"])
            fuser.set_sor_params(
                config["depthmap_fusion_sor_knn"],
                config["depthmap_fusion_sor_stddev_factor"],
            )
            fuser.set_behind_depth_factor(
                config["depthmap_fusion_behind_depth_factor"]
            )

            dm_views = _prepare_views(ordered)
            for sid in ordered:
                result = dm_views.get(sid)
                if result is None:
                    continue
                K, R, t, depth, normal, color, vmask, confidence = result
                nbrs = cluster_set
                nbr_local = np.array(
                    sorted(
                        local_idx[n] for n in nbrs if n in local_idx
                    ),
                    dtype=np.int32,
                )
                is_primary = sid in cluster_set
                fuser.add_view(
                    K, R, t,
                    depth, normal, color,
                    vmask,
                    nbr_local,
                    primary=is_primary,
                )

            logger.info(
                f"Fusing cluster {batch_num}: {len(cluster_set)} "
                f"primary views, {len(ordered)} total views …"
            )
            points_arr, normals_arr, colors_arr = fuser.fuse()
            context.log_memory("fused cluster data in memory")
            points = np.asarray(points_arr, dtype=np.float32)
            normals = np.asarray(normals_arr, dtype=np.float32)
            colors = np.asarray(colors_arr, dtype=np.uint8)
            del fuser
            gc.collect()

        # Clip to cluster bbox with Voronoi deduplication.
        if len(points) > 0:
            bb_min, bb_max = cluster_bboxes[batch_num]
            inside = np.all(
                (points >= bb_min) & (points <= bb_max), axis=1
            )

            def _cam_centroid(shot_ids: List[str]) -> NDArray:
                origins = [
                    reconstruction.shots[s].pose.get_origin()
                    for s in shot_ids
                    if s in reconstruction.shots
                ]
                if origins:
                    return np.mean(origins, axis=0).astype(np.float64)
                return ((bb_min + bb_max) / 2.0).astype(np.float64)

            my_center = _cam_centroid(cluster_shots)
            my_dist_sq = np.sum(
                (points.astype(np.float64) - my_center) ** 2, axis=1
            )
            for other_idx in range(len(cluster_bboxes)):
                if other_idx == batch_num:
                    continue
                obb_min, obb_max = cluster_bboxes[other_idx]
                in_other = np.all(
                    (points >= obb_min) & (points <= obb_max), axis=1
                )
                contested = in_other & inside
                if not np.any(contested):
                    continue
                o_center = _cam_centroid(clusters[other_idx])
                o_dist_sq = np.sum(
                    (points.astype(np.float64) - o_center) ** 2, axis=1
                )
                give_away = contested & (
                    (o_dist_sq < my_dist_sq)
                    | (
                        (o_dist_sq == my_dist_sq)
                        & (other_idx < batch_num)
                    )
                )
                inside &= ~give_away

            n_before = len(points)
            points = points[inside]
            normals = normals[inside]
            colors = colors[inside]
        logger.info(
            "Batch %d: %d fused points (%d clipped by bbox)",
            batch_num, len(points), n_before - len(points),
        )

        if len(points) > 0:
            labels = np.zeros(len(points), dtype=np.uint8)
            filename = f"fused_batch_{batch_num:04d}.ply"
            data.save_point_cloud(
                points, normals, colors, labels, filename=filename
            )

        if config.get("depthmap_save_debug_ply", False) and len(points) > 0:
            cr, cg, cb = _CLUSTER_COLORS[
                batch_num % len(_CLUSTER_COLORS)
            ]

            bbox_pts = _sample_bbox_wireframe(
                cluster_bboxes[batch_num], step=0.01
            )
            n_bbox = len(bbox_pts)
            bbox_normals = np.zeros((n_bbox, 3), dtype=np.float32)
            bbox_colors = np.full(
                (n_bbox, 3), (cr, cg, cb), dtype=np.uint8
            )

            frustum_parts: List[NDArray] = []
            for sid in cluster_shots:
                if sid in reconstruction.shots:
                    fpts = _sample_frustum_wireframe(
                        reconstruction.shots[sid],
                        depth=0.3, step=0.01,
                    )
                    frustum_parts.append(fpts)
            if frustum_parts:
                frustum_pts = np.concatenate(frustum_parts)
            else:
                frustum_pts = np.empty((0, 3), dtype=np.float32)
            n_frust = len(frustum_pts)
            frustum_normals = np.zeros(
                (n_frust, 3), dtype=np.float32
            )
            frustum_colors = np.full(
                (n_frust, 3), (cr, cg, cb), dtype=np.uint8,
            )

            dbg_points = np.concatenate([points, bbox_pts, frustum_pts])
            dbg_normals = np.concatenate(
                [normals, bbox_normals, frustum_normals],
            )
            dbg_colors = np.concatenate(
                [colors, bbox_colors, frustum_colors],
            )
            dbg_labels = np.zeros(len(dbg_points), dtype=np.uint8)
            data.save_point_cloud(
                dbg_points, dbg_normals, dbg_colors, dbg_labels,
                filename=f"fused_batch_{batch_num:04d}_debug.ply",
            )

        gc.collect()

    # Run cluster fusions in parallel.
    n_fusion_workers = 1  # min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(
        max_workers=n_fusion_workers, thread_name_prefix="fuse"
    ) as pool:
        futures: Dict[Future, int] = {}
        for batch_num in range(len(clusters)):
            fut = pool.submit(_fuse_single_cluster, batch_num)
            futures[fut] = batch_num

        for fut in as_completed(futures):
            batch_num = futures[fut]
            try:
                fut.result()
                logger.info(f"Fusion cluster {batch_num} complete")
            except Exception as exc:
                logger.error(f"Fusion cluster {batch_num} failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
#  Phase 4: Merge batch PLYs
# ═══════════════════════════════════════════════════════════════════════


def _merge_fusion_batches(
    data: UndistortedDataSet,
    batch_nums: List[int],
) -> None:
    """Concatenate all batch PLYs into the final ``fused.ply``.

    Batch PLYs are loaded in parallel to overlap I/O.  After
    concatenation the per-batch arrays are released immediately.
    """
    if not batch_nums:
        logger.warning("No fusion batches to merge.")
        return

    def _load_batch(
        batch_num: int,
    ) -> Tuple[int, NDArray, NDArray, NDArray, NDArray] | None:
        filename = f"fused_batch_{batch_num:04d}.ply"
        path = data.point_cloud_file(filename)
        if not data.io_handler.isfile(path):
            return None
        p, n, c, lbl = data.load_point_cloud(filename)
        if len(p) == 0:
            return None
        return (batch_num, p, n, c, lbl)

    # Load batch PLYs in parallel.
    with ThreadPoolExecutor(
        max_workers=min(4, len(batch_nums)), thread_name_prefix="merge"
    ) as pool:
        results = list(pool.map(_load_batch, sorted(batch_nums)))

    all_points: List[NDArray] = []
    all_normals: List[NDArray] = []
    all_colors: List[NDArray] = []
    all_labels: List[NDArray] = []

    for r in results:
        if r is not None:
            _, p, n, c, lbl = r
            all_points.append(p)
            all_normals.append(n)
            all_colors.append(c)
            all_labels.append(lbl)
    del results

    if not all_points:
        logger.warning("No points found in any batch PLY.")
        return

    points = np.concatenate(all_points)
    del all_points
    normals_arr = np.concatenate(all_normals)
    del all_normals
    colors = np.concatenate(all_colors)
    del all_colors
    labels = np.concatenate(all_labels)
    del all_labels
    gc.collect()

    data.save_point_cloud(
        points, normals_arr, colors, labels, filename="fused.ply"
    )
    logger.info(
        f"Merged {len(batch_nums)} batches → fused.ply ({len(points)} points)"
    )
    del points, normals_arr, colors, labels
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  Neighbours & tracks-graph utilities
# ═══════════════════════════════════════════════════════════════════════


def find_best_all_neighboring_images(
    shot: pymap.Shot,
    common_tracks: Dict[str, Dict[str, List[str]]],
    reconstruction: types.Reconstruction,
    num_neighbors: int,
    min_point_best: int = 20,
    min_point_all: int = 40,
) -> Tuple[List[pymap.Shot], List[pymap.Shot]]:
    """Find neighboring images based on common tracks.

    Returns ``(best_neighbors, all_neighbors)`` where *best_neighbors*
    starts with *shot* itself followed by the top ``num_neighbors``
    views ranked by baseline-angle score.
    """
    # Cosine bounds (cos is monotonically decreasing on [0, pi]):
    #   theta in (theta_min, theta_max)  ↔  cos in (cos_hi_bound, cos_lo_bound)
    theta_min = np.pi / 60
    theta_max = np.pi / 3

    cos_lo = np.cos(theta_max)   # lower cosine bound
    cos_hi = np.cos(theta_min)   # upper cosine bound

    ns_best: List[Tuple[pymap.Shot, int]] = []
    ns_all: List[Tuple[pymap.Shot, int]] = []
    C1 = np.asarray(shot.pose.get_origin(), dtype=np.float64)
    rec_points = reconstruction.points

    for other_id, tracks in common_tracks.get(shot.id, {}).items():
        if other_id not in reconstruction.shots:
            continue
        other = reconstruction.shots[other_id]

        # Gather 3-D coordinates of common tracks present in the
        # reconstruction — single Python loop of dict lookups, no
        # per-element math.
        coords = [rec_points[tid].coordinates
                  for tid in tracks if tid in rec_points]
        score_all = len(coords)
        if score_all == 0:
            continue

        # Record the all-neighbors candidate early so we can skip
        # the expensive angle computation when there aren't enough
        # tracks to qualify for best-neighbors.
        if score_all > min_point_all:
            ns_all.append((other, 1, score_all))
        if score_all <= min_point_best:
            continue

        # Vectorised angle computation (replaces the per-track loop
        # that called angle_between_points with scalar np.arccos).
        P = np.array(coords, dtype=np.float64)      # (N, 3)
        C2 = np.asarray(other.pose.get_origin(), dtype=np.float64)
        a = C1 - P                                   # vectors point → C1
        b = C2 - P                                   # vectors point → C2
        dot = np.einsum("ij,ij->i", a, b)
        la = np.einsum("ij,ij->i", a, a)
        lb = np.einsum("ij,ij->i", b, b)
        denom = np.sqrt(la * lb)
        valid = denom > 0
        cos_theta = np.divide(dot, denom, where=valid, out=np.ones_like(dot))
        np.clip(cos_theta, -1.0, 1.0, out=cos_theta)
        score_best = int(np.count_nonzero(
            (cos_theta > cos_lo) & (cos_theta < cos_hi)
        ))
        avg_theta = np.arccos(np.clip(np.mean(cos_theta), -1.0, 1.0))
        if avg_theta < theta_min or avg_theta > theta_max:
            continue

        if score_best > min_point_best:
            ns_best.append((other, score_best, avg_theta))

    ns_best.sort(key=lambda ns: ns[1], reverse=True)

    best_neighbors = [shot] + [n for n, _, _ in ns_best[:num_neighbors]]
    all_neighbors = best_neighbors + \
        [n for n, _, _ in ns_all if n not in best_neighbors]

    return best_neighbors, all_neighbors


def compute_depth_range(
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    shot: pymap.Shot,
    config: Dict[str, Any],
    down: float = 0.1,
    up: float = 0.9,
    extent: bool = True,
) -> Tuple[float, float]:
    """Compute min and max depth based on reconstruction points."""
    depths = []
    for track in tracks_manager.get_shot_observations(shot.id):
        if track in reconstruction.points:
            p = reconstruction.points[track].coordinates
            z = shot.pose.transform(p)[2]
            depths.append(z)
    min_depth = np.percentile(depths, down * 100) * \
        ((1 - down) if extent else 1)
    max_depth = np.percentile(depths, up * 100) * \
        ((1 + (1 - up)) if extent else 1)

    config_min_depth: float = config["depthmap_min_depth"]
    config_max_depth: float = config["depthmap_max_depth"]

    return config_min_depth or min_depth, config_max_depth or max_depth


# ═══════════════════════════════════════════════════════════════════════
#  Image / geometry utilities  (kept for public API)
# ═══════════════════════════════════════════════════════════════════════


def load_combined_mask(data: UndistortedDataSet, shot: pymap.Shot) -> NDArray:
    """Load the undistorted mask.

    If no mask exists return an array of ones.
    """
    mask = data.load_undistorted_combined_mask(shot.id)
    if mask is None:
        size = int(shot.camera.height), int(shot.camera.width)
        return np.ones(size, dtype=np.uint8)
    else:
        return mask


def scale_image(
    image: NDArray, width: int, height: int, interpolation: int
) -> NDArray:
    return cv2.resize(image, (width, height), interpolation=interpolation)


def scale_down_image(
    image: NDArray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_AREA,
) -> NDArray:
    width = min(width, image.shape[1])
    height = min(height, image.shape[0])
    return scale_image(image, width, height, interpolation)


def _save_depthmap_as_ply(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    shot_id: str,
    depth: NDArray,
    prefix: str,
) -> None:
    """Back-project a depthmap to 3-D and save as a binary PLY."""
    shot = reconstruction.shots[shot_id]
    h, w = depth.shape[:2]
    K = shot.camera.get_K_in_pixel_coordinates(w, h)
    R = shot.pose.get_rotation_matrix()
    t = shot.pose.translation

    y, x = np.mgrid[:h, :w]
    v = np.vstack((x.ravel(), y.ravel(), np.ones(w * h)))
    cam_coords = depth.reshape((1, -1)) * np.linalg.inv(K).dot(v)
    pts_world = (R.T @ (cam_coords - t[:, np.newaxis])).T.astype(np.float32)

    valid = depth.ravel() > 0
    points = pts_world[valid]

    color = data.load_undistorted_image(shot_id)
    color = scale_down_image(color, w, h)
    colors = color.reshape(-1, 3)[valid].astype(np.uint8)

    normals = np.zeros_like(points)
    labels = np.zeros(len(points), dtype=np.uint8)
    data.save_point_cloud(
        points, normals, colors, labels,
        filename=f"{prefix}_{shot_id}.ply",
    )


def depthmap_to_ply(shot: pymap.Shot, depth: NDArray, image: NDArray) -> str:
    """Export depthmap points as a PLY string."""
    height, width = depth.shape
    K = shot.camera.get_K_in_pixel_coordinates(width, height)
    R = shot.pose.get_rotation_matrix()
    t = shot.pose.translation
    y, x = np.mgrid[:height, :width]
    v = np.vstack((x.ravel(), y.ravel(), np.ones(width * height)))
    camera_coords = depth.reshape((1, -1)) * np.linalg.inv(K).dot(v)
    points = R.T.dot(camera_coords - t.reshape(3, 1))

    vertices = []
    for p, c, d in zip(points.T, image.reshape(-1, 3), depth.reshape(-1, 1)):
        if d != 0:
            s = "{} {} {} {} {} {}".format(p[0], p[1], p[2], c[0], c[1], c[2])
            vertices.append(s)

    return io.points_to_ply_string(vertices)


# ═══════════════════════════════════════════════════════════════════════
#  Octree tile export (viewer streaming)
# ═══════════════════════════════════════════════════════════════════════


def _export_octree_tiles(
    data: UndistortedDataSet,
    config: Dict[str, Any],
) -> None:
    """Convert the dense point cloud PLY to octree tiles for the viewer.

    Looks for fused.ply or merged.ply (in that order) and builds an octree
    tile set under ``point_cloud/`` next to the reconstruction.

    The octree uses Morton ordering and LOD subsampling so that the viewer
    can progressively stream tiles based on the camera frustum.
    """

    # Find the source PLY.
    for ply_name in ("fused.ply", "merged.ply"):
        ply_path = data.point_cloud_file(ply_name)
        if data.io_handler.isfile(ply_path):
            break
    else:
        logger.warning("No dense point cloud found for octree export.")
        return

    logger.info(f"Building octree tiles from {ply_name} …")

    # Load PLY data.
    with data.io_handler.open_rb(ply_path) as fp:
        points, normals, colors, labels = io.point_cloud_from_ply(fp)

    if len(points) == 0:
        logger.warning("Point cloud is empty, skipping octree export.")
        return

    # Output directory: next to the undistorted data, under point_cloud/.
    output_dir = os.path.join(data.data_path, "point_cloud")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the builder.
    builder_config = pypointcloud.OctreeBuilderConfig()
    builder_config.output_dir = output_dir
    builder_config.max_points_per_tile = config.get(
        "octree_max_points_per_tile", 50000
    )
    builder_config.max_depth = config.get("octree_max_depth", 15)
    builder_config.lod_sample_count = config.get(
        "octree_lod_sample_count", 10000
    )

    # Ensure float32 contiguous arrays.
    points = np.ascontiguousarray(points, dtype=np.float32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.uint8)

    # No per-point radii yet — the builder will use the tile spacing.
    radii = np.array([], dtype=np.float32)

    meta = pypointcloud.build_octree(
        positions=points,
        normals=normals,
        colors=colors,
        radii=radii,
        config=builder_config,
    )

    logger.info(
        f"Octree export complete: {meta.total_points} points, depth {meta.max_depth}, {output_dir}"
    )
