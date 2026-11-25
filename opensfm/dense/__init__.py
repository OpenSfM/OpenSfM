# pyre-strict
"""Cluster-based dense reconstruction pipeline — high-level orchestrator.

The pipeline is split into four stages that hand off via disk, so each can run
in its own process (bounded memory, clean teardown) or even on a different
machine sharing the dataset directory:

  1. ``run_clustering`` — clusters + super-points + neighbours + depth ranges.
  2. ``run_depthmaps``  — raw (PatchMatch) then cleaned depthmaps (GPU).
  3. ``run_fusion``     — per-cluster SVO fusion → ``fused_batch_*.ply`` (+ DSM/ortho tiles).
  4. ``run_merge``      — merge batch PLYs + DSM/ortho tiles, export LAS/LAZ + octree.

Stages 2–4 require stage 1's artefacts on disk and fail fast otherwise.  The
heavy implementations live in the sibling submodules; this module only wires
inputs/outputs and sequences the stages.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

from numpy.typing import NDArray

from opensfm import context, pymap, pysfm, types
from opensfm.dataset import UndistortedDataSet

from . import clustering, crop, depthmaps, equalize, fusion, merge
from .common import compute_depth_range, depthmap_to_ply, scale_down_image

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "run_clustering",
    "run_depthmaps",
    "run_equalize",
    "run_fusion",
    "run_merge",
    "depthmap_to_ply",
    "scale_down_image",
]


def run_equalize(
    data: UndistortedDataSet,
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> None:
    """Estimate per-image radiometric corrections (gain + vignetting) from the
    SfM track correspondences and persist them for the colour bake."""
    context.log_memory("dense equalize start")
    corrections = equalize.estimate_image_corrections(
        tracks_manager, reconstruction, data.config
    )
    data.save_equalization(corrections)
    logger.info(
        "Equalization saved for %d image(s) → %s",
        len(corrections), data.equalization_file(),
    )
    context.log_memory("dense equalize done")


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════


def _processable(
    reconstruction: types.Reconstruction
) -> List[str]:
    """Sorted shot ids present in both the reconstruction"""
    return sorted(
        set(reconstruction.shots.keys())
    )


def _require_clustering(
    data: UndistortedDataSet,
    *,
    need_neighbors: bool,
    need_bboxes: bool = False,
) -> None:
    """Fail fast if the dense-clustering stage has not produced its outputs."""
    missing: List[str] = []
    if not (
        os.path.exists(data.clusters_file())
        and os.path.isfile(data.clusters_points_file())
    ):
        missing.append("clusters")
    if need_neighbors and not data.neighbors_exist():
        missing.append("neighbors")
    if need_neighbors and not data.depth_ranges_exist():
        missing.append("depth ranges")
    if need_bboxes and not data.cluster_bboxes_exist():
        missing.append("cluster bounding boxes")
    if missing:
        raise RuntimeError(
            f"Missing dense-clustering outputs ({', '.join(missing)}). "
            "Run the 'dense_clustering' command first."
        )


def _load_neighbors(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    processable: List[str],
) -> Tuple[Dict[str, List[pymap.Shot]], Dict[str, List[pymap.Shot]]]:
    """Rebuild the per-shot neighbour ``pymap.Shot`` lists from the saved id maps."""
    best_map = data.load_neighbors_best()
    all_map = data.load_neighbors_all()
    shots_view = reconstruction.shots
    best_neighbors: Dict[str, List[pymap.Shot]] = {}
    all_neighbors: Dict[str, List[pymap.Shot]] = {}
    for sid in processable:
        best_neighbors[sid] = [shots_view[nid]
                               for nid in best_map.get(sid, [])]
        all_neighbors[sid] = [shots_view[nid] for nid in all_map.get(sid, [])]
    return best_neighbors, all_neighbors


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1 — clustering, neighbours, depth ranges
# ═══════════════════════════════════════════════════════════════════════


def run_clustering(
    data: UndistortedDataSet,
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> None:
    """Build (or load) clusters and persist clusters, super-points, neighbour
    selection and per-shot depth ranges for the downstream stages."""
    context.log_memory("dense clustering start")
    config = data.config
    num_neighbors: int = config["depthmap_num_neighbors"]

    processable = sorted(
        set(_processable(reconstruction)) & set(graph.get_shot_ids())
    )
    if not processable:
        logger.warning("No processable shots found for depthmap computation")
        return

    # ── Clusters (+ super-points) — load if present, else compute & save ──
    if os.path.exists(data.clusters_file()) and os.path.isfile(
        data.clusters_points_file()
    ):
        clusters = data.load_clusters()
        super_points = data.load_clusters_points()
        logger.info(
            f"Loaded {len(clusters)} clusters from {data.clusters_file()}")
    else:
        clusters, sp_coords, sp_vis, _, covis, leiden_raw = clustering.cluster_views(
            processable,
            graph,
            reconstruction,
            config["depthmap_cluster_max_size"],
            fuse_knn=15,
            fuse_radius_factor=0.5,
            edge_max_factor=config["depthmap_cluster_edge_max_factor"],
        )
        clustering._save_cluster_debug_ply(
            data, clusters, reconstruction, sp_coords, sp_vis)
        super_points = covis.super_points
        data.save_clusters(clusters)
        data.save_clusters_points(super_points)
        logger.info(f"Cluster assignments saved to {data.clusters_file()}")

    # ── Per-cluster bounding boxes ──
    #   Persisted here (the only stage with the tracks graph) so the fusion
    #   stage can clip to them without loading the tracks manager.
    if data.cluster_bboxes_exist():
        logger.info("Cluster bounding boxes already computed; skipping")
    else:
        cluster_bboxes = [
            clustering._compute_cluster_bbox(cl, graph, reconstruction)
            for cl in clusters
        ]
        data.save_cluster_bboxes(cluster_bboxes)
        context.log_memory("compute cluster bounding boxes done")

    # ── Global neighbor selection (C++, multithreaded) ──
    if data.neighbors_exist():
        logger.info("Neighbors already computed; skipping selection")
    else:
        theta_min: float = config.get("depthmap_neighbor_min_angle", 3.0)
        theta_max: float = config.get("depthmap_neighbor_max_angle", 60.0)
        nbr_result: pysfm.NeighborResult = pysfm.select_neighbors(
            graph,
            reconstruction.map,
            super_points,
            processable,
            num_neighbors,
            theta_min_deg=theta_min,
            theta_max_deg=theta_max,
        )
        context.log_memory("C++ neighbor selection done")
        best_nbr_map = nbr_result.best_neighbors
        all_nbr_map = nbr_result.all_neighbors
        del nbr_result
        data.save_neighbors_best(
            {sid: list(best_nbr_map.get(sid, [])) for sid in processable})
        data.save_neighbors_all(
            {sid: list(all_nbr_map.get(sid, [])) for sid in processable})

    # ── Per-shot depth ranges ──
    if data.depth_ranges_exist():
        logger.info("Depth ranges already computed; skipping")
    else:
        depth_ranges: Dict[str, Tuple[float, float]] = {}
        for shot_id in processable:
            shot = reconstruction.shots[shot_id]
            mind, maxd = compute_depth_range(
                graph, reconstruction, shot, config)
            depth_ranges[shot_id] = (mind, maxd)
        data.save_depth_ranges(depth_ranges)

    # ── Dense crop hull (only when DSM/ortho is enabled) ──
    if config["dsm_enabled"] and config["dense_crop_to_sfm_hull"]:
        if data.crop_hull_exists():
            logger.info("Dense crop hull already computed; skipping")
        else:
            hull = crop.compute_sfm_crop_hull(
                reconstruction, config["dense_crop_percentile"])
            if hull is not None:
                data.save_crop_hull(hull, config["dense_crop_percentile"])
                logger.info(
                    "Dense crop hull saved (%d vertices) → %s",
                    len(hull), data.crop_hull_file())

    logger.info("Clusters, neighbors and depth ranges ready")
    context.log_memory("dense clustering done")


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2 — raw + clean depthmaps
# ═══════════════════════════════════════════════════════════════════════


def run_depthmaps(
    data: UndistortedDataSet,
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> None:
    """Compute raw (PatchMatch) then cleaned depthmaps on GPU, using the
    clusters/neighbours/depth-ranges produced by ``run_clustering``."""
    context.log_memory("compute_depthmaps start")
    config = data.config

    processable = _processable(reconstruction)
    if not processable:
        logger.warning("No processable shots found for depthmap computation")
        return
    _require_clustering(data, need_neighbors=True)

    clusters = data.load_clusters()
    best_neighbors, all_neighbors = _load_neighbors(
        data, reconstruction, processable)
    depth_ranges = data.load_depth_ranges()

    device_order = depthmaps.discover_gpu_devices(config)
    if not device_order:
        return

    # ── Phase 1: per-cluster raw depthmaps ──
    depthmaps.run_raw_depthmaps(
        data, graph, reconstruction, best_neighbors,
        clusters, depth_ranges, processable, device_order,
    )
    context.log_memory("phase 1 raw depthmaps done")

    # ── Phase 2: GPU cleaning with N-hop neighbors ──
    depthmaps.clean_depthmaps(
        data, reconstruction, clusters, all_neighbors, processable, config,
        device_order=device_order,
    )
    context.log_memory("phase 2 cleaning done")

    # Final safety-net sweep: clean_depthmaps already reclaims raw maps on the
    # fly (reference-counted, as soon as each one's last consumer is cleaned), so
    # this normally finds only stragglers — raws of clusters that were skipped or
    # failed to clean.  Delete a raw map only once its clean counterpart exists,
    # so a cluster that failed to clean keeps its raw for a resumed re-run.
    if config.get("depthmap_delete_raw_after_clean", True):
        removed = 0
        for sid in processable:
            if data.raw_depthmap_exists(sid) and data.clean_depthmap_exists(sid):
                data.io_handler.rm_if_exist(data.depthmap_file(sid, "raw.npz"))
                removed += 1
        if removed:
            logger.info(f"Removed {removed} raw depthmap(s) after cleaning")
        context.log_memory("raw depthmaps reclaimed")


# ═══════════════════════════════════════════════════════════════════════
#  Stage 3 — per-cluster fusion (+ DSM/ortho tiles)
# ═══════════════════════════════════════════════════════════════════════


def run_fusion(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
) -> None:
    """Fuse the cleaned depthmaps into ``fused_batch_*.ply`` (and, when enabled,
    per-chunk DSM/ortho tiles).

    Space is partitioned by a global KD-tree split of the occupied grid, so no
    cluster territories are needed here.  Needs no tracks graph; ``best_neighbors``
    (used by the optional photometric refine) is loaded from disk."""
    config = data.config

    processable = _processable(reconstruction)
    if not processable:
        logger.warning("No processable shots found for depthmap computation")
        return
    _require_clustering(data, need_neighbors=True)

    best_neighbors, _all_neighbors = _load_neighbors(
        data, reconstruction, processable)
    depth_ranges = data.load_depth_ranges()

    fusion.fuse_chunks(
        data, processable, config, reconstruction, depth_ranges,
        best_neighbors,
    )
    context.log_memory("phase 3 fusion done")


# ═══════════════════════════════════════════════════════════════════════
#  Stage 4 — merge + export
# ═══════════════════════════════════════════════════════════════════════


def run_merge(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    output_crs: Optional[str] = None,
) -> None:
    """Merge per-cluster fused PLYs and DSM/ortho tiles into the final products,
    then export LAS/LAZ and octree tiles.  Needs no tracks graph.

    When ``output_crs`` is given, the LAS/LAZ and DSM/ortho products are
    reprojected to that CRS (georeferenced); otherwise they stay in the
    topocentric frame.  ``fused.ply`` and the octree always stay topocentric.
    """
    config = data.config

    # Fusion emits one batch per KD-tree chunk (no longer the graph-cluster
    # count), and some chunks may yield no points/tile — so the merge discovers
    # the actual batch indices by listing the depthmap folder per file type.

    # ── Optional crop to the SfM footprint (only with DSM enabled) ──
    crop_hull: Optional[NDArray] = None
    if (
        config["dsm_enabled"]
        and config["dense_crop_to_sfm_hull"]
        and data.crop_hull_exists()
    ):
        crop_hull = data.load_crop_hull()
        logger.info(
            "Cropping dense outputs to the SfM hull (%d vertices)",
            len(crop_hull))

    # ── Merge batch PLYs into final fused.ply ──
    merge.merge_fusion_batches(
        data, data.list_batch_indices("fused_batch_", ".ply"),
        crop_hull=crop_hull,
    )

    # ── Merge per-chunk Surface Nets meshes into final mesh.ply ──
    if config["depthmap_fusion_mesh_enabled"]:
        merge.merge_mesh_batches(
            data, data.list_batch_indices("mesh_batch_", ".ply")
        )

    # Composite the per-chunk DSM/ortho tiles into the final raster.  Without
    # this each chunk's save would overwrite dsm.tif / ortho.tif and only the
    # last chunk processed would appear.
    if config["dsm_enabled"]:
        merge.merge_dsm_ortho_batches(
            data, data.list_batch_indices("dsm_ortho_batch_", ".npz"),
            reconstruction.reference, output_crs=output_crs,
            crop_hull=crop_hull,
        )

    context.log_memory("phase 4 merge done")

    # Optionally also export the merged cloud as LAS / LAZ (archival/interchange).
    merge.export_pointcloud_formats(
        data, config, reference=reconstruction.reference, output_crs=output_crs
    )

    # Build octree tiles for the viewer.
    merge.export_octree_tiles(data, config)

    context.log_memory("compute_depthmaps end")
