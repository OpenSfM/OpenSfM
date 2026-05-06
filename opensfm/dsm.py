# pyre-strict

import logging
import os

import numpy as np
from numpy.typing import NDArray

from opensfm import pydense, types
from opensfm.dataset import UndistortedDataSet

logger: logging.Logger = logging.getLogger(__name__)


def compute_dsm(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
) -> None:
    """Generate a Digital Surface Model from cleaned depthmaps.

    Two-pass GPU rasterization:
      Pass 1 — weighted mean of back-projected depthmap pixels (weight =
               confidence × max(0, normal_z_world)).
      Pass 2 — re-scatter rejecting outliers that deviate from the pass-1
               mean by more than ``dsm_outlier_threshold`` meters.

    Followed by optional bilateral smoothing and hole filling.
    """
    config = data.config

    # --- Collect shots with a clean depthmap ---
    shots_with_dm = []
    for shot in reconstruction.shots.values():
        path = data.depthmap_file(shot.id, "clean.npz")
        if os.path.isfile(path):
            shots_with_dm.append(shot)
    if not shots_with_dm:
        logger.warning("No clean depthmaps found — skipping DSM generation")
        return
    logger.info("Found %d shots with clean depthmaps", len(shots_with_dm))

    # --- Resolve GSD ---
    gsd: float = config.get("dsm_gsd", 0.0)
    if gsd <= 0.0:
        gsd = config.get("depthmap_fusion_svo_voxel_size", 0.05)
        logger.info("Auto GSD from voxel size: %.4f m/px", gsd)
    else:
        logger.info("Using configured GSD: %.4f m/px", gsd)

    # --- Compute XY bounding box from SfM points ---
    if not reconstruction.points:
        logger.warning("No SfM points — cannot compute DSM extent")
        return
    coords = np.array(
        [p.coordinates for p in reconstruction.points.values()]
    )
    min_xy = coords[:, :2].min(axis=0).astype(np.float32)
    max_xy = coords[:, :2].max(axis=0).astype(np.float32)
    extent = max_xy - min_xy
    margin = np.maximum(extent * 0.05, gsd * 2).astype(np.float32)
    min_xy -= margin
    max_xy += margin

    width = int(np.ceil((max_xy[0] - min_xy[0]) / gsd))
    height = int(np.ceil((max_xy[1] - min_xy[1]) / gsd))
    logger.info(
        "DSM grid: %d x %d (%.1f x %.1f m)",
        width,
        height,
        max_xy[0] - min_xy[0],
        max_xy[1] - min_xy[1],
    )

    # --- Configure rasterizer ---
    rasterizer = pydense.DSMRasterizer()
    rasterizer.set_gsd(gsd)
    rasterizer.set_bbox(min_xy, max_xy)
    rasterizer.set_device(0)
    rasterizer.set_outlier_threshold(
        config.get("dsm_outlier_threshold", 1.0)
    )
    rasterizer.set_min_count(
        config.get("dsm_min_count", 3)
    )
    rasterizer.set_z_bias(
        config.get("dsm_z_bias", 2.5)
    )
    rasterizer.set_bilateral(
        config.get("dsm_bilateral_enabled", True),
        config.get("dsm_bilateral_spatial", 2),
        config.get("dsm_bilateral_range", 0.3),
    )
    rasterizer.begin()

    # --- Pass 1: weighted mean ---
    logger.info("Pass 1: scattering %d views", len(shots_with_dm))
    for shot in shots_with_dm:
        depth, plane, _score, confidence = data.load_clean_depthmap(shot.id)
        if confidence is None:
            confidence = np.ones_like(depth)
        h, w = depth.shape
        K = np.asarray(shot.camera.get_K_in_pixel_coordinates(w, h))
        R = np.asarray(shot.pose.get_rotation_matrix())
        t = np.asarray(shot.pose.translation)
        rasterizer.scatter(
            K, R, t,
            depth.astype(np.float32, copy=False),
            plane.astype(np.float32, copy=False),
            confidence.astype(np.float32, copy=False),
        )

    rasterizer.begin_pass2()

    percentile: float = config.get("dsm_percentile", 0.75)
    median_radius: int = config.get("dsm_median_radius", 1)

    # --- Coarse P90 (2× GSD) for structure reference ---
    coarse_gsd = gsd * 2.0
    coarse_rast = pydense.DSMRasterizer()
    coarse_rast.set_gsd(coarse_gsd)
    coarse_rast.set_bbox(min_xy, max_xy)
    coarse_rast.set_device(0)
    coarse_rast.set_outlier_threshold(
        config.get("dsm_outlier_threshold", 1.0)
    )
    coarse_rast.set_min_count(1)
    coarse_rast.set_bilateral(False, 0, 0.0)
    coarse_rast.begin()

    logger.info("Coarse pass 1: scattering %d views at %.4f m/px",
                len(shots_with_dm), coarse_gsd)
    for shot in shots_with_dm:
        depth, plane, _score, confidence = data.load_clean_depthmap(shot.id)
        if confidence is None:
            confidence = np.ones_like(depth)
        h, w = depth.shape
        K = np.asarray(shot.camera.get_K_in_pixel_coordinates(w, h))
        R = np.asarray(shot.pose.get_rotation_matrix())
        t = np.asarray(shot.pose.translation)
        coarse_rast.scatter(
            K, R, t,
            depth.astype(np.float32, copy=False),
            plane.astype(np.float32, copy=False),
            confidence.astype(np.float32, copy=False),
        )

    coarse_rast.begin_pass2()

    logger.info("Coarse pass 2: CPU P%.0f scatter", percentile * 100)
    for shot in shots_with_dm:
        depth, plane, _score, confidence = data.load_clean_depthmap(shot.id)
        if confidence is None:
            confidence = np.ones_like(depth)
        h, w = depth.shape
        K = np.asarray(shot.camera.get_K_in_pixel_coordinates(w, h))
        R = np.asarray(shot.pose.get_rotation_matrix())
        t = np.asarray(shot.pose.translation)
        coarse_rast.scatter_cpu(
            K, R, t,
            depth.astype(np.float32, copy=False),
            plane.astype(np.float32, copy=False),
            confidence.astype(np.float32, copy=False),
        )

    coarse_grid: NDArray = coarse_rast.finish_percentile(percentile)
    logger.info(
        "Coarse grid: %d valid / %d total",
        int(np.count_nonzero(~np.isnan(coarse_grid))),
        coarse_grid.size,
    )

    # --- Upsample coarse P90 to fine resolution as reference ---
    from scipy.ndimage import zoom

    coarse_h, coarse_w = coarse_grid.shape
    # Compute exact zoom factors to match fine grid dimensions.
    zoom_y = height / coarse_h
    zoom_x = width / coarse_w
    ref_z = zoom(coarse_grid, (zoom_y, zoom_x), order=1).astype(np.float32)
    # Trim/pad to exact fine grid size (rounding can cause ±1 mismatch).
    ref_z = ref_z[:height, :width]
    if ref_z.shape[0] < height or ref_z.shape[1] < width:
        padded = np.full((height, width), np.nan, dtype=np.float32)
        padded[: ref_z.shape[0], : ref_z.shape[1]] = ref_z
        ref_z = padded

    rasterizer.set_reference_z(ref_z)

    # --- Fine pass 2: CPU P90 with coarse-informed rejection ---
    logger.info(
        "Fine pass 2: CPU P%.0f scatter (%d views, coarse-guided)",
        percentile * 100,
        len(shots_with_dm),
    )
    for shot in shots_with_dm:
        depth, plane, _score, confidence = data.load_clean_depthmap(shot.id)
        if confidence is None:
            confidence = np.ones_like(depth)
        h, w = depth.shape
        K = np.asarray(shot.camera.get_K_in_pixel_coordinates(w, h))
        R = np.asarray(shot.pose.get_rotation_matrix())
        t = np.asarray(shot.pose.translation)
        rasterizer.scatter_cpu(
            K, R, t,
            depth.astype(np.float32, copy=False),
            plane.astype(np.float32, copy=False),
            confidence.astype(np.float32, copy=False),
        )

    grid: NDArray = rasterizer.finish_percentile(percentile)
    logger.info(
        "Rasterized: %d valid cells / %d total",
        int(np.count_nonzero(~np.isnan(grid))),
        grid.size,
    )

    # --- Post-process median filter ---
    if median_radius > 0:
        from scipy.ndimage import median_filter

        kernel_size = 2 * median_radius + 1
        valid_mask = ~np.isnan(grid)
        # median_filter doesn't handle NaN; substitute with a sentinel,
        # filter, then restore NaN.
        sentinel = np.nanmin(grid) - 9999.0 if valid_mask.any() else 0.0
        work = np.where(valid_mask, grid, sentinel)
        filtered = median_filter(work, size=kernel_size).astype(np.float32)
        grid = np.where(valid_mask, filtered, np.nan)
        logger.info("Applied %dx%d median filter", kernel_size, kernel_size)

    # --- Hole filling ---
    if config.get("dsm_fill_holes", True):
        dilation_radius: int = config.get("dsm_fill_max_radius", 3)
        triangulate: bool = config.get("dsm_fill_triangulate", True)
        grid = _fill_holes(grid, dilation_radius, triangulate)

    # --- Save GeoTIFF ---
    reference = reconstruction.reference
    data.save_dsm(grid, float(min_xy[0]), float(min_xy[1]), gsd, reference)
    logger.info("DSM saved to %s", data.dsm_file())


def _fill_holes(
    grid: NDArray, dilation_radius: int, triangulate: bool
) -> NDArray:
    """Fill interior NaN holes while preserving exterior nodata.

    Strategy:
      1. Build an *interior mask* via morphological closing to distinguish
         genuine interior holes from the exterior nodata boundary.
      2. Small-hole dilation (explicit 4-neighbor mean, no zero padding)
         up to ``dilation_radius`` pixels.
      3. Triangulation-based interpolation (Delaunay + barycentric) fills
         remaining interior holes.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.ndimage import binary_closing

    filled = grid.copy()
    valid = ~np.isnan(filled)
    if valid.all():
        return filled

    # --- Build interior mask -----------------------------------------------
    close_radius = max(dilation_radius * 2, 10)
    struct = np.ones((close_radius * 2 + 1, close_radius * 2 + 1), dtype=bool)
    interior_hull = binary_closing(valid, structure=struct, iterations=1)
    interior_hole = ~valid & interior_hull

    logger.info(
        "Holes: %d interior, %d exterior",
        int(interior_hole.sum()),
        int((~valid & ~interior_hull).sum()),
    )

    # --- Pass 1: 4-neighbor dilation (no zero-padding) ---------------------
    h, w = filled.shape
    for _r in range(dilation_radius):
        hole = np.isnan(filled) & interior_hole
        if not hole.any():
            break
        # Accumulate valid neighbors via explicit shifts (no filter padding).
        sum_z = np.zeros_like(filled)
        count = np.zeros_like(filled)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            # Slice source and destination so we never go out of bounds.
            sy = slice(max(0, -dy), h + min(0, -dy))
            sx = slice(max(0, -dx), w + min(0, -dx))
            ty = slice(max(0, dy), h + min(0, dy))
            tx = slice(max(0, dx), w + min(0, dx))
            src = filled[sy, sx]
            vmask = ~np.isnan(src)
            sum_z[ty, tx] += np.where(vmask, src, 0.0)
            count[ty, tx] += vmask.astype(np.float64)
        fillable = hole & (count > 0)
        filled[fillable] = (sum_z[fillable] / count[fillable]).astype(
            np.float32
        )

    # --- Pass 2: triangulation for remaining interior holes ----------------
    if triangulate:
        remaining = np.isnan(filled) & interior_hole
        n_remaining = int(remaining.sum())
        if n_remaining > 0:
            logger.info(
                "Triangulation fill: %d remaining interior cells", n_remaining
            )
            vy, vx = np.nonzero(~np.isnan(filled))
            vz = filled[vy, vx]
            hy, hx = np.nonzero(remaining)
            interp = LinearNDInterpolator(
                np.column_stack((vx, vy)), vz
            )
            hz = interp(np.column_stack((hx, hy)))
            good = ~np.isnan(hz)
            filled[hy[good], hx[good]] = hz[good]

    return filled
