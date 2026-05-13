# pyre-strict
"""Fast preview mode for OpenSfM.

Provides a multi-stage pipeline that runs feature extraction, matching,
and incremental reconstruction in parallel processes, producing per-image
reconstruction snapshots and meshes as visual feedback.

The pipeline deliberately skips the global ``create_tracks`` stage and
instead builds ad-hoc tracks from pairwise matches on the fly.
"""

import logging
import multiprocessing as mp
from collections import defaultdict
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from opensfm import (
    features_processing,
    matching,
    mesh,
    multiview,
    pairs_selection,
    pygeometry,
    reconstruction as sfm,
    reconstruction_helpers as helpers,
    types,
)
from opensfm.actions import extract_metadata as metadata_action
from opensfm.align import align_reconstruction
from opensfm.dataset import DataSet
from opensfm.preview_dataset import PreviewDataset


logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preview-specific config overrides
# ---------------------------------------------------------------------------

PREVIEW_CONFIG_OVERRIDES: Dict[str, Any] = {
    # Smaller images for speed
    "feature_process_size": 1024,
    # Cap features to 4000 per image
    "feature_min_frames": 4000,
    # Use CPU FLANN matcher
    "matcher_type": "FLANN",
    # Pair selection: order + time neighbors, max 5 each
    "matching_order_neighbors": 5,
    "matching_time_neighbors": 5,
    "matching_graph_rounds": 10,
    # Disable heavier pair selection strategies
    "matching_gps_neighbors": 0,
    "matching_gps_distance": 0,
    "matching_bow_neighbors": 0,
    "matching_vlad_neighbors": 0,
}


# ---------------------------------------------------------------------------
# Ad-hoc track helpers
# ---------------------------------------------------------------------------


class AdHocTrackBuilder:
    """Manages ad-hoc tracks constructed from pairwise matches.

    Instead of a full TracksManager, we maintain a lightweight mapping from
    (image, feature_index) → track_id and track_id → list of (image, feature_index,
    observation) so that we can resect new images and triangulate without
    the global union-find step.
    """

    def __init__(self) -> None:
        self._next_track_id: int = 0
        # (image, feature_index) -> track_id
        self._feature_to_track: Dict[Tuple[str, int], str] = {}
        # track_id -> list of (image, feature_index)
        self._track_members: Dict[str, List[Tuple[str, int]]] = {}
        # Cache: image -> {track_id} for quick lookups
        self._image_tracks: Dict[str, Set[str]] = defaultdict(set)

    def _new_track_id(self) -> str:
        tid = str(self._next_track_id)
        self._next_track_id += 1
        return tid

    def add_matches(
        self,
        im1: str,
        im2: str,
        match_indices: NDArray,
    ) -> None:
        """Integrate pairwise matches into the track structure.

        ``match_indices`` is an (N, 2) array where each row is
        (feature_index_im1, feature_index_im2).
        """
        if len(match_indices) == 0:
            return

        for f1, f2 in match_indices:
            f1 = int(f1)
            f2 = int(f2)
            key1 = (im1, f1)
            key2 = (im2, f2)

            tid1 = self._feature_to_track.get(key1)
            tid2 = self._feature_to_track.get(key2)

            if tid1 is not None and tid2 is not None:
                if tid1 == tid2:
                    continue
                # Merge tracks: absorb tid2 into tid1
                for member in self._track_members[tid2]:
                    self._feature_to_track[member] = tid1
                    self._track_members[tid1].append(member)
                    self._image_tracks[member[0]].discard(tid2)
                    self._image_tracks[member[0]].add(tid1)
                del self._track_members[tid2]
            elif tid1 is not None:
                self._feature_to_track[key2] = tid1
                self._track_members[tid1].append(key2)
                self._image_tracks[im2].add(tid1)
            elif tid2 is not None:
                self._feature_to_track[key1] = tid2
                self._track_members[tid2].append(key1)
                self._image_tracks[im1].add(tid2)
            else:
                tid = self._new_track_id()
                self._feature_to_track[key1] = tid
                self._feature_to_track[key2] = tid
                self._track_members[tid] = [key1, key2]
                self._image_tracks[im1].add(tid)
                self._image_tracks[im2].add(tid)

    def get_tracks_for_image(self, image: str) -> Set[str]:
        """Return the set of track ids visible in *image*."""
        return self._image_tracks.get(image, set())

    def get_track_members(self, track_id: str) -> List[Tuple[str, int]]:
        """Return the list of (image, feature_index) for a track."""
        return self._track_members.get(track_id, [])

    def get_feature_track(self, image: str, feat_idx: int) -> Optional[str]:
        return self._feature_to_track.get((image, feat_idx))

    def common_tracks_between(
        self, im1: str, im2: str
    ) -> List[Tuple[str, int, int]]:
        """Return (track_id, feat_idx_im1, feat_idx_im2) for tracks seen in both images."""
        tracks1 = self._image_tracks.get(im1, set())
        tracks2 = self._image_tracks.get(im2, set())
        common_tids = tracks1 & tracks2
        result = []
        for tid in common_tids:
            f1 = f2 = None
            for img, fidx in self._track_members[tid]:
                if img == im1 and f1 is None:
                    f1 = fidx
                elif img == im2 and f2 is None:
                    f2 = fidx
            if f1 is not None and f2 is not None:
                result.append((tid, f1, f2))
        return result

    def get_common_features(
        self,
        im1: str,
        im2: str,
        features1: NDArray,
        features2: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Return aligned feature coordinate arrays for common tracks."""
        common = self.common_tracks_between(im1, im2)
        if not common:
            return np.empty((0, 2)), np.empty((0, 2))
        _, idxs1, idxs2 = zip(*common)
        return features1[list(idxs1), :2], features2[list(idxs2), :2]


# ---------------------------------------------------------------------------
# Per-view mesh (uses mesh.triangle_mesh_from_points)
# ---------------------------------------------------------------------------


def _add_shot(
    data: PreviewDataset,
    reconstruction: types.Reconstruction,
    shot_id: str,
    pose: pygeometry.Pose,
) -> None:
    """Add a shot to the reconstruction (simplified, no rig support)."""
    exif = data.load_exif(shot_id)
    camera_id = exif["camera"]
    shot = reconstruction.create_shot(shot_id, camera_id, pose)
    reference = data.load_reference()
    shot.metadata = helpers.exif_to_metadata(
        exif, data.config["use_altitude_tag"], reference
    )


# ---------------------------------------------------------------------------
# Fast incremental reconstruction
# ---------------------------------------------------------------------------


def _load_features_for_image(
    data: PreviewDataset, image: str
) -> Optional[NDArray]:
    """Load feature points (x, y, size) for an image."""
    fd = data.load_features(image)
    if fd is None:
        return None
    return fd.points[:, :3]


class CandidateGraph:
    """Pre-built adjacency graph for O(1) next-best-candidate lookup.

    Stores, for every image, its neighbors and the number of matches to each.
    Maintains a running score (total matches to already-reconstructed images)
    so that picking the next candidate is O(1) via a sorted structure.
    """

    def __init__(self, match_cache: Dict[str, Dict[str, NDArray]]) -> None:
        # adjacency: image -> {neighbor -> match_count}
        self._adj: Dict[str, Dict[str, int]] = defaultdict(dict)
        for im1, matches_dict in match_cache.items():
            for im2, match_arr in matches_dict.items():
                n = len(match_arr)
                if n == 0:
                    continue
                self._adj[im1][im2] = n
                # Store reverse only if not already present
                if im1 not in self._adj[im2]:
                    self._adj[im2][im1] = n

        # score: total match count to reconstructed images
        self._score: Dict[str, int] = defaultdict(int)
        # best partner for each candidate
        self._best_partner: Dict[str, Tuple[str, int]] = {}

    def mark_reconstructed(self, image: str) -> None:
        """Update scores of all neighbors when *image* is reconstructed."""
        for neighbor, count in self._adj.get(image, {}).items():
            self._score[neighbor] += count
            cur = self._best_partner.get(neighbor)
            if cur is None or count > cur[1]:
                self._best_partner[neighbor] = (image, count)

    def pop_best_candidate(
        self, remaining: Set[str]
    ) -> Optional[Tuple[str, str, int]]:
        """Return (candidate, best_partner, total_score) for the best unreconstructed image."""
        best: Optional[Tuple[str, str, int]] = None
        for cand in remaining:
            s = self._score.get(cand, 0)
            if s <= 0:
                continue
            if best is None or s > best[2]:
                partner_info = self._best_partner.get(cand)
                partner = partner_info[0] if partner_info else ""
                best = (cand, partner, s)
        return best

    def neighbors(self, image: str) -> Dict[str, int]:
        """Return {neighbor -> match_count} for *image*."""
        return self._adj.get(image, {})


def _resect_image(
    reconstruction: types.Reconstruction,
    image: str,
    features_pts: NDArray,
    track_builder: AdHocTrackBuilder,
    camera_priors: Dict[str, pygeometry.Camera],
    data: PreviewDataset,
    threshold: float,
) -> Optional[pygeometry.Pose]:
    """Try to resect (PnP) an image into the reconstruction.

    Uses the ad-hoc tracks to find 2D-3D correspondences.
    Returns the estimated pose or None if resection fails.
    """
    image_tracks = track_builder.get_tracks_for_image(image)

    bearings = []
    points_3d = []
    for tid in image_tracks:
        if tid not in reconstruction.points:
            continue
        # Find the feature index for this image in the track
        for member_im, member_fidx in track_builder.get_track_members(tid):
            if member_im == image:
                pt_2d = features_pts[member_fidx, :2]
                pt_3d = reconstruction.points[tid].coordinates
                bearings.append(pt_2d)
                points_3d.append(pt_3d)
                break

    if len(bearings) < 6:
        return None

    bearings_arr = np.array(bearings)
    points_3d_arr = np.array(points_3d)

    exif = data.load_exif(image)
    camera = camera_priors[exif["camera"]]
    bearing_vectors = camera.pixel_bearing_many(bearings_arr)

    try:
        Rt = multiview.absolute_pose_ransac(
            bearing_vectors, points_3d_arr, threshold, 1000, 0.999
        )
    except Exception:
        return None

    R = Rt[:3, :3]
    t = Rt[:, 3]
    pose = pygeometry.Pose(cv2.Rodrigues(R)[0].ravel(), t)
    return pose


def _triangulate_tracks_for_image(
    reconstruction: types.Reconstruction,
    image: str,
    features_cache: Dict[str, NDArray],
    track_builder: AdHocTrackBuilder,
    camera_priors: Dict[str, pygeometry.Camera],
    data: PreviewDataset,
    min_ray_angle: float = 2.0,
) -> int:
    """Triangulate new points for tracks visible in *image* that don't have a 3D point yet.

    Returns the number of newly triangulated points.
    """
    count = 0
    image_tracks = track_builder.get_tracks_for_image(image)

    for tid in image_tracks:
        if tid in reconstruction.points:
            continue
        members = track_builder.get_track_members(tid)
        # Collect bearings from reconstructed shots
        shot_bearings = []
        shot_origins = []
        for member_im, member_fidx in members:
            if member_im not in reconstruction.shots:
                continue
            if member_im not in features_cache:
                continue
            shot = reconstruction.shots[member_im]
            pt_2d = features_cache[member_im][member_fidx, :2]
            camera = camera_priors[shot.camera.id]
            bearing = camera.pixel_bearing(pt_2d)
            # Transform bearing to world frame
            R = shot.pose.get_R_cam_to_world()
            bearing_world = R @ bearing
            origin = shot.pose.get_origin()
            shot_bearings.append(bearing_world)
            shot_origins.append(origin)

        if len(shot_bearings) < 2:
            continue

        # Simple two-view triangulation using the first two views
        b1 = shot_bearings[0]
        o1 = shot_origins[0]
        b2 = shot_bearings[1]
        o2 = shot_origins[1]

        # Check ray angle
        cos_angle = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(abs(cos_angle)))
        if angle_deg < min_ray_angle:
            continue

        # Linear triangulation
        point_3d = _linear_triangulation(o1, b1, o2, b2)
        if point_3d is None:
            continue

        # Check that point is in front of all cameras
        valid = True
        for member_im, _ in members:
            if member_im not in reconstruction.shots:
                continue
            shot = reconstruction.shots[member_im]
            pt_cam = shot.pose.transform(point_3d)
            if pt_cam[2] <= 0:
                valid = False
                break

        if not valid:
            continue

        reconstruction.create_point(tid, point_3d)
        count += 1

    return count


def _linear_triangulation(
    o1: NDArray, d1: NDArray, o2: NDArray, d2: NDArray
) -> Optional[NDArray]:
    """Triangulate a 3D point from two rays (origin + direction)."""
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    # Solve for closest point between two rays
    w0 = o1 - o2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    p1 = o1 + s * d1
    p2 = o2 + t * d2

    return (p1 + p2) / 2.0


def _save_per_image_mesh(
    data: PreviewDataset,
    reconstruction: types.Reconstruction,
    image: str,
) -> None:
    """Generate and save a per-image Delaunay mesh."""
    if image not in reconstruction.shots:
        return

    shot = reconstruction.shots[image]

    # Collect points visible in this shot
    point_coords_list: List[NDArray] = []
    for pid, point in reconstruction.points.items():
        coord = point.coordinates
        projected = shot.project(coord)
        if not np.isnan(projected).any():
            point_coords_list.append(coord)

    if len(point_coords_list) < 3:
        return

    point_coords = np.array(point_coords_list)
    vertices, faces = mesh.triangle_mesh_from_points(shot, point_coords)

    if vertices and faces:
        data.save_preview_mesh(image, vertices, faces)


# ---------------------------------------------------------------------------
# EXIF-based image ordering from putative pairs
# ---------------------------------------------------------------------------


def _compute_image_order(
    data: DataSet,
    config_override: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Compute a processing order and per-image putative neighbors from EXIF.

    Uses the same pair-selection strategies as the matcher (order, time, GPS)
    but purely on EXIF metadata — no features needed.

    Returns:
        ordered_images: images sorted by decreasing connectivity (number of
            putative pairs).
        neighbors: for each image, the set of its putative match partners.
    """
    images = data.images()
    exifs = {im: data.load_exif(im) for im in images}

    pairs, _ = pairs_selection.match_candidates_from_metadata(
        images, images, exifs, data, config_override,
    )

    # Build adjacency and degree from putative pairs
    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for im1, im2 in pairs:
        neighbors[im1].add(im2)
        neighbors[im2].add(im1)

    # BFS ordering from the most-connected image for spatial locality
    start = max(images, key=lambda im: len(neighbors.get(im, set())))
    visited_set: Set[str] = set()
    ordered: List[str] = []
    queue: List[str] = [start]
    while queue:
        im = queue.pop(0)
        if im in visited_set:
            continue
        visited_set.add(im)
        ordered.append(im)
        # Enqueue neighbors sorted by descending connectivity
        nbrs = sorted(
            neighbors.get(im, set()) - visited_set,
            key=lambda n: len(neighbors.get(n, set())),
            reverse=True,
        )
        queue.extend(nbrs)

    # Append any disconnected images at the end
    for im in images:
        if im not in visited_set:
            ordered.append(im)

    logger.info(
        "Preview: image ordering computed — %d images, %d putative pairs, "
        "starting from %s (%d neighbors)",
        len(ordered),
        len(pairs),
        start,
        len(neighbors.get(start, set())),
    )
    return ordered, dict(neighbors)


# ---------------------------------------------------------------------------
# Streaming workers
# ---------------------------------------------------------------------------


def _feature_extraction_worker(
    data_path: str,
    config_override: Dict[str, Any],
    ordered_images: List[str],
    feat_queue: "mp.Queue[Optional[str]]",
) -> None:
    """Process 1: extract features one image at a time in the given order."""
    data = DataSet(data_path)
    data.config.update(config_override)

    for image in ordered_images:
        if data.features_exist(image):
            feat_queue.put(image)
            continue
        try:
            features_processing.run_features_processing(data, [image], False)
            feat_queue.put(image)
        except Exception:
            logger.exception("Feature extraction failed for %s", image)
    feat_queue.put(None)  # sentinel


def _matching_worker(
    data_path: str,
    config_override: Dict[str, Any],
    putative_neighbors: Dict[str, List[str]],
    feat_queue: "mp.Queue[Optional[str]]",
    match_queue: "mp.Queue[Optional[str]]",
) -> None:
    """Process 2: match each image against its putative neighbors that already have features.

    Only matches against neighbors whose features are already extracted
    (i.e. already seen from the feat_queue).
    """
    data = DataSet(data_path)
    data.config.update(config_override)

    features_ready: Set[str] = set()

    while True:
        image = feat_queue.get()
        if image is None:
            break
        features_ready.add(image)

        # Match this image against its putative neighbors that are ready
        neighbors = putative_neighbors.get(image, [])
        ready_neighbors = [
            n for n in neighbors if n in features_ready and n != image]

        if ready_neighbors:
            try:
                pairs = [(image, n) for n in ready_neighbors]
                exifs = {im: data.load_exif(im)
                         for im in [image] + ready_neighbors}
                matched = matching.match_images_with_pairs(
                    data, config_override, exifs, pairs,
                )
                # Save matches: store under the ref image
                matching.save_matches(data, [image], matched)
            except Exception:
                logger.exception("Matching failed for %s", image)

        match_queue.put(image)

    match_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Streaming incremental reconstruction
# ---------------------------------------------------------------------------


def _streaming_incremental_reconstruction(
    data: PreviewDataset,
    match_queue: "mp.Queue[Optional[str]]",
) -> Tuple[Dict[str, Any], Optional[types.Reconstruction]]:
    """Incremental SfM that consumes images from the match queue.

    Starts bootstrapping as soon as the first viable pair arrives,
    then resects + triangulates each subsequent image as it becomes available.
    """
    config = data.config.copy()
    config.update(PREVIEW_CONFIG_OVERRIDES)
    report: Dict[str, Any] = {"steps": [], "num_images": 0}

    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()

    all_images = data.images()
    data.init_reference(all_images)

    features_cache: Dict[str, NDArray] = {}
    match_cache: Dict[str, Dict[str, NDArray]] = {}
    track_builder = AdHocTrackBuilder()
    candidate_graph = CandidateGraph({})  # starts empty

    reconstruction: Optional[types.Reconstruction] = None
    reconstructed_images: Set[str] = set()
    deferred_images: List[str] = []  # images waiting for bootstrap
    threshold = config["five_point_algo_threshold"]
    min_inliers = config.get("five_point_algo_min_inliers", 50)

    def _ingest_image(image: str) -> None:
        """Load features + matches for a newly-matched image into caches."""
        feats = _load_features_for_image(data, image)
        if feats is not None:
            features_cache[image] = feats

        if data.matches_exists(image):
            matches_dict = data.load_matches(image)
        else:
            matches_dict = {}
        if matches_dict:
            match_cache[image] = matches_dict
            # Update candidate graph adjacency
            for other, marr in matches_dict.items():
                n = len(marr)
                if n == 0:
                    continue
                candidate_graph._adj[image][other] = n
                if image not in candidate_graph._adj[other]:
                    candidate_graph._adj[other][image] = n

        if image in reconstructed_images:
            candidate_graph.mark_reconstructed(image)

    def _try_bootstrap(im1: str, im2: str) -> bool:
        """Attempt to bootstrap a reconstruction from a pair."""
        nonlocal reconstruction
        for im in (im1, im2):
            if im not in features_cache:
                return False

        matches_arr = data.find_matches(im1, im2)
        if len(matches_arr) == 0:
            return False
        track_builder.add_matches(im1, im2, matches_arr)

        p1, p2 = track_builder.get_common_features(
            im1, im2, features_cache[im1], features_cache[im2]
        )
        if len(p1) < min_inliers:
            return False

        camera1 = camera_priors[data.load_exif(im1)["camera"]]
        camera2 = camera_priors[data.load_exif(im2)["camera"]]
        iterations = config["five_point_refine_rec_iterations"]

        R, t, inliers, _ = sfm.two_view_reconstruction_general(
            p1, p2, camera1, camera2, threshold, iterations
        )
        if R is None or t is None or len(inliers) < min_inliers:
            return False

        reconstruction = types.Reconstruction()
        reconstruction.reference = data.load_reference()
        reconstruction.cameras = camera_priors
        reconstruction.rig_cameras = rig_camera_priors

        _add_shot(data, reconstruction, im1, pygeometry.Pose())
        _add_shot(data, reconstruction, im2, pygeometry.Pose(R, t))

        _triangulate_tracks_for_image(
            reconstruction, im1, features_cache, track_builder, camera_priors, data
        )
        _triangulate_tracks_for_image(
            reconstruction, im2, features_cache, track_builder, camera_priors, data
        )

        if len(reconstruction.points) < min_inliers:
            reconstruction = None
            return False

        sfm.bundle_shot_poses(
            reconstruction, {im2}, camera_priors, rig_camera_priors, config,
        )
        align_reconstruction(reconstruction, [], config)

        reconstructed_images.add(im1)
        reconstructed_images.add(im2)
        candidate_graph.mark_reconstructed(im1)
        candidate_graph.mark_reconstructed(im2)

        data.save_preview_reconstruction(im1, reconstruction)
        data.save_preview_reconstruction(im2, reconstruction)
        _save_per_image_mesh(data, reconstruction, im1)
        _save_per_image_mesh(data, reconstruction, im2)

        logger.info(
            "Preview: bootstrapped with %s and %s (%d points)",
            im1, im2, len(reconstruction.points),
        )
        report["bootstrap_pair"] = (im1, im2)
        report["bootstrap_points"] = len(reconstruction.points)
        return True

    def _try_add_image(image: str) -> bool:
        """Try to resect and triangulate a single image."""
        if image in reconstructed_images or reconstruction is None:
            return False
        if image not in features_cache:
            return False

        t_start = timer()

        # Feed matches between this image and reconstructed neighbors
        for neighbor in candidate_graph.neighbors(image):
            if neighbor not in reconstructed_images:
                continue
            matches_arr = data.find_matches(image, neighbor)
            if len(matches_arr) > 0:
                track_builder.add_matches(image, neighbor, matches_arr)

        pose = _resect_image(
            reconstruction, image, features_cache[image],
            track_builder, camera_priors, data, threshold,
        )
        if pose is None:
            return False

        _add_shot(data, reconstruction, image, pose)
        reconstructed_images.add(image)
        candidate_graph.mark_reconstructed(image)

        new_points = _triangulate_tracks_for_image(
            reconstruction, image, features_cache, track_builder, camera_priors, data,
        )

        sfm.bundle_shot_poses(
            reconstruction, {image}, camera_priors, rig_camera_priors, config,
        )

        t_elapsed = timer() - t_start
        step = len(report["steps"]) + 1
        logger.info(
            "Preview step %d: added %s in %.2fs "
            "(%d new pts, %d total shots, %d total pts)",
            step, image, t_elapsed, new_points,
            len(reconstruction.shots), len(reconstruction.points),
        )

        data.save_preview_reconstruction(image, reconstruction)
        _save_per_image_mesh(data, reconstruction, image)

        report["steps"].append({
            "image": image,
            "new_points": new_points,
            "total_shots": len(reconstruction.shots),
            "total_points": len(reconstruction.points),
            "time": round(t_elapsed, 3),
        })
        return True

    # --- Main consumption loop ---
    while True:
        image = match_queue.get()
        if image is None:
            break

        _ingest_image(image)

        if reconstruction is None:
            # Not yet bootstrapped — try every pair we have so far
            deferred_images.append(image)
            bootstrapped = False
            for im in deferred_images:
                if im not in match_cache:
                    continue
                for other in match_cache[im]:
                    if other in features_cache and _try_bootstrap(im, other):
                        bootstrapped = True
                        break
                if bootstrapped:
                    break

            if bootstrapped:
                # Try to add all deferred images
                for dim in deferred_images:
                    if dim not in reconstructed_images:
                        _try_add_image(dim)
                deferred_images.clear()
        else:
            _try_add_image(image)

    # --- Final pass: retry any remaining images ---
    all_remaining = set(all_images) - reconstructed_images
    if reconstruction is not None and all_remaining:
        # Re-ingest anything we might have missed
        for im in list(all_remaining):
            if im not in match_cache:
                if data.matches_exists(im):
                    _ingest_image(im)
        for _ in range(3):  # multiple passes for chain dependencies
            added = False
            for im in list(all_remaining):
                if _try_add_image(im):
                    all_remaining.discard(im)
                    added = True
            if not added:
                break

    if reconstruction is not None:
        report["num_images"] = len(reconstruction.shots)
        report["num_points"] = len(reconstruction.points)
        data.save_reconstruction([reconstruction])
        logger.info(
            "Preview done: %d images, %d points",
            len(reconstruction.shots), len(reconstruction.points),
        )
    else:
        logger.warning("Preview: could not bootstrap any pair")

    return report, reconstruction


# ---------------------------------------------------------------------------
# Entry point: streaming pipeline
# ---------------------------------------------------------------------------


def run_preview_dataset(data: DataSet) -> Dict[str, Any]:
    """Run the full streaming preview pipeline.

    Stages:
    0. Extract metadata (EXIF + camera models) — synchronous, fast
    1. Compute image ordering from EXIF-based putative pairs
    2. Feature extraction process — one image at a time in order
    3. Matching process — matches each image against ready neighbors
    4. Incremental reconstruction — bootstraps ASAP, adds images as they arrive

    Stages 2, 3, 4 run as three concurrent processes connected by queues.
    """
    report: Dict[str, Any] = {}

    # Apply preview overrides
    data.config.update(PREVIEW_CONFIG_OVERRIDES)

    # 0. Extract metadata (synchronous) ------------------------------------
    t0 = timer()
    needs_metadata = any(not data.exif_exists(im) for im in data.images())
    if needs_metadata:
        logger.info("Preview: extracting metadata for %d images",
                    len(data.images()))
        metadata_action.run_dataset(data)
    else:
        logger.info("Preview: metadata already exists, skipping extraction")
    report["metadata_time"] = round(timer() - t0, 3)

    # 1. Compute processing order from EXIF pairs --------------------------
    t0 = timer()
    ordered_images, neighbors = _compute_image_order(
        data, PREVIEW_CONFIG_OVERRIDES)
    # Convert neighbor sets to lists for pickling across processes
    neighbors_lists: Dict[str, List[str]] = {
        im: list(nbrs) for im, nbrs in neighbors.items()
    }
    report["ordering_time"] = round(timer() - t0, 3)

    # 2-4. Launch streaming pipeline ----------------------------------------
    feat_queue: mp.Queue = mp.Queue(maxsize=50)
    match_queue: mp.Queue = mp.Queue(maxsize=50)

    feat_proc = mp.Process(
        target=_feature_extraction_worker,
        args=(data.data_path, PREVIEW_CONFIG_OVERRIDES,
              ordered_images, feat_queue),
    )
    match_proc = mp.Process(
        target=_matching_worker,
        args=(data.data_path, PREVIEW_CONFIG_OVERRIDES,
              neighbors_lists, feat_queue, match_queue),
    )

    feat_proc.start()
    match_proc.start()

    # Reconstruction runs in the main process, consuming from match_queue
    preview_data = PreviewDataset(data)
    rec_report, reconstruction = _streaming_incremental_reconstruction(
        preview_data, match_queue,
    )

    feat_proc.join()
    match_proc.join()

    report["reconstruction"] = rec_report
    return report
