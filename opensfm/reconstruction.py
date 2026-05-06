# pyre-strict
"""Incremental reconstruction pipeline"""

import datetime
import enum
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from opensfm import (
    config,
    matching,
    multiview,
    pygeometry,
    pymap,
    pysfm,
    reconstruction_helpers as helpers,
    rig,
    types,
)
from opensfm.align import align_reconstruction, apply_similarity
from opensfm import context
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)


class ReconstructionAlgorithm(str, enum.Enum):
    INCREMENTAL = "incremental"
    TRIANGULATION = "triangulation"


def log_bundle_stats(bundle_type: str, bundle_report: Dict[str, Any]) -> None:
    times = bundle_report["wall_times"]
    time_secs = times["run"] + times["setup"] + \
        times["teardown"] + times["triangulate"]
    num_images, num_points, num_reprojections = (
        bundle_report["num_images"],
        bundle_report["num_points"],
        bundle_report["num_reprojections"],
    )

    msg = f"Ran {bundle_type} bundle in {time_secs:.2f} secs."
    if num_points > 0:
        msg += f"with {num_images}/{num_points}/{num_reprojections} ({num_reprojections / num_points:.2f}) "
        msg += "shots/points/proj. (avg. length)"

    logger.info(msg)


def bundle(
    reconstruction: types.Reconstruction,
    camera_priors: Dict[str, pygeometry.Camera],
    rig_camera_priors: Dict[str, pymap.RigCamera],
    gcp: Optional[List[pymap.GroundControlPoint]],
    grid_size: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Bundle adjust a reconstruction."""
    report = pysfm.BAHelpers.bundle(
        reconstruction.map,
        dict(camera_priors),
        dict(rig_camera_priors),
        gcp if gcp is not None else [],
        grid_size,
        config,
    )
    log_bundle_stats("GLOBAL", report)
    logger.debug(report["brief_report"])
    for line in report["irls_report"]:
        logger.debug(line)
    return report


def bundle_with_gcp_annealing(
    reconstruction: types.Reconstruction,
    camera_priors: Dict[str, pygeometry.Camera],
    rig_camera_priors: Dict[str, pymap.RigCamera],
    gcp: List[pymap.GroundControlPoint],
    grid_size: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Bundle adjust with graduated GCP weight annealing.

    Runs multiple bundle passes with increasing GCP weight multipliers
    to smoothly steer the reconstruction toward GCP constraints.
    """
    annealing_steps = config.get("gcp_annealing_steps", [1.0])
    base_weight = config["gcp_global_weight"]
    report = {}
    for i, multiplier in enumerate(annealing_steps):
        step_config = config.copy()
        step_config["gcp_global_weight"] = base_weight * multiplier
        logger.info(
            "GCP annealing step %d/%d: weight_multiplier=%.2f, "
            "effective_gcp_global_weight=%.4f",
            i + 1, len(annealing_steps), multiplier,
            step_config["gcp_global_weight"],
        )
        report = bundle(
            reconstruction, camera_priors, rig_camera_priors,
            gcp, grid_size, step_config,
        )
    return report


def bundle_shot_poses(
    reconstruction: types.Reconstruction,
    shot_ids: Set[str],
    camera_priors: Dict[str, pygeometry.Camera],
    rig_camera_priors: Dict[str, pymap.RigCamera],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Bundle adjust a set of shots poses."""
    report = pysfm.BAHelpers.bundle_shot_poses(
        reconstruction.map,
        shot_ids,
        dict(camera_priors),
        dict(rig_camera_priors),
        config,
    )
    return report


def pairwise_reconstructability(common_tracks: int, rotation_inliers: int) -> float:
    """Likeliness of an image pair giving a good initial reconstruction."""
    outliers = common_tracks - rotation_inliers
    outlier_ratio = float(outliers) / common_tracks
    if outlier_ratio >= 0.3:
        return outliers
    else:
        return 0


def calculate_pair_reconstructability(
    tracks_manager: pymap.TracksManager,
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    threshold: float,
) -> float:
    p1, p2 = _get_common_feature_arrays(tracks_manager, im1, im2)
    R, inliers = two_view_reconstruction_rotation_only(
        p1, p2, camera1, camera2, threshold
    )
    return pairwise_reconstructability(len(p1), len(inliers))


TPairArguments = Tuple[
    str, str, NDArray, NDArray, pygeometry.Camera, pygeometry.Camera, float
]


def add_shot(
    data: DataSetBase,
    reconstruction: types.Reconstruction,
    rig_assignments: Dict[str, Tuple[str, str, List[str]]],
    shot_id: str,
    pose: pygeometry.Pose,
) -> Set[str]:
    """Add a shot to the reconstruction.

    In case of a shot belonging to a rig instance, the pose of
    shot will drive the initial pose setup of the rig instance.
    All necessary shots and rig models will be created.
    """

    added_shots = set()
    if shot_id not in rig_assignments:
        camera_id = data.load_exif(shot_id)["camera"]
        shot = reconstruction.create_shot(shot_id, camera_id, pose)
        shot.metadata = helpers.get_image_metadata(data, shot_id)
        added_shots = {shot_id}
    else:
        instance_id, _, instance_shots = rig_assignments[shot_id]
        rig_instance = reconstruction.add_rig_instance(
            pymap.RigInstance(instance_id))

        for shot in instance_shots:
            _, rig_camera_id, _ = rig_assignments[shot]
            camera_id = data.load_exif(shot)["camera"]
            created_shot = reconstruction.create_shot(
                shot,
                camera_id,
                pygeometry.Pose(),
                rig_camera_id,
                instance_id,
            )
            created_shot.metadata = helpers.get_image_metadata(data, shot)
        rig_instance.update_instance_pose_with_shot(shot_id, pose)
        added_shots = set(instance_shots)

    return added_shots


def _two_view_reconstruction_inliers(
    b1: NDArray, b2: NDArray, R: NDArray, t: NDArray, threshold: float
) -> List[int]:
    """Returns indices of matches that can be triangulated."""
    ok = matching.compute_inliers_bearings(b1, b2, R, t, threshold)
    # pyre-fixme[7]: Expected `List[int]` but got `ndarray[typing.Any,
    #  dtype[typing.Any]]`.
    return np.nonzero(ok)[0]


def two_view_reconstruction_plane_based(
    b1: NDArray,
    b2: NDArray,
    threshold: float,
) -> Tuple[Optional[NDArray], Optional[NDArray], List[int]]:
    """Reconstruct two views from point correspondences lying on a plane.

    Args:
        b1, b2: lists bearings in the images
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    x1 = multiview.euclidean(b1)
    x2 = multiview.euclidean(b2)

    H, inliers = cv2.findHomography(x1, x2, cv2.RANSAC, threshold)
    motions = multiview.motion_from_plane_homography(H)

    if not motions:
        return None, None, []

    if len(motions) == 0:
        return None, None, []

    motion_inliers = []
    for R, t, _, _ in motions:
        inliers = _two_view_reconstruction_inliers(
            b1, b2, R.T, -R.T.dot(t), threshold)
        motion_inliers.append(inliers)

    # pyre-fixme[6]: For 1st argument expected `Union[_SupportsArray[dtype[typing.Any...
    best = np.argmax(map(len, motion_inliers))
    R, t, n, d = motions[best]
    inliers = motion_inliers[best]
    return cv2.Rodrigues(R)[0].ravel(), t, inliers


def two_view_reconstruction_and_refinement(
    b1: NDArray,
    b2: NDArray,
    R: NDArray,
    t: NDArray,
    threshold: float,
    iterations: int,
    transposed: bool,
) -> Tuple[NDArray, NDArray, List[int]]:
    """Reconstruct two views using provided rotation and translation.

    Args:
        b1, b2: lists bearings in the images
        R, t: rotation & translation
        threshold: reprojection error threshold
        iterations: number of iteration for refinement
        transposed: use transposed R, t instead

    Returns:
        rotation, translation and inlier list
    """
    if transposed:
        t_curr = -R.T.dot(t)
        R_curr = R.T
    else:
        t_curr = t.copy()
        R_curr = R.copy()

    inliers = _two_view_reconstruction_inliers(
        b1, b2, R_curr, t_curr, threshold)

    if len(inliers) > 5:
        T = multiview.relative_pose_optimize_nonlinear(
            b1[inliers], b2[inliers], t_curr, R_curr, iterations
        )
        R_curr = T[:, :3]
        t_curr = T[:, 3]
        inliers = _two_view_reconstruction_inliers(
            b1, b2, R_curr, t_curr, threshold)

    return cv2.Rodrigues(R_curr.T)[0].ravel(), -R_curr.T.dot(t_curr), inliers


def _two_view_rotation_inliers(
    b1: NDArray, b2: NDArray, R: NDArray, threshold: float
) -> List[int]:
    br2 = R.dot(b2.T).T
    ok = np.linalg.norm(br2 - b1, axis=1) < threshold
    # pyre-fixme[7]: Expected `List[int]` but got `ndarray[typing.Any,
    #  dtype[typing.Any]]`.
    return np.nonzero(ok)[0]


def two_view_reconstruction_rotation_only(
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    threshold: float,
) -> Tuple[NDArray, List[int]]:
    """Find rotation between two views from point correspondences.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation and inlier list
    """
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    R = multiview.relative_pose_ransac_rotation_only(
        b1, b2, threshold, 1000, 0.999)
    inliers = _two_view_rotation_inliers(b1, b2, R, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), inliers


def two_view_reconstruction_5pt(
    b1: NDArray,
    b2: NDArray,
    R: NDArray,
    t: NDArray,
    threshold: float,
    iterations: int,
    check_reversal: bool = False,
    reversal_ratio: float = 1.0,
) -> Tuple[Optional[NDArray], Optional[NDArray], List[int]]:
    """Run 5-point reconstruction and refinement, given computed relative rotation and translation.

    Optionally, the method will perform reconstruction and refinement for both given and transposed
    rotation and translation.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold
        iterations: number of step for the non-linear refinement of the relative pose
        check_reversal: whether to check for Necker reversal ambiguity
        reversal_ratio: ratio of triangulated point between normal and reversed
                        configuration to consider a pair as being ambiguous

    Returns:
        rotation, translation and inlier list
    """

    configurations = [False, True] if check_reversal else [False]

    # Refine both normal and transposed relative motion
    results_5pt = []
    for transposed in configurations:
        R_5p, t_5p, inliers_5p = two_view_reconstruction_and_refinement(
            b1,
            b2,
            R,
            t,
            threshold,
            iterations,
            transposed,
        )

        valid_curr_5pt = R_5p is not None and t_5p is not None
        if len(inliers_5p) <= 5 or not valid_curr_5pt:
            continue

        logger.info(
            f"Two-view 5-points reconstruction inliers (transposed={transposed}): {len(inliers_5p)} / {len(b1)}"
        )
        results_5pt.append((R_5p, t_5p, inliers_5p))

    # Use relative motion if one version stands out
    if len(results_5pt) == 1:
        R_5p, t_5p, inliers_5p = results_5pt[0]
    elif len(results_5pt) == 2:
        inliers1, inliers2 = results_5pt[0][2], results_5pt[1][2]
        len1, len2 = len(inliers1), len(inliers2)
        ratio = min(len1, len2) / max(len1, len2)
        if ratio > reversal_ratio:
            logger.warning(
                f"Un-decidable Necker configuration (ratio={ratio}), skipping."
            )
            R_5p, t_5p, inliers_5p = None, None, []
        else:
            index = 0 if len1 > len2 else 1
            R_5p, t_5p, inliers_5p = results_5pt[index]
    else:
        R_5p, t_5p, inliers_5p = None, None, []

    return R_5p, t_5p, inliers_5p


def two_view_reconstruction_general(  # pyre-ignore[3]: pyre is not happy with the Dict[str, Any]
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    threshold: float,
    iterations: int,
    check_reversal: bool = False,
    reversal_ratio: float = 1.0,
) -> Tuple[Optional[NDArray], Optional[NDArray], List[int], Dict[str, Any]]:
    """Reconstruct two views from point correspondences.

    These will try different reconstruction methods and return the
    results of the one with most inliers.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold
        iterations: number of step for the non-linear refinement of the relative pose
        check_reversal: whether to check for Necker reversal ambiguity
        reversal_ratio: ratio of triangulated point between normal and reversed
                        configuration to consider a pair as being ambiguous

    Returns:
        rotation, translation and inlier list
    """

    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    # Get 5-point relative motion
    T_robust = multiview.relative_pose_ransac(b1, b2, threshold, 1000, 0.999)
    R_robust = T_robust[:, :3]
    t_robust = T_robust[:, 3]
    R_5p, t_5p, inliers_5p = two_view_reconstruction_5pt(
        b1,
        b2,
        R_robust,
        t_robust,
        threshold,
        iterations,
        check_reversal,
        reversal_ratio,
    )
    valid_5pt = R_5p is not None and t_5p is not None

    # Compute plane-based relative-motion
    R_plane, t_plane, inliers_plane = two_view_reconstruction_plane_based(
        b1,
        b2,
        threshold,
    )
    valid_plane = R_plane is not None and t_plane is not None

    report: Dict[str, Any] = {
        "5_point_inliers": len(inliers_5p),
        "plane_based_inliers": len(inliers_plane),
    }

    if valid_5pt and len(inliers_5p) > len(inliers_plane):
        report["method"] = "5_point"
        R, t, inliers = R_5p, t_5p, inliers_5p
    elif valid_plane:
        report["method"] = "plane_based"
        R, t, inliers = R_plane, t_plane, inliers_plane
    else:
        report["decision"] = "Could not find initial motion"
        logger.info(report["decision"])
        R, t, inliers = None, None, []
    return R, t, inliers, report


def reconstruction_from_relative_pose(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    im1: str,
    im2: str,
    R: NDArray,
    t: NDArray,
) -> Tuple[Optional[types.Reconstruction], Dict[str, Any]]:
    """Create a reconstruction from 'im1' and 'im2' using the provided rotation 'R' and translation 't'."""
    report = {}

    min_inliers = data.config["five_point_algo_min_inliers"]

    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()
    rig_assignments = rig.rig_assignments_per_image(
        data.load_rig_assignments())

    reconstruction = types.Reconstruction()
    reconstruction.reference = data.load_reference()
    reconstruction.cameras = camera_priors
    reconstruction.rig_cameras = rig_camera_priors
    reconstruction.map.set_observation_pool(
        tracks_manager.get_observation_pool())

    new_shots = add_shot(data, reconstruction,
                         rig_assignments, im1, pygeometry.Pose())

    if im2 not in new_shots:
        new_shots |= add_shot(
            data, reconstruction, rig_assignments, im2, pygeometry.Pose(R, t)
        )

    align_reconstruction(reconstruction, [], data.config)
    retriangulate(tracks_manager, reconstruction, data.config)

    logger.info("Triangulated: {}".format(len(reconstruction.points)))
    report["triangulated_points"] = len(reconstruction.points)
    if len(reconstruction.points) < min_inliers:
        report["decision"] = "Initial motion did not generate enough points"
        logger.info(report["decision"])
        return None, report

    to_adjust = {s for s in new_shots if s != im1}
    bundle_shot_poses(
        reconstruction, to_adjust, camera_priors, rig_camera_priors, data.config
    )
    retriangulate(tracks_manager, reconstruction, data.config)

    if len(reconstruction.points) < min_inliers:
        report["decision"] = (
            "Re-triangulation after initial motion did not generate enough points"
        )
        logger.info(report["decision"])
        return None, report

    bundle_shot_poses(
        reconstruction, to_adjust, camera_priors, rig_camera_priors, data.config
    )

    report["decision"] = "Success"
    report["memory_usage"] = context.current_memory_usage()
    return reconstruction, report


def bootstrap_reconstruction(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    im1: str,
    im2: str,
    p1: NDArray,
    p2: NDArray,
) -> Tuple[Optional[types.Reconstruction], Dict[str, Any]]:
    """Start a reconstruction using two shots."""
    logger.info("Starting reconstruction with {} and {}".format(im1, im2))
    report: Dict[str, Any] = {
        "image_pair": (im1, im2),
        "common_tracks": len(p1),
    }

    camera_priors = data.load_camera_models()
    camera1 = camera_priors[data.load_exif(im1)["camera"]]
    camera2 = camera_priors[data.load_exif(im2)["camera"]]

    threshold = data.config["five_point_algo_threshold"]
    iterations = data.config["five_point_refine_rec_iterations"]
    check_reversal = data.config["five_point_reversal_check"]
    reversal_ratio = data.config["five_point_reversal_ratio"]

    (
        R,
        t,
        inliers,
        report["two_view_reconstruction"],
    ) = two_view_reconstruction_general(
        p1, p2, camera1, camera2, threshold, iterations, check_reversal, reversal_ratio
    )

    if R is None or t is None:
        return None, report

    rec, rec_report = reconstruction_from_relative_pose(
        data, tracks_manager, im1, im2, R, t
    )
    report.update(rec_report)

    return rec, report


def corresponding_tracks(
    tracks1: Dict[str, pymap.Observation], tracks2: Dict[str, pymap.Observation]
) -> List[Tuple[str, str]]:
    features1 = {obs.id: t1 for t1, obs in tracks1.items()}
    corresponding_tracks = []
    for t2, obs in tracks2.items():
        feature_id = obs.id
        if feature_id in features1:
            corresponding_tracks.append((features1[feature_id], t2))
    return corresponding_tracks


def compute_common_tracks(
    reconstruction1: types.Reconstruction,
    reconstruction2: types.Reconstruction,
    tracks_manager1: pymap.TracksManager,
    tracks_manager2: pymap.TracksManager,
) -> List[Tuple[str, str]]:
    common_tracks = set()
    common_images = set(reconstruction1.shots.keys()).intersection(
        reconstruction2.shots.keys()
    )

    all_shot_ids1 = set(tracks_manager1.get_shot_ids())
    all_shot_ids2 = set(tracks_manager2.get_shot_ids())
    for image in common_images:
        if image not in all_shot_ids1 or image not in all_shot_ids2:
            continue
        at_shot1 = tracks_manager1.get_shot_observations(image)
        at_shot2 = tracks_manager2.get_shot_observations(image)
        for t1, t2 in corresponding_tracks(at_shot1, at_shot2):
            if t1 in reconstruction1.points and t2 in reconstruction2.points:
                common_tracks.add((t1, t2))
    return list(common_tracks)


def resect_reconstruction(
    reconstruction1: types.Reconstruction,
    reconstruction2: types.Reconstruction,
    tracks_manager1: pymap.TracksManager,
    tracks_manager2: pymap.TracksManager,
    threshold: float,
    min_inliers: int,
) -> Tuple[bool, NDArray, List[Tuple[str, str]]]:
    """Compute a similarity transform `similarity` such as :

    reconstruction2 = T . reconstruction1

    between two reconstruction 'reconstruction1' and 'reconstruction2'.

    Their respective tracks managers are used to find common tracks that
    are further used to compute the 3D similarity transform T using RANSAC.
    """

    common_tracks = compute_common_tracks(
        reconstruction1, reconstruction2, tracks_manager1, tracks_manager2
    )
    worked, similarity, inliers = align_two_reconstruction(
        reconstruction1, reconstruction2, common_tracks, threshold
    )
    if not worked or similarity is None:
        return False, np.ones((4, 4)), []

    inliers = [common_tracks[inliers[i]] for i in range(len(inliers))]
    return True, similarity, inliers


def retriangulate(
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Retrianguate all points"""
    chrono = Chronometer()
    report = {}
    report["num_points_before"] = len(reconstruction.points)

    use_robust = config["triangulation_type"] == "ROBUST"
    reconstruction.points = {}
    pysfm.reconstruct_from_tracks_manager(
        reconstruction.map, tracks_manager, config, use_robust
    )

    report["num_points_after"] = len(reconstruction.points)
    chrono.lap("retriangulate")
    report["wall_time"] = chrono.total_time()
    return report


def remove_outliers(
    reconstruction: types.Reconstruction,
    config: Dict[str, Any],
    points: Optional[Any] = None,
) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """Remove points with large reprojection error.

    A list of point ids to be processed can be given in ``points``.
    Delegates to C++ BAHelpers.remove_outliers for efficiency and to
    avoid persistent storage of reproj errors/weights on landmarks.
    """
    if points is None:
        point_ids: List[str] = []
    elif isinstance(points, dict):
        point_ids = list(points.keys())
    else:
        # points is a list of landmark IDs (from bundle_local)
        point_ids = list(points)
    outliers, removed_tracks = pysfm.BAHelpers.remove_outliers(
        reconstruction.map, config, point_ids
    )
    logger.info("Removed outliers: {}".format(len(outliers)))
    return outliers, removed_tracks


def shot_lla_and_compass(
    shot: pymap.Shot, reference: types.TopocentricConverter
) -> Tuple[float, float, float, float]:
    """Lat, lon, alt and compass of the reconstructed shot position."""
    topo = shot.pose.get_origin()
    lat, lon, alt = reference.to_lla(*topo)

    dz = shot.pose.get_R_cam_to_world()[:, 2]
    angle = np.rad2deg(np.arctan2(dz[0], dz[1]))
    angle = (angle + 360) % 360
    return lat, lon, alt, angle


def align_two_reconstruction(
    r1: types.Reconstruction,
    r2: types.Reconstruction,
    common_tracks: List[Tuple[str, str]],
    threshold: float,
) -> Tuple[bool, Optional[NDArray], List[int]]:
    """Estimate similarity transform T between two,
    reconstructions r1 and r2 such as r2 = T . r1
    """
    t1, t2 = r1.points, r2.points

    if len(common_tracks) > 6:
        p1 = np.array([t1[t[0]].coordinates for t in common_tracks])
        p2 = np.array([t2[t[1]].coordinates for t in common_tracks])

        # 3 samples / 100 trials / 50% outliers = 0.99 probability
        # with probability = 1-(1-(1-outlier)^model)^trial
        T, inliers = multiview.fit_similarity_transform(
            p1, p2, max_iterations=100, threshold=threshold
        )
        if len(inliers) > 0:
            return True, T, list(inliers)
    return False, None, []


def merge_two_reconstructions(
    r1: types.Reconstruction,
    r2: types.Reconstruction,
    config: Dict[str, Any],
    threshold: float = 1,
) -> List[types.Reconstruction]:
    """Merge two reconstructions with common tracks IDs."""
    common_tracks = list(set(r1.points) & set(r2.points))
    worked, T, inliers = align_two_reconstruction(
        r1, r2, common_tracks, threshold)

    if T and worked and len(inliers) >= 10:
        s, A, b = multiview.decompose_similarity_transform(T)
        r1p = r1
        apply_similarity(r1p, s, A, b)
        r = r2
        r.shots.update(r1p.shots)
        r.points.update(r1p.points)
        align_reconstruction(r, [], config)
        return [r]
    else:
        return [r1, r2]


def merge_reconstructions(
    reconstructions: List[types.Reconstruction], config: Dict[str, Any]
) -> List[types.Reconstruction]:
    """Greedily merge reconstructions with common tracks."""
    num_reconstruction = len(reconstructions)
    ids_reconstructions = np.arange(num_reconstruction)
    remaining_reconstruction = ids_reconstructions
    reconstructions_merged = []
    num_merge = 0

    for i, j in combinations(ids_reconstructions, 2):
        if (i in remaining_reconstruction) and (j in remaining_reconstruction):
            r = merge_two_reconstructions(
                reconstructions[i], reconstructions[j], config
            )
            if len(r) == 1:
                remaining_reconstruction = list(
                    set(remaining_reconstruction) - {i, j})
                for k in remaining_reconstruction:
                    rr = merge_two_reconstructions(
                        r[0], reconstructions[k], config)
                    if len(r) == 2:
                        break
                    else:
                        r = rr
                        remaining_reconstruction = list(
                            set(remaining_reconstruction) - {k}
                        )
                reconstructions_merged.append(r[0])
                num_merge += 1

    for k in remaining_reconstruction:
        reconstructions_merged.append(reconstructions[k])

    logger.info("Merged {0} reconstructions".format(num_merge))

    return reconstructions_merged


def paint_reconstruction(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> None:
    """Set the color of the points from the color of the tracks."""
    for k, point in reconstruction.points.items():
        point.color = list(
            map(
                float,
                next(
                    iter(tracks_manager.get_track_observations(str(k)).values())
                ).color,
            )
        )


def grow_reconstruction(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    images: Set[str],
    gcp: List[pymap.GroundControlPoint],
) -> Tuple[types.Reconstruction, Dict[str, Any]]:
    """Incrementally add shots to an initial reconstruction."""
    config = data.config
    report = {"steps": []}

    initial_memory = context.log_memory("grow_reconstruction start")
    report["initial_memory_usage"] = initial_memory

    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()

    paint_reconstruction(data, tracks_manager, reconstruction)
    align_reconstruction(reconstruction, [], config)

    bundle(reconstruction, camera_priors, rig_camera_priors, None, 0, config)
    remove_outliers(reconstruction, config)
    paint_reconstruction(data, tracks_manager, reconstruction)

    # Build shot_id -> camera_id mapping from exif
    shot_camera_map = {}
    for image in images:
        exif = data.load_exif(image)
        shot_camera_map[image] = exif["camera"]

    # Also include rig member shots that might not be in images (no tracks)
    rig_assignments_raw = rig.rig_assignments_per_image(
        data.load_rig_assignments())
    for shot_id, (instance_id, rig_camera_id, instance_shots) in rig_assignments_raw.items():
        for member_shot in instance_shots:
            if member_shot not in shot_camera_map:
                exif = data.load_exif(member_shot)
                shot_camera_map[member_shot] = exif["camera"]

    # Build rig assignments for C++
    rig_assignments_cpp = {}
    for shot_id, (instance_id, rig_camera_id, instance_shots) in rig_assignments_raw.items():
        ra = pysfm.RigAssignment()
        ra.instance_id = instance_id
        ra.rig_camera_id = rig_camera_id
        ra.instance_shots = instance_shots
        rig_assignments_cpp[shot_id] = ra

    # Run the main grow loop in C++
    grow_report = pysfm.ReconstructionGrower.grow(
        reconstruction.map,
        tracks_manager,
        dict(camera_priors),
        dict(rig_camera_priors),
        shot_camera_map,
        rig_assignments_cpp,
        images,
        reconstruction,
        data,
        config,
    )
    report["grow"] = grow_report

    # Remove resected shots from the caller's image set
    images -= set(grow_report["resected_shots"])

    # Post-loop finalization
    final_bundle_grid = config["final_bundle_grid"]

    logger.info("-------------------------------------------------------")

    align_gcp_result = align_reconstruction(
        reconstruction, gcp, config, bias_override=True)
    if not align_gcp_result and config["bundle_compensate_gps_bias"]:
        overidden_config = config.copy()
        overidden_config["bundle_compensate_gps_bias"] = False
        config = overidden_config

    if not align_gcp_result:
        align_reconstruction(reconstruction, gcp, config)
        bundle(reconstruction, camera_priors, rig_camera_priors,
               gcp, final_bundle_grid, config)
    else:
        bundle_with_gcp_annealing(reconstruction, camera_priors, rig_camera_priors,
                                  gcp, final_bundle_grid, config)
    remove_outliers(reconstruction, config)

    if config["filter_final_point_cloud"]:
        bad_condition = pysfm.filter_badly_conditioned_points(
            reconstruction.map, config["triangulation_min_ray_angle"]
        )
        logger.info("Removed bad-condition: {}".format(bad_condition))
        isolated = pysfm.remove_isolated_points(reconstruction.map)
        logger.info("Removed isolated: {}".format(isolated))

    paint_reconstruction(data, tracks_manager, reconstruction)
    final_memory = context.log_memory("grow_reconstruction end")
    report["final_memory_usage"] = final_memory
    report["memory_delta"] = final_memory - initial_memory
    logger.info(
        f"[Memory] Total memory change during grow_reconstruction: "
        f"{(final_memory - initial_memory) / 1024:.1f} GB"
    )
    return reconstruction, report


def triangulation_reconstruction(
    data: DataSetBase, tracks_manager: pymap.TracksManager
) -> Tuple[Dict[str, Any], List[types.Reconstruction]]:
    """Run the triangulation reconstruction pipeline."""
    logger.info("Starting triangulation reconstruction")
    report = {}
    chrono = Chronometer()

    images = tracks_manager.get_shot_ids()
    data.init_reference(images)

    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()
    gcp = data.load_ground_control_points()

    reconstruction = helpers.reconstruction_from_metadata(data, images)
    reconstruction.map.set_observation_pool(
        tracks_manager.get_observation_pool())

    config = data.config
    config_override = config.copy()
    config_override["triangulation_type"] = "ROBUST"
    config_override["bundle_max_iterations"] = 50
    bundle_grid = config["local_bundle_grid"]

    # Core loop in C++
    loop_report = pysfm.ReconstructionGrower.triangulation_reconstruction(
        reconstruction.map, tracks_manager, dict(
            camera_priors), dict(rig_camera_priors),
        bundle_grid, reconstruction, config_override,
        outer_iterations=3, inner_iterations=2,
    )
    report["loop"] = loop_report

    logger.info("Triangulation SfM done.")
    chrono.lap("compute_reconstructions")
    report["wall_times"] = dict(chrono.lap_times())

    align_result = align_reconstruction(
        reconstruction, gcp, config, bias_override=True)
    if not align_result and config["bundle_compensate_gps_bias"]:
        overidden_bias_config = config.copy()
        overidden_bias_config["bundle_compensate_gps_bias"] = False
        config = overidden_bias_config

    bundle_with_gcp_annealing(reconstruction, camera_priors,
                              rig_camera_priors, gcp, bundle_grid, config)
    remove_outliers(reconstruction, config_override)

    if config["filter_final_point_cloud"]:
        bad_condition = pysfm.filter_badly_conditioned_points(
            reconstruction.map, config["triangulation_min_ray_angle"]
        )
        logger.info("Removed bad-condition: {}".format(bad_condition))
        isolated = pysfm.remove_isolated_points(reconstruction.map)
        logger.info("Removed isolated: {}".format(isolated))

    paint_reconstruction(data, tracks_manager, reconstruction)
    return report, [reconstruction]


def _get_common_feature_arrays(
    tracks_manager: pymap.TracksManager,
    im1: str,
    im2: str,
) -> Tuple[NDArray, NDArray]:
    """Return the feature arrays of common tracks between two images."""
    _, p1, p2 = tracks_manager.get_all_common_observations_arrays(im1, im2)
    return p1, p2


def compute_image_pairs_sequential(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
) -> List[Tuple[str, str]]:
    """Compute all possible pairs of images that share at least `min_common` points.
    The pairs are sorted by "reconstructability" which is estimated by the
    pairwise_reconstructability function.
    """
    cameras = data.load_camera_models()
    threshold = 4 * data.config["five_point_algo_threshold"]
    results = []
    connectivity = tracks_manager.get_all_pairs_connectivity()
    for (im1, im2), size in connectivity.items():
        if size < min_common:
            continue
        camera1 = cameras[data.load_exif(im1)["camera"]]
        camera2 = cameras[data.load_exif(im2)["camera"]]
        r = calculate_pair_reconstructability(
            tracks_manager, im1, im2, camera1, camera2, threshold
        )
        if r > 0:
            results.append((im1, im2, r))
    results.sort(key=lambda x: x[2], reverse=True)
    return [(im1, im2) for im1, im2, _ in results]


def compute_image_pairs_with_reconstructability_parallel(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
) -> List[Tuple[str, str]]:
    """Compute all possible pairs of images that share at least `min_common` points.

    This runs in parallel and fuses the feature extraction and reconstructability
    computation to save memory.
    """
    cameras = data.load_camera_models()
    threshold = 4 * data.config["five_point_algo_threshold"]

    shot_ids = tracks_manager.get_shot_ids()
    shot_to_camera_id = {}
    for shot in shot_ids:
        shot_to_camera_id[shot] = data.load_exif(shot)["camera"]

    def process_pair(pair):
        im1, im2, size = pair
        if size < min_common:
            return None

        camera1 = cameras[shot_to_camera_id[im1]]
        camera2 = cameras[shot_to_camera_id[im2]]

        r = calculate_pair_reconstructability(
            tracks_manager, im1, im2, camera1, camera2, threshold
        )
        if r > 0:
            return (im1, im2, r)
        return None

    logger.debug("Computing pairwise connectivity of images")
    all_pairs_connectivity = tracks_manager.get_all_pairs_connectivity()
    pairs = [(k[0], k[1], v) for k, v in all_pairs_connectivity.items()]

    logger.debug(
        f"Computing pairwise reconstructability with {data.config['processes']} processes")
    processes = data.config["processes"]
    batch_size = max(1, len(pairs) // (2 * processes))

    results = context.parallel_map(process_pair, pairs, processes, batch_size)

    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x[2], reverse=True)
    return [(im1, im2) for im1, im2, _ in valid_results]


def incremental_reconstruction(
    data: DataSetBase, tracks_manager: pymap.TracksManager
) -> Tuple[Dict[str, Any], List[types.Reconstruction]]:
    """Run the entire incremental reconstruction pipeline."""
    logger.info("Starting incremental reconstruction")
    report = {}
    chrono = Chronometer()

    images = tracks_manager.get_shot_ids()

    data.init_reference(images)

    remaining_images = set(images)
    gcp = data.load_ground_control_points()
    logger.info(f"Loaded {len(gcp)} ground control points.")

    common_tracks = None
    if data.config["processes"] > 1:
        # Get pairs in parallel, computing reconstructability on the fly.
        # Pros: fast, low memory usage
        pairs = compute_image_pairs_with_reconstructability_parallel(
            data, tracks_manager)
    else:
        # Get pairs sequentially, load features lazily.
        # Pros: low memory usage; Cons: slow
        pairs = compute_image_pairs_sequential(data, tracks_manager)
    logging.info(f"Estimated reconstructability of {len(pairs)} image pairs.")

    reconstructions = []
    chrono.lap("compute_image_pairs")
    report["num_candidate_image_pairs"] = len(pairs)
    report["reconstructions"] = []
    for im1, im2 in pairs:
        if im1 in remaining_images and im2 in remaining_images:
            rec_report = {}
            report["reconstructions"].append(rec_report)
            p1, p2 = _get_common_feature_arrays(tracks_manager, im1, im2)
            reconstruction, rec_report["bootstrap"] = bootstrap_reconstruction(
                data, tracks_manager, im1, im2, p1, p2
            )

            if reconstruction:
                remaining_images -= set(reconstruction.shots)
                reconstruction, rec_report["grow"] = grow_reconstruction(
                    data,
                    tracks_manager,
                    reconstruction,
                    remaining_images,
                    gcp,
                )
                reconstructions.append(reconstruction)
                reconstructions = sorted(
                    reconstructions, key=lambda x: -len(x.shots))

            if data.config["incremental_max_shots_count"] > 0 and sum(
                len(r.shots) for r in reconstructions
            ) >= data.config["incremental_max_shots_count"]:
                logger.info(
                    f"Reached the maximum number of shots")
                break

    for k, r in enumerate(reconstructions):
        logger.info(
            "Reconstruction {}: {} images, {} points".format(
                k, len(r.shots), len(r.points)
            )
        )
    logger.info("{} partial reconstructions in total.".format(
        len(reconstructions)))
    chrono.lap("compute_reconstructions")
    report["wall_times"] = dict(chrono.lap_times())
    report["not_reconstructed_images"] = list(remaining_images)
    return report, reconstructions


def reconstruct_from_prior(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    rec_prior: types.Reconstruction,
) -> Tuple[Dict[str, Any], types.Reconstruction]:
    """Retriangulate a new reconstruction from the rec_prior"""
    reconstruction = types.Reconstruction()
    reconstruction.reference = rec_prior.reference
    report = {}
    rec_report = {}
    report["retriangulate"] = [rec_report]
    images = tracks_manager.get_shot_ids()

    # copy prior poses, cameras
    reconstruction.cameras = rec_prior.cameras
    for shot in rec_prior.shots.values():
        reconstruction.add_shot(shot)
    prior_images = set(rec_prior.shots)
    remaining_images = set(images) - prior_images
    reconstruction.map.set_observation_pool(
        tracks_manager.get_observation_pool())

    rec_report["num_prior_images"] = len(prior_images)
    rec_report["num_remaining_images"] = len(remaining_images)

    # Start with the known poses
    retriangulate(tracks_manager, reconstruction, data.config)
    paint_reconstruction(data, tracks_manager, reconstruction)
    report["not_reconstructed_images"] = list(remaining_images)
    return report, reconstruction


class Chronometer:
    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        t = timer()
        lap = ("start", 0, t)
        self.laps = [lap]
        self.laps_dict = {"start": lap}

    def lap(self, key: str) -> None:
        t = timer()
        dt = t - self.laps[-1][2]
        lap = (key, dt, t)
        self.laps.append(lap)
        self.laps_dict[key] = lap

    def lap_time(self, key: str) -> float:
        return self.laps_dict[key][1]

    def lap_times(self) -> List[Tuple[str, float]]:
        return [(k, dt) for k, dt, t in self.laps[1:]]

    def total_time(self) -> float:
        return self.laps[-1][2] - self.laps[0][2]
