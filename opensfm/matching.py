# pyre-strict
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from timeit import default_timer as timer
from typing import Any, Dict, Generator, List, Optional, Sized, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from opensfm import (
    context,
    feature_loader,
    log,
    multiview,
    pairs_selection,
    pyfeatures,
    pygeometry,
)
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)


def clear_cache() -> None:
    feature_loader.instance.clear_cache()


def match_images(
    data: DataSetBase,
    config_override: Dict[str, Any],
    ref_images: List[str],
    cand_images: List[str],
) -> Tuple[Dict[Tuple[str, str], List[Tuple[int, int]]], Dict[str, Any]]:
    """Perform pair matchings between two sets of images.

    It will do matching for each pair (i, j), i being in
    ref_images and j in cand_images, taking assumption that
    matching(i, j) == matching(j ,i). This does not hold for
    non-symmetric matching options like WORDS. Data will be
    stored in i matching only.
    """

    # Get EXIFs data
    all_images = list(set(ref_images + cand_images))
    exifs = {im: data.load_exif(im) for im in all_images}

    # Generate pairs for matching
    pairs, preport = pairs_selection.match_candidates_from_metadata(
        ref_images,
        cand_images,
        exifs,
        data,
        config_override,
    )

    # Match them !
    return (
        match_images_with_pairs(data, config_override, exifs, pairs),
        preport,
    )


def match_pairs_with_binary_projection_batched(
        data: DataSetBase,
        pairs: List[Tuple[str, str]],
        config: Dict[str, Any],
        cameras: Dict[str, Any],
        exifs: Dict[str, Any]) -> List[NDArray]:
    """
    Perform pair matchings using learned binary projection and batched GPU matching.

    Args:
    data: dataset to load features from
    pairs: list of image pairs to match
    config: config parameters, must contain "lowes_ratio" and binary matching parameters
    cameras: camera models for the images
    exifs: exif data for the images
    """
    binary_cache = generate_binary_cache(
        data, pairs, config, cameras, exifs)
    logger.info("Binarized descriptors for %d images", len(binary_cache))

    bin1_list = []
    bin2_list = []
    total_features = 0
    for im1, im2 in pairs:
        b1 = binary_cache.get(im1, np.zeros((0, 4), dtype=np.uint32))
        b2 = binary_cache.get(im2, np.zeros((0, 4), dtype=np.uint32))
        bin1_list.append(b1)
        bin2_list.append(b2)
        total_features += len(b1) + len(b2)
    logger.info(
        "Batched Hamming matching %d pairs (%d total features) on GPU",
        len(pairs), total_features)
    time_start = time.time()
    batch_results = pyfeatures.match_hamming_opencl_batch_symmetric(
        bin1_list, bin2_list, config["lowes_ratio"], 0)
    logger.info(
        "Batched Hamming matching completed in %.2f seconds", time.time() - time_start
    )

    return batch_results


def match_images_with_pairs(
    data: DataSetBase,
    config_override: Dict[str, Any],
    exifs: Dict[str, Any],
    pairs: List[Tuple[str, str]],
    poses: Optional[Dict[str, pygeometry.Pose]] = None,
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """Perform pair matchings given pairs."""
    cameras = data.load_camera_models()

    # Perform all pair matchings in parallel
    start = timer()
    logger.info("Matching {} image pairs".format(len(pairs)))
    processes = config_override.get("processes", data.config["processes"])
    mem_per_process = 512
    jobs_per_process = 2
    processes = context.processes_that_fit_in_memory(
        processes, mem_per_process)
    logger.info("Using {} processes for matching".format(processes))

    # Fallback to FLANN is GPU matching is requested but not available.
    overriden_config = data.config.copy()
    overriden_config.update(config_override)
    required_gpu = overriden_config["matcher_type"].upper() in [
        "OPENCL_HAMMING", "OPENCL_BF"]
    required_opencl_hamming = overriden_config["matcher_type"].upper() in [
        "OPENCL_HAMMING"]

    use_gpu = required_gpu
    if required_gpu and not pyfeatures.opencl_matching_available():
        logger.warning(
            "GPU matching requested but OpenCL is not available. Falling back to CPU FLANN."
        )
        overriden_config["matcher_type"] = "FLANN"
        use_gpu = False
    if use_gpu:
        num_devices = pyfeatures.opencl_num_devices()
        logger.info(
            "Computing pair matching on GPU (OpenCL, %d device(s))" % num_devices
        )
        overriden_config["_opencl_device_idx"] = 0

    if use_gpu and required_opencl_hamming:
        batch_results = match_pairs_with_binary_projection_batched(
            data, pairs, overriden_config, cameras, exifs
        )
        args = list(match_arguments(
            pairs, data, overriden_config, cameras, exifs, poses, batch_results))
        use_gpu = False  # do robust matching on CPU, as GPU matching already done
    else:
        # Build per-pair arguments
        args = list(match_arguments(
            pairs, data, overriden_config, cameras, exifs, poses, None))

    if use_gpu:
        logger.info("Computing pair matching with GPU 0")
        matches = [match_unwrap_args(a) for a in args]
    else:
        logger.info("Computing pair matching with %d processes" % processes)
        matches = context.parallel_map(
            match_unwrap_args, args, processes, jobs_per_process)
    logger.info(
        "Matched {} pairs {} in {} seconds ({} seconds/pair).".format(
            len(pairs),
            log_projection_types(pairs, exifs, cameras),
            timer() - start,
            (timer() - start) / len(pairs) if pairs else 0,
        )
    )

    # Index results per pair
    resulting_pairs = {}
    for im1, im2, m in matches:
        resulting_pairs[im1, im2] = m
    return resulting_pairs


def log_projection_types(
    pairs: List[Tuple[str, str]],
    exifs: Dict[str, Any],
    cameras: Dict[str, pygeometry.Camera],
) -> str:
    if not pairs:
        return ""

    projection_type_pairs = {}
    for im1, im2 in pairs:
        pt1 = cameras[exifs[im1]["camera"]].projection_type
        pt2 = cameras[exifs[im2]["camera"]].projection_type

        if pt1 not in projection_type_pairs:
            projection_type_pairs[pt1] = {}

        if pt2 not in projection_type_pairs[pt1]:
            projection_type_pairs[pt1][pt2] = []

        projection_type_pairs[pt1][pt2].append((im1, im2))

    output = "("
    for pt1 in projection_type_pairs:
        for pt2 in projection_type_pairs[pt1]:
            output += "{}-{}: {}, ".format(
                pt1, pt2, len(projection_type_pairs[pt1][pt2])
            )

    return output[:-2] + ")"


def save_matches(
    data: DataSetBase,
    images_ref: List[str],
    matched_pairs: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> None:
    """Given pairwise matches (image 1, image 2) - > matches,
    save them such as only {image E images_ref} will store the matches.
    """
    images_ref_set = set(images_ref)
    matches_per_im1 = {im: {} for im in images_ref}
    for (im1, im2), m in matched_pairs.items():
        if im1 in images_ref_set:
            matches_per_im1[im1][im2] = m
        elif im2 in images_ref_set:
            matches_per_im1[im2][im1] = m
        else:
            raise RuntimeError(
                "Couldn't save matches for {}. No image found in images_ref.".format(
                    (im1, im2)
                )
            )

    for im1, im1_matches in matches_per_im1.items():
        data.save_matches(im1, im1_matches)


def match_arguments(
    pairs: List[Tuple[str, str]],
    data: DataSetBase,
    config_override: Dict[str, Any],
    cameras: Dict[str, pygeometry.Camera],
    exifs: Dict[str, pygeometry.Camera],
    poses: Optional[Dict[str, pygeometry.Pose]],
    desc_matches: Optional[List[NDArray]],
) -> Generator[
    Tuple[
        str,
        str,
        Dict[str, pygeometry.Camera],
        Dict[str, pygeometry.Camera],
        DataSetBase,
        Dict[str, Any],
        Optional[Dict[str, pygeometry.Pose]],
        Optional[NDArray]],
    None,
    None,
]:
    """Generate arguments for parallel processing of pair matching"""
    for i, (im1, im2) in enumerate(pairs):
        matches = desc_matches[i] if desc_matches else None
        yield im1, im2, cameras, exifs, data, config_override, poses, matches


def match_unwrap_args(
    args: Tuple[
        str,
        str,
        Dict[str, pygeometry.Camera],
        Dict[str, Any],
        DataSetBase,
        Dict[str, Any],
        Optional[Dict[str, pygeometry.Pose]],
        Optional[NDArray],
    ],
) -> Tuple[str, str, NDArray]:
    """Wrapper for parallel processing of pair matching.

    Compute all pair matchings of a given image and save them.
    """
    log.setup()
    im1 = args[0]
    im2 = args[1]
    cameras = args[2]
    exifs = args[3]
    data: DataSetBase = args[4]
    config_override = args[5]
    poses = args[6]
    desc_matches = args[7]
    if poses:
        pose1 = poses[im1]
        pose2 = poses[im2]
        pose = pose2.relative_to(pose1)
    else:
        pose = None

    camera1 = cameras[exifs[im1]["camera"]]
    camera2 = cameras[exifs[im2]["camera"]]

    if desc_matches is not None:
        matches = match_robust(im1, im2, desc_matches,
                               camera1, camera2, data, config_override, False)
    else:
        matches = match(im1, im2, camera1, camera2,
                        data, config_override, pose)
    return im1, im2, matches


def match_descriptors(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    config_override: Dict[str, Any],
) -> NDArray:
    """Perform descriptor matching for a pair of images."""
    # Override parameters
    overriden_config = data.config.copy()
    overriden_config.update(config_override)

    # Run descriptor matching
    time_start = timer()
    _, _, matches, matcher_type = _match_descriptors_impl(
        im1, im2, camera1, camera2, data, overriden_config
    )
    time_2d_matching = timer() - time_start

    # From indexes in filtered sets, to indexes in original sets of features
    matches_unfiltered = []
    m1 = feature_loader.instance.load_mask(data, im1)
    m2 = feature_loader.instance.load_mask(data, im2)
    if m1 is not None and m2 is not None:
        matches_unfiltered = unfilter_matches(matches, m1, m2)

    symmetric = "symmetric" if overriden_config["symmetric_matching"] else "one-way"
    logger.debug(
        "Matching {} and {}.  Matcher: {} ({}) " "T-desc: {:1.3f} Matches: {}".format(
            im1,
            im2,
            matcher_type,
            symmetric,
            time_2d_matching,
            len(matches_unfiltered),
        )
    )
    return np.array(matches_unfiltered, dtype=int)


def _match_descriptors_guided_impl(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    relative_pose: pygeometry.Pose,
    data: DataSetBase,
    overriden_config: Dict[str, Any],
) -> Tuple[NDArray, NDArray, NDArray, str]:
    """Perform descriptor guided matching for a pair of images, using their relative pose. It also apply static objects removal."""
    guided_matcher_override = "BRUTEFORCE"
    matcher_type = overriden_config["matcher_type"].upper()
    symmetric_matching = overriden_config["symmetric_matching"]
    if matcher_type in ["WORDS", "FLANN"] or symmetric_matching:
        logger.warning(
            f"{matcher_type} and/or symmetric isn't supported for guided matching, switching to asymmetric {guided_matcher_override}"
        )
        matcher_type = guided_matcher_override

    # Will apply mask to features if any
    dummy = np.array([])
    segmentation_in_descriptor = overriden_config["matching_use_segmentation"]
    features_data1 = feature_loader.instance.load_all_data(
        data,
        im1,
        masked=True,
        segmentation_in_descriptor=segmentation_in_descriptor,
    )
    features_data2 = feature_loader.instance.load_all_data(
        data, im2, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    bearings1 = feature_loader.instance.load_bearings(
        data, im1, masked=True, camera=camera1
    )
    bearings2 = feature_loader.instance.load_bearings(
        data, im2, masked=True, camera=camera2
    )

    if (
        features_data1 is None
        or bearings1 is None
        or len(features_data1.points) < 2
        or features_data2 is None
        or bearings2 is None
        or len(features_data2.points) < 2
    ):
        return dummy, dummy, dummy, matcher_type

    d1 = features_data1.descriptors
    d2 = features_data2.descriptors
    if d1 is None or d2 is None:
        return dummy, dummy, dummy, matcher_type

    epipolar_mask = compute_inliers_bearing_epipolar(
        bearings1,
        bearings2,
        relative_pose,
        overriden_config["guided_matching_threshold"],
    )
    matches = match_brute_force_symmetric(
        d1, d2, overriden_config, epipolar_mask)

    # Adhoc filters
    if overriden_config["matching_use_filters"]:
        matches = apply_adhoc_filters(
            data,
            matches,
            im1,
            camera1,
            features_data1.points,
            im2,
            camera2,
            features_data2.points,
        )
    return (
        features_data1.points,
        features_data2.points,
        np.array(matches, dtype=int),
        matcher_type,
    )


def _match_descriptors_impl(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    overriden_config: Dict[str, Any],
) -> Tuple[NDArray, NDArray, NDArray, str]:
    """Perform descriptor matching for a pair of images. It also apply static objects removal."""
    dummy = np.array([])
    matcher_type = overriden_config["matcher_type"].upper()
    dummy_ret = dummy, dummy, dummy, matcher_type

    # Will apply mask to features if any
    dummy = np.array([])
    segmentation_in_descriptor = overriden_config["matching_use_segmentation"]
    features_data1 = feature_loader.instance.load_all_data(
        data, im1, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    features_data2 = feature_loader.instance.load_all_data(
        data, im2, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    if (
        features_data1 is None
        or len(features_data1.points) < 2
        or features_data2 is None
        or len(features_data2.points) < 2
    ):
        return dummy_ret

    d1 = features_data1.descriptors
    d2 = features_data2.descriptors
    if d1 is None or d2 is None:
        return dummy_ret

    symmetric_matching = overriden_config["symmetric_matching"]
    if matcher_type == "OPENCL_BF":
        if symmetric_matching:
            matches = match_brute_force_symmetric(d1, d2, overriden_config)
        else:
            matches = match_brute_force(d1, d2, overriden_config)
    elif matcher_type == "WORDS":
        words1 = feature_loader.instance.load_words(data, im1, masked=True)
        words2 = feature_loader.instance.load_words(data, im2, masked=True)
        if words1 is None or words2 is None:
            return dummy_ret

        if symmetric_matching:
            matches = match_words_symmetric(
                d1,
                words1,
                d2,
                words2,
                overriden_config,
            )
        else:
            matches = match_words(
                d1,
                words1,
                d2,
                words2,
                overriden_config,
            )

    elif matcher_type == "FLANN":
        f1 = feature_loader.instance.load_features_index(
            data,
            im1,
            masked=True,
            segmentation_in_descriptor=segmentation_in_descriptor,
        )
        if not f1:
            return dummy_ret
        feat_data_index1, index1 = f1
        if symmetric_matching:
            f2 = feature_loader.instance.load_features_index(
                data,
                im2,
                masked=True,
                segmentation_in_descriptor=segmentation_in_descriptor,
            )
            if not f2:
                return dummy_ret
            feat_data_index2, index2 = f2

            descriptors1 = feat_data_index1.descriptors
            descriptors2 = feat_data_index2.descriptors
            if descriptors1 is None or descriptors2 is None:
                return dummy_ret

            matches = match_flann_symmetric(
                descriptors1,
                index1,
                descriptors2,
                index2,
                overriden_config,
            )
        else:
            matches = match_flann(index1, d2, overriden_config)
    elif matcher_type == "BRUTEFORCE":
        if symmetric_matching:
            matches = match_brute_force_symmetric(d1, d2, overriden_config)
        else:
            matches = match_brute_force(d1, d2, overriden_config)
    else:
        raise ValueError("Invalid matcher_type: {}".format(matcher_type))

    # Adhoc filters
    if overriden_config["matching_use_filters"]:
        matches = apply_adhoc_filters(
            data,
            list(matches),
            im1,
            camera1,
            features_data1.points,
            im2,
            camera2,
            features_data2.points,
        )
    return (
        features_data1.points,
        features_data2.points,
        np.array(matches, dtype=int),
        matcher_type,
    )


def match_robust(
    im1: str,
    im2: str,
    matches: Sized,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    config_override: Dict[str, Any],
    input_is_masked: bool = True,
) -> NDArray:
    """Perform robust geometry matching on a set of matched descriptors indexes."""
    # Override parameters
    overriden_config = data.config.copy()
    overriden_config.update(config_override)

    # Will apply mask to features if any
    segmentation_in_descriptor = overriden_config[
        "matching_use_segmentation"
    ]  # unused but keep using the same cache
    features_data1 = feature_loader.instance.load_all_data(
        data,
        im1,
        masked=input_is_masked,
        segmentation_in_descriptor=segmentation_in_descriptor,
    )
    features_data2 = feature_loader.instance.load_all_data(
        data,
        im2,
        masked=input_is_masked,
        segmentation_in_descriptor=segmentation_in_descriptor,
    )
    if (
        features_data1 is None
        or len(features_data1.points) < 2
        or features_data2 is None
        or len(features_data2.points) < 2
    ):
        return np.array([])

    # Run robust matching
    np_matches = np.array(matches, dtype=int)
    t = timer()
    rmatches = _match_robust_impl(
        features_data1.points,
        features_data2.points,
        np_matches,
        camera1,
        camera2,
        overriden_config,
    )
    time_robust_matching = timer() - t

    # From indexes in filtered sets, to indexes in original sets of features
    rmatches_unfiltered = []
    m1 = feature_loader.instance.load_mask(data, im1)
    m2 = feature_loader.instance.load_mask(data, im2)
    if m1 is not None and m2 is not None and input_is_masked:
        rmatches_unfiltered = unfilter_matches(rmatches, m1, m2)
    else:
        rmatches_unfiltered = rmatches

    robust_matching_min_match = overriden_config["robust_matching_min_match"]
    logger.debug(
        "Matching {} and {}. T-robust: {:1.3f} "
        "Matches: {} Robust: {} Success: {}".format(
            im1,
            im2,
            time_robust_matching,
            len(matches),
            len(rmatches_unfiltered),
            len(rmatches_unfiltered) >= robust_matching_min_match,
        )
    )

    if len(rmatches_unfiltered) < robust_matching_min_match:
        return np.array([])
    return np.array(rmatches_unfiltered, dtype=int)


def _match_robust_impl(
    p1: NDArray,
    p2: NDArray,
    matches: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    overriden_config: Dict[str, Any],
) -> NDArray:
    """Perform robust geometry matching on a set of matched descriptors indexes."""
    # robust matching
    rmatches = robust_match(p1, p2, camera1, camera2,
                            matches, overriden_config)
    rmatches = np.array([[a, b] for a, b in rmatches])
    return rmatches


def match(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    config_override: Dict[str, Any],
    guided_matching_pose: Optional[pygeometry.Pose],
) -> NDArray:
    """Perform full matching (descriptor+robust, optionally guided) for a pair of images."""
    # Override parameters
    overriden_config = data.config.copy()
    overriden_config.update(config_override)

    # Run descriptor matching
    time_start = timer()
    if guided_matching_pose:
        p1, p2, matches, matcher_type = _match_descriptors_guided_impl(
            im1, im2, camera1, camera2, guided_matching_pose, data, overriden_config
        )
    else:
        p1, p2, matches, matcher_type = _match_descriptors_impl(
            im1, im2, camera1, camera2, data, overriden_config
        )
    time_2d_matching = timer() - time_start

    symmetric = "symmetric" if overriden_config["symmetric_matching"] else "one-way"
    robust_matching_min_match = overriden_config["robust_matching_min_match"]
    if len(matches) < robust_matching_min_match:
        logger.debug(
            "Matching {} and {}.  Matcher: {} ({}) T-desc: {:1.3f} "
            "Matches: FAILED".format(
                im1, im2, matcher_type, symmetric, time_2d_matching
            )
        )
        return np.array([])

    # Run robust matching (non guided case only)
    t = timer()
    matches_arr = np.array(matches, dtype=int).reshape(-1, 2)
    rmatches = _match_robust_impl(
        p1, p2, matches_arr, camera1, camera2, overriden_config
    )
    time_robust_matching = timer() - t

    # From indexes in filtered sets, to indexes in original sets of features
    m1 = feature_loader.instance.load_mask(data, im1)
    m2 = feature_loader.instance.load_mask(data, im2)
    if m1 is not None and m2 is not None:
        rmatches = unfilter_matches(rmatches, m1, m2)

    time_total = timer() - time_start

    logger.debug(
        "Matching {} and {}.  Matcher: {} ({}) "
        "T-desc: {:1.3f} T-robust: {:1.3f} T-total: {:1.3f} "
        "Matches: {} Robust: {} Success: {}".format(
            im1,
            im2,
            matcher_type,
            symmetric,
            time_2d_matching,
            time_robust_matching,
            time_total,
            len(matches),
            len(rmatches),
            len(rmatches) >= robust_matching_min_match,
        )
    )

    if len(rmatches) < robust_matching_min_match:
        return np.array([])
    return np.array(rmatches, dtype=int)


def match_opencl_bruteforce(f1: NDArray, f2: NDArray, config: Dict[str, Any]):
    """
    Match real-valued descriptors using brute force on OpenCL GPU.

    Args:
    f1: real-valued descriptors of the first image, shape (N1, K), dtype float32
    f2: real-valued descriptors of the second image, shape (N2, K), dtype float32
    N1 and N2 must be multiples of 4, and K must be a multiple of 4.
    config: config parameters, must contain "lowes_ratio"
    """
    ratio = config["lowes_ratio"]
    device_index = config["_opencl_device_idx"]
    matches = pyfeatures.match_brute_force_opencl(
        f1, f2, ratio, device_index
    )
    return [(int(m[0]), int(m[1])) for m in matches]


def match_opencl_bruteforce_symmetric(f1: NDArray, f2: NDArray, config: Dict[str, Any]):
    """
    Match real-valued descriptors using symmetric brute force on OpenCL GPU.

    Args:
    f1: real-valued descriptors of the first image, shape (N1, K), dtype float32
    f2: real-valued descriptors of the second image, shape (N2, K), dtype float32
    N1 and N2 must be multiples of 4, and K must be a multiple of 4.
    config: config parameters, must contain "lowes_ratio"
    """
    ratio = config["lowes_ratio"]
    device_index = config["_opencl_device_idx"]
    matches = pyfeatures.match_brute_force_opencl_symmetric(
        f1, f2, ratio, device_index
    )
    return [(int(m[0]), int(m[1])) for m in matches]


def match_words(
    f1: NDArray,
    words1: NDArray,
    f2: NDArray,
    words2: NDArray,
    config: Dict[str, Any],
) -> NDArray:
    """Match using words and apply Lowe's ratio filter.

    Args:
        f1: feature descriptors of the first image
        w1: the nth closest words for each feature in the first image
        f2: feature descriptors of the second image
        w2: the nth closest words for each feature in the second image
        config: config parameters
    """
    ratio = config["lowes_ratio"]
    num_checks = config["bow_num_checks"]
    return pyfeatures.match_using_words(f1, words1, f2, words2[:, 0], ratio, num_checks)


def match_words_symmetric(
    f1: NDArray,
    words1: NDArray,
    f2: NDArray,
    words2: NDArray,
    config: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """Match using words in both directions and keep consistent matches.

    Args:
        f1: feature descriptors of the first image
        w1: the nth closest words for each feature in the first image
        f2: feature descriptors of the second image
        w2: the nth closest words for each feature in the second image
        config: config parameters
    """
    matches_ij = match_words(f1, words1, f2, words2, config)
    matches_ji = match_words(f2, words2, f1, words1, config)
    matches_ij = [(a, b) for a, b in matches_ij]
    matches_ji = [(b, a) for a, b in matches_ji]

    return list(set(matches_ij).intersection(set(matches_ji)))


def match_flann(
    index: cv2.flann_Index, f2: NDArray, config: Dict[str, Any]
) -> List[Tuple[int, int]]:
    """Match using FLANN and apply Lowe's ratio filter.

    Args:
        index: flann index if the first image
        f2: feature descriptors of the second image
        config: config parameters
    """
    search_params = dict(checks=config["flann_checks"])
    results, dists = index.knnSearch(
        f2, 2, params=search_params)  # pyre-ignore[16]
    # Flann returns squared L2 distances
    squared_ratio = config["lowes_ratio"] ** 2
    good = dists[:, 0] < squared_ratio * dists[:, 1]
    return list(zip(results[good, 0], good.nonzero()[0]))


def match_flann_symmetric(
    fi: NDArray,
    indexi: cv2.flann_Index,
    fj: NDArray,
    indexj: cv2.flann_Index,
    config: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """Match using FLANN in both directions and keep consistent matches.

    Args:
        fi: feature descriptors of the first image
        indexi: flann index if the first image
        fj: feature descriptors of the second image
        indexj: flann index of the second image
        config: config parameters
        maskij: optional boolean mask of len(i descriptors) x len(j descriptors)
    """
    matches_ij = [(a, b) for a, b in match_flann(indexi, fj, config)]
    matches_ji = [(b, a) for a, b in match_flann(indexj, fi, config)]

    return list(set(matches_ij).intersection(set(matches_ji)))


def match_brute_force(
    f1: NDArray,
    f2: NDArray,
    config: Dict[str, Any],
    maskij: Optional[NDArray] = None,
) -> List[Tuple[int, int]]:
    """Brute force matching and Lowe's ratio filtering.

    Args:
        f1: feature descriptors of the first image
        f2: feature descriptors of the second image
        config: config parameters
        maskij: optional boolean mask of len(i descriptors) x len(j descriptors)
    """
    assert f1.dtype.type == f2.dtype.type
    if f1.dtype.type == np.uint8:
        matcher_type = "BruteForce-Hamming"
    else:
        matcher_type = "BruteForce"
    matcher = cv2.DescriptorMatcher_create(matcher_type)
    matcher.add([f2])
    if maskij is not None:
        matches = matcher.knnMatch(
            f1, k=2, masks=np.array([maskij]).astype(np.uint8))
    else:
        matches = matcher.knnMatch(f1, k=2)

    ratio = config["lowes_ratio"]
    good_matches = []
    for match in matches:
        if match and len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return [(mm.queryIdx, mm.trainIdx) for mm in good_matches]


def match_brute_force_symmetric(
    fi: NDArray,
    fj: NDArray,
    config: Dict[str, Any],
    maskij: Optional[NDArray] = None,
) -> List[Tuple[int, int]]:
    """Match with brute force in both directions and keep consistent matches.

    Args:
        fi: feature descriptors of the first image
        fj: feature descriptors of the second image
        config: config parameters
        maskij: optional boolean mask of len(i descriptors) x len(j descriptors)
    """
    matches_ij = [(a, b) for a, b in match_brute_force(fi, fj, config, maskij)]
    maskijT = maskij.T if maskij is not None else None
    matches_ji = [(b, a)
                  for a, b in match_brute_force(fj, fi, config, maskijT)]

    return list(set(matches_ij).intersection(set(matches_ji)))


def robust_match_fundamental(
    p1: NDArray,
    p2: NDArray,
    matches: NDArray,
    config: Dict[str, Any],
) -> Tuple[NDArray, NDArray]:
    """Filter matches by estimating the Fundamental matrix via RANSAC."""
    if len(matches) < 8:
        return np.array([]), np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()

    FM_RANSAC = cv2.FM_RANSAC if context.OPENCV3 else cv2.cv.CV_FM_RANSAC
    threshold = config["robust_matching_threshold"]
    F, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, threshold, 0.9999)
    inliers = mask.ravel().nonzero()

    if F is None or F[2, 2] == 0.0:
        return F, np.array([])

    return F, matches[inliers]


def compute_inliers_bearings(
    b1: NDArray,
    b2: NDArray,
    R: NDArray,
    t: NDArray,
    threshold: float = 0.01,
) -> List[bool]:
    """Compute points that can be triangulated.

    Args:
        b1, b2: Bearings in the two images.
        R, t: Rotation and translation from the second image to the first.
              That is the convention and the opposite of many
              functions in this module.
        threshold: max reprojection error in radians.
    Returns:
        array: Array of boolean indicating inliers/outliers
    """
    p = pygeometry.triangulate_two_bearings_midpoint_many(b1, b2, R, t)

    good_idx = [i for i in range(len(p)) if p[i][0]]
    points = np.array([p[i][1] for i in range(len(p)) if p[i][0]])

    inliers = [False] * len(b1)
    if len(points) < 1:
        return inliers

    br1 = points.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]
    br2 = R.T.dot((points - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    ok1 = np.linalg.norm(br1 - b1[good_idx], axis=1) < threshold
    ok2 = np.linalg.norm(br2 - b2[good_idx], axis=1) < threshold
    is_ok = ok1 * ok2

    for i, ok in enumerate(is_ok):
        inliers[good_idx[i]] = ok
    return inliers


def compute_inliers_bearing_epipolar(
    b1: NDArray, b2: NDArray, pose: pygeometry.Pose, threshold: float
) -> NDArray:
    """Compute mask of epipolarly consistent bearings, given two lists of bearings

    Args:
        b1, b2: Bearings in the two images. Expected to be normalized.
        pose: Pose of the second image wrt. the first one (relative pose)
        threshold: max reprojection error in radians.
    Returns:
        array: Matrix of boolean indicating inliers/outliers
    """
    symmetric_angle_error = pygeometry.epipolar_angle_two_bearings_many(
        b1.astype(np.float32),
        b2.astype(np.float32),
        pose.get_R_cam_to_world(),
        pose.get_origin(),
    )
    mask = symmetric_angle_error < threshold
    return mask


def robust_match_calibrated(
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    matches: NDArray,
    config: Dict[str, Any],
) -> NDArray:
    """Filter matches by estimating the Essential matrix via RANSAC."""

    if len(matches) < 8:
        return np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    threshold = config["robust_matching_calib_threshold"]
    T = multiview.relative_pose_ransac(b1, b2, threshold, 1000, 0.999)

    for relax in [4, 2, 1]:
        inliers = compute_inliers_bearings(
            b1, b2, T[:, :3], T[:, 3], relax * threshold)
        if np.sum(inliers) < 8:
            return np.array([])
        iterations = config["five_point_refine_match_iterations"]
        T = multiview.relative_pose_optimize_nonlinear(
            b1[inliers], b2[inliers], T[:3, 3], T[:3, :3], iterations
        )

    inliers = compute_inliers_bearings(b1, b2, T[:, :3], T[:, 3], threshold)

    return matches[inliers]


def robust_match(
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    matches: NDArray,
    config: Dict[str, Any],
) -> NDArray:
    """Filter matches by fitting a geometric model.

    If cameras are perspective without distortion, then the Fundamental
    matrix is used.  Otherwise, we use the Essential matrix.
    """
    if (
        camera1.projection_type in ["perspective", "brown"]
        and camera1.k1 == 0.0
        and camera1.k2 == 0.0
        and camera2.projection_type in ["perspective", "brown"]
        and camera2.k1 == 0.0
        and camera2.k2 == 0.0
    ):
        return robust_match_fundamental(p1, p2, matches, config)[1]
    else:
        return robust_match_calibrated(p1, p2, camera1, camera2, matches, config)


def unfilter_matches(matches: NDArray, m1: NDArray, m2: NDArray) -> NDArray:
    """Given matches and masking arrays, get matches with un-masked indexes."""
    i1 = np.flatnonzero(m1)
    i2 = np.flatnonzero(m2)
    return np.array([(i1[match[0]], i2[match[1]]) for match in matches])


def apply_adhoc_filters(
    data: DataSetBase,
    matches: List[Tuple[int, int]],
    im1: str,
    camera1: pygeometry.Camera,
    p1: NDArray,
    im2: str,
    camera2: pygeometry.Camera,
    p2: NDArray,
) -> List[Tuple[int, int]]:
    """Apply a set of filters functions defined further below
    for removing static data in images.

    """
    matches = _non_static_matches(p1, p2, matches)
    matches = _not_on_pano_poles_matches(p1, p2, matches, camera1, camera2)
    matches = _not_on_vermont_watermark(p1, p2, matches, im1, im2, data)
    matches = _not_on_blackvue_watermark(p1, p2, matches, im1, im2, data)
    return matches


def _non_static_matches(
    p1: NDArray, p2: NDArray, matches: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Remove matches with same position in both images.

    That should remove matches on that are likely belong to rig occluders,
    watermarks or dust, but not discard entirely static images.
    """
    threshold = 0.001
    res = []
    for match in matches:
        d = p1[match[0]] - p2[match[1]]
        if d[0] ** 2 + d[1] ** 2 >= threshold**2:
            res.append(match)

    static_ratio_threshold = 0.85
    static_ratio_removed = 1 - len(res) / max(len(matches), 1)
    if static_ratio_removed > static_ratio_threshold:
        return matches
    else:
        return res


def _not_on_pano_poles_matches(
    p1: NDArray,
    p2: NDArray,
    matches: List[Tuple[int, int]],
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
) -> List[Tuple[int, int]]:
    """Remove matches for features that are too high or to low on a pano.

    That should remove matches on the sky and and carhood part of panoramas
    """
    min_lat = -0.125
    max_lat = 0.125
    is_pano1 = pygeometry.Camera.is_panorama(camera1.projection_type)
    is_pano2 = pygeometry.Camera.is_panorama(camera2.projection_type)
    if is_pano1 or is_pano2:
        res = []
        for match in matches:
            if (not is_pano1 or min_lat < p1[match[0]][1] < max_lat) and (
                not is_pano2 or min_lat < p2[match[1]][1] < max_lat
            ):
                res.append(match)
        return res
    else:
        return matches


def _not_on_vermont_watermark(
    p1: NDArray,
    p2: NDArray,
    matches: List[Tuple[int, int]],
    im1: str,
    im2: str,
    data: DataSetBase,
) -> List[Tuple[int, int]]:
    """Filter Vermont images watermark."""
    meta1 = data.load_exif(im1)
    meta2 = data.load_exif(im2)

    if meta1["make"] == "VTrans_Camera" and meta1["model"] == "VTrans_Camera":
        matches = [m for m in matches if _vermont_valid_mask(p1[m[0]])]
    if meta2["make"] == "VTrans_Camera" and meta2["model"] == "VTrans_Camera":
        matches = [m for m in matches if _vermont_valid_mask(p2[m[1]])]
    return matches


def _vermont_valid_mask(p: NDArray) -> bool:
    """Check if pixel inside the valid region.

    Pixel coord Y should be larger than 50.
    In normalized coordinates y > (50 - h / 2) / w
    """
    return p[1] > -0.255


def _not_on_blackvue_watermark(
    p1: NDArray,
    p2: NDArray,
    matches: List[Tuple[int, int]],
    im1: str,
    im2: str,
    data: DataSetBase,
) -> List[Tuple[int, int]]:
    """Filter Blackvue's watermark."""
    meta1 = data.load_exif(im1)
    meta2 = data.load_exif(im2)

    if meta1["make"].lower() == "blackvue":
        matches = [m for m in matches if _blackvue_valid_mask(p1[m[0]])]
    if meta2["make"].lower() == "blackvue":
        matches = [m for m in matches if _blackvue_valid_mask(p2[m[1]])]
    return matches


def _blackvue_valid_mask(p: NDArray) -> bool:
    """Check if pixel inside the valid region.

    Pixel coord Y should be smaller than h - 70.
    In normalized coordinates y < (h - 70 - h / 2) / w,
    with h = 2160 and w = 3840
    """
    return p[1] < 0.263


def generate_binary_cache(
        data: DataSetBase,
        pairs: List[Tuple[str, str]],
        config: Dict[str, Any],
        cameras: Dict[str, pygeometry.Camera],
        exifs: Dict[str, Dict[str, Any]]):
    """
    Generate a cache of binarized descriptors for all images, using a projection trained on a random sample of matched pairs.

     - Select a random subset of pairs for training.
     - Match them with FLANN on CPU to get training matches.
     - Collect positive and negative descriptor pairs from the matches.
     - Train a linear projection to separate positives from negatives.
     - Binarize descriptors for all images using the learned projection and store in cache.

    Args:
        data: Dataset to load descriptors from.
        pairs: List of image pairs to sample from for training.
        config: Matching config dict, may contain parameters for training and binarization.
        cameras: Dict mapping image names to their camera models.
        exifs: Dict mapping image names to their EXIF metadata.
        poses: Dict mapping image names to their poses.
    """
    n_sample = min(config.get(
        "binary_training_pairs", 100), len(pairs))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(pairs), n_sample, replace=False)
    sample_pairs = [pairs[i] for i in sample_idx]

    # Force FLANN matching on CPU for training pairs.
    training_config = dict(config)
    training_config["use_opencl_matching"] = False
    training_config["matcher_type"] = "FLANN"
    training_config["use_robust_matching"] = True

    training_args = list(match_arguments(
        sample_pairs, data, training_config, cameras, exifs, None, None))
    processes = config.get("processes", 1)
    processes = context.processes_that_fit_in_memory(processes, 512)
    logger.info(
        "Matching %d training pairs with FLANN (%d processes)",
        len(sample_pairs), processes,
    )
    training_matches = context.parallel_map(
        match_unwrap_args, training_args, processes, 2)

    pos_d1, pos_d2, neg_d1, neg_d2 = collect_training_pairs(
        data, training_matches, training_config,
    )
    P, t = train_dif_projection(
        pos_d1, pos_d2, neg_d1, neg_d2)
    config_override = dict(config)
    config_override["_binary_P"] = P
    config_override["_binary_t"] = t
    logger.info("Binary projection trained: %d positive, %d negative pairs",
                len(pos_d1), len(neg_d1))

    # Binarize all image descriptors once upfront so per-pair
    # matching doesn't repeat the (N, 128) @ (128, 128) matmul.
    segmentation_in_desc = config.get(
        "matching_use_segmentation", False)
    binary_cache: Dict[str, NDArray] = {}
    all_images = set()
    for im1, im2 in pairs:
        all_images.add(im1)
        all_images.add(im2)
    for im in all_images:
        fd = feature_loader.instance.load_all_data(
            data, im, masked=True,
            segmentation_in_descriptor=segmentation_in_desc,
        )
        if fd is not None and fd.descriptors is not None:
            d = fd.descriptors.astype(np.float32)
            binary_cache[im] = binarize_descriptors(d, P, t)

    return binary_cache


def collect_training_pairs(
    data: DataSetBase,
    matched_results: List[Tuple[str, str, NDArray]],
    config: Dict[str, Any],
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Collect positive and negative descriptor pairs from pre-matched image pairs.

    Args:
        data: Dataset to load descriptors from.
        matched_results: List of (im1, im2, matches) from match_unwrap_args.
        config: Matching config dict.

    Returns:
        (pos_d1, pos_d2, neg_d1, neg_d2): Arrays of shape (M, D) each.
    """
    rng = np.random.RandomState(42)
    segmentation_in_desc = config["matching_use_segmentation"]

    pos_d1_list: List[NDArray] = []
    pos_d2_list: List[NDArray] = []
    neg_d1_list: List[NDArray] = []
    neg_d2_list: List[NDArray] = []

    for im1, im2, match_arr in matched_results:
        if len(match_arr) < 5:
            continue

        # Load descriptors for both images.
        fd1 = feature_loader.instance.load_all_data(
            data, im1, masked=True,
            segmentation_in_descriptor=segmentation_in_desc,
        )
        fd2 = feature_loader.instance.load_all_data(
            data, im2, masked=True,
            segmentation_in_descriptor=segmentation_in_desc,
        )
        if fd1 is None or fd2 is None:
            continue
        d1 = fd1.descriptors
        d2 = fd2.descriptors
        if d1 is None or d2 is None or len(d1) < 10 or len(d2) < 10:
            continue
        d1 = d1.astype(np.float32)
        d2 = d2.astype(np.float32)

        match_arr = np.asarray(match_arr, dtype=int)
        if match_arr.ndim != 2 or match_arr.shape[1] != 2:
            continue

        logger.info(
            "Collected %d matches for training from pair (%s, %s)",
            len(match_arr), im1, im2,
        )

        # Positive pairs
        pos_d1_list.append(d1[match_arr[:, 0]])
        pos_d2_list.append(d2[match_arr[:, 1]])

        # Negative pairs: random pairings, excluding actual matches
        n_neg = len(match_arr)
        match_set = set(map(tuple, match_arr.tolist()))
        neg_i = rng.randint(0, len(d1), n_neg * 2)
        neg_j = rng.randint(0, len(d2), n_neg * 2)
        count = 0
        for qi, ri in zip(neg_i, neg_j):
            if (qi, ri) not in match_set:
                neg_d1_list.append(d1[qi: qi + 1])
                neg_d2_list.append(d2[ri: ri + 1])
                count += 1
                if count >= n_neg:
                    break

    if not pos_d1_list:
        raise RuntimeError(
            "No positive pairs found for binary projection training"
        )

    return (
        np.concatenate(pos_d1_list).astype(np.float32),
        np.concatenate(pos_d2_list).astype(np.float32),
        np.concatenate(neg_d1_list).astype(np.float32),
        np.concatenate(neg_d2_list).astype(np.float32),
    )


def train_dif_projection(
    pos_d1: NDArray,
    pos_d2: NDArray,
    neg_d1: NDArray,
    neg_d2: NDArray,
    n_bits: int = 128,
    alpha: float = 10.0,
) -> Tuple[NDArray, NDArray]:
    """Train DIF binary projection from positive and negative descriptor pairs.

    Args:
        pos_d1, pos_d2: Matched descriptor pairs (M_pos, D).
        neg_d1, neg_d2: Non-matched descriptor pairs (M_neg, D).
        n_bits: Number of output bits (must be multiple of 32).
        alpha: Weight balancing negative vs positive covariance.

    Returns:
        P: (n_bits, D) projection matrix (float32).
        t: (n_bits,) threshold vector (float32).
    """
    desc_dim = pos_d1.shape[1]
    logger.info(
        "Training DIF projection: %d positive, %d negative pairs, "
        "dim=%d, bits=%d, alpha=%.1f",
        len(pos_d1), len(neg_d1), desc_dim, n_bits, alpha,
    )

    # DIF: covariance of DIFFERENCE vectors (d1 - d2), not raw descriptors.
    # Σ_S = Cov(d1_pos - d2_pos): similar pairs should have small differences.
    # Σ_D = Cov(d1_neg - d2_neg): dissimilar pairs should have large differences.
    # We maximize w^T (Σ_D - α·Σ_S) w, i.e. find directions where dissimilar
    # differences are spread out but similar differences are concentrated.
    diff_pos = (pos_d1 - pos_d2).astype(np.float64)
    diff_neg = (neg_d1 - neg_d2).astype(np.float64)

    sigma_pos = np.cov(diff_pos.T)  # (D, D)
    sigma_neg = np.cov(diff_neg.T)  # (D, D)

    # DIF: eigendecompose (Sigma_neg - alpha * Sigma_pos), take top eigenvectors
    m = sigma_neg - alpha * sigma_pos
    eigenvalues, eigenvectors = np.linalg.eigh(m)

    # eigh returns ascending order; take the n_bits largest
    idx = np.argsort(eigenvalues)[::-1][:n_bits]
    P = eigenvectors[:, idx].T.astype(np.float32)  # (n_bits, D)

    # Optimize thresholds
    t = optimize_dif_thresholds(P, pos_d1, pos_d2, neg_d1, neg_d2)

    logger.info("DIF projection trained successfully")
    return P, t


def optimize_dif_thresholds(
    P: NDArray,
    pos_d1: NDArray,
    pos_d2: NDArray,
    neg_d1: NDArray,
    neg_d2: NDArray,
    n_bins: int = 500,
) -> NDArray:
    """Optimize per-dimension thresholds to maximize TP + TN.

    For each projection dimension i, finds t_i that minimizes
    FP(t_i) + FN(t_i) using histogram-based CDF estimation.
    """
    n_bits = P.shape[0]

    # Project all pairs at once: (n_bits, N)
    proj_p1 = (P @ pos_d1.T).astype(np.float64)  # (n_bits, N_pos)
    proj_p2 = (P @ pos_d2.T).astype(np.float64)
    proj_n1 = (P @ neg_d1.T).astype(np.float64)  # (n_bits, N_neg)
    proj_n2 = (P @ neg_d2.T).astype(np.float64)

    t = np.zeros(n_bits, dtype=np.float32)

    for i in range(n_bits):
        yp1, yp2 = proj_p1[i], proj_p2[i]
        yn1, yn2 = proj_n1[i], proj_n2[i]

        min_pos = np.minimum(yp1, yp2)
        max_pos = np.maximum(yp1, yp2)
        min_neg = np.minimum(yn1, yn2)
        max_neg = np.maximum(yn1, yn2)

        # Candidate thresholds spanning the data range
        lo = min(min_pos.min(), min_neg.min())
        hi = max(max_pos.max(), max_neg.max())
        candidates = np.linspace(lo, hi, n_bins)  # (n_bins,)

        # Vectorized: FN = P(bits differ for positive) = P(min <= t <= max)
        fn = np.mean(
            (min_pos[None, :] <= candidates[:, None])
            & (candidates[:, None] <= max_pos[None, :]),
            axis=1,
        )
        # FP = P(bits agree for negative) = P(t < min or t > max)
        fp = np.mean(
            (candidates[:, None] < min_neg[None, :])
            | (candidates[:, None] > max_neg[None, :]),
            axis=1,
        )

        score = (1.0 - fn) + (1.0 - fp)  # TP + TN
        # The optimization finds the optimal boundary c in raw projection
        # space P·x.  Binarization uses sign(P·x + t), whose boundary is
        # at P·x = -t.  So the additive term must be t = -c.
        t[i] = -candidates[np.argmax(score)]

    return t


def binarize_descriptors(
    descriptors: NDArray, P: NDArray, t: NDArray
) -> NDArray:
    """Binarize float descriptors using learned projection.

    Args:
        descriptors: (N, D) float32 descriptors.
        P: (n_bits, D) projection matrix.
        t: (n_bits,) thresholds.

    Returns:
        (N, n_bits // 32) uint32 array of packed binary descriptors.
    """
    # Project: (N, D) @ (D, n_bits) -> (N, n_bits)
    projected = descriptors @ P.T + t[None, :]
    bits = projected > 0  # (N, n_bits) bool

    # Pack into uint32 words
    n_bits = P.shape[0]
    n_words = n_bits // 32
    bits_reshaped = bits.reshape(-1, n_words, 32)
    powers = np.uint32(1) << np.arange(32, dtype=np.uint32)
    packed = (bits_reshaped.astype(np.uint32) * powers).sum(axis=2).astype(
        np.uint32
    )

    return np.ascontiguousarray(packed)
