# pyre-strict
import sys
import argparse
import os.path
import shutil
from typing import Any, Dict, Iterator

import numpy as np
import pytest

from opensfm import commands, dataset, feature_loader, geo, pairs_selection
from opensfm.dataset_base import DataSetBase
from opensfm.test import data_generation


NEIGHBORS = 6


@pytest.fixture(scope="module", autouse=True)
def clear_cache() -> Iterator[None]:
    """
    Clear feature loader cache to avoid using cached
    masks etc from berlin dataset which has the same
    naming convention for images.
    """
    feature_loader.instance.clear_cache()
    yield
    feature_loader.instance.clear_cache()


@pytest.fixture(scope="module", autouse=True)
def lund_path(tmpdir_factory: Any) -> str:
    """
    Precompute exif and features to avoid doing
    it for every test which is time consuming.
    """
    src = os.path.join(data_generation.DATA_PATH, "lund", "images")
    path = str(tmpdir_factory.mktemp("lund"))
    dst = os.path.join(path, "images")
    if sys.platform == "win32":
        os.mkdir(dst)
        for f in os.listdir(src):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    else:
        os.symlink(src, dst)

    # Use words matcher type to support the bow retrieval test
    data_generation.save_config({"matcher_type": "WORDS"}, path)

    args = argparse.Namespace()
    args.dataset = path
    args.force = True

    data = dataset.DataSet(path)
    commands.extract_metadata.Command().run(data, args)
    commands.detect_features.Command().run(data, args)

    return path


def match_candidates_from_metadata(
    data: DataSetBase, neighbors: int = NEIGHBORS, assert_count: int = NEIGHBORS
) -> None:
    assert neighbors >= assert_count

    ims = sorted(data.images())
    ims_ref = ims[:1]
    ims_cand = ims[1:]

    exifs = {im: data.load_exif(im) for im in ims}

    pairs, _ = pairs_selection.match_candidates_from_metadata(
        ims_ref,
        ims_cand,
        exifs,
        data,
        {},
    )

    matches = [p[1] for p in pairs]
    names = ["{}.jpg".format(str(i).zfill(2)) for i in range(2, 2 + neighbors)]
    count = 0
    for name in names:
        if name in matches:
            count += 1

    assert count >= assert_count


def create_match_candidates_config(**kwargs: Any) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "matcher_type": "BRUTEFORCE",
        "matching_gps_distance": 0,
        "matching_gps_neighbors": 0,
        "matching_time_neighbors": 0,
        "matching_order_neighbors": 0,
        "matching_bow_neighbors": 0,
        "matching_vlad_neighbors": 0,
        "matching_graph_rounds": 0,
    }

    for key, value in kwargs.items():
        config[key] = value

    return config


def test_match_candidates_from_metadata_vlad(lund_path: str) -> None:
    config = create_match_candidates_config(matching_vlad_neighbors=NEIGHBORS)
    data_generation.save_config(config, lund_path)
    data = dataset.DataSet(lund_path)
    match_candidates_from_metadata(data, assert_count=5)


def test_match_candidates_from_metadata_bow(lund_path: str) -> None:
    config = create_match_candidates_config(
        matching_bow_neighbors=NEIGHBORS, matcher_type="WORDS"
    )
    data_generation.save_config(config, lund_path)
    data = dataset.DataSet(lund_path)
    match_candidates_from_metadata(data, assert_count=4)


def test_match_candidates_from_metadata_gps(lund_path: str) -> None:
    config = create_match_candidates_config(matching_gps_neighbors=NEIGHBORS)
    data_generation.save_config(config, lund_path)
    data = dataset.DataSet(lund_path)
    match_candidates_from_metadata(data)


def test_match_candidates_from_metadata_time(lund_path: str) -> None:
    config = create_match_candidates_config(matching_time_neighbors=NEIGHBORS)
    data_generation.save_config(config, lund_path)
    data = dataset.DataSet(lund_path)
    match_candidates_from_metadata(data)


def test_match_candidates_from_metadata_graph(lund_path: str) -> None:
    config = create_match_candidates_config(matching_graph_rounds=50)
    data_generation.save_config(config, lund_path)
    data = dataset.DataSet(lund_path)
    match_candidates_from_metadata(data)


def test_match_candidates_no_gps_ref_against_gps_candidates(lund_path: str) -> None:
    """A ref image without GPS must still be matched against GPS candidates."""
    config = create_match_candidates_config(matching_order_neighbors=2)
    data_generation.save_config(config, lund_path)
    data = dataset.DataSet(lund_path)

    ims = sorted(data.images())
    ims_ref, ims_cand = ims[:1], ims[1:]
    exifs = {im: data.load_exif(im) for im in ims}
    exifs[ims_ref[0]] = {k: v for k, v in exifs[ims_ref[0]].items() if k != "gps"}

    pairs, _ = pairs_selection.match_candidates_from_metadata(
        ims_ref, ims_cand, exifs, data, {}
    )

    assert len(pairs) > 0
    assert all(ims_ref[0] in pair for pair in pairs)


def test_get_gps_point() -> None:
    reference = geo.TopocentricConverter(0, 0, 0)
    exifs = {}
    exifs["gps"] = {
        "latitude": 0.0001,
        "longitude": 0.0001,
        "altitude": 100.0,
    }
    origin, direction = pairs_selection.get_gps_point(exifs, reference)
    assert np.allclose(origin, [[11.131, 11.057, 0.0]], atol=1e-3)
    assert np.allclose(direction, [[0, 0, 1]])


def test_get_gps_opk_point() -> None:
    reference = geo.TopocentricConverter(0, 0, 0)
    exifs = {}
    exifs["gps"] = {
        "latitude": 0.0001,
        "longitude": 0.0001,
        "altitude": 100.0,
    }
    exifs["opk"] = {
        "omega": 45,
        "phi": 0,
        "kappa": 45,
    }
    origin, direction = pairs_selection.get_gps_opk_point(exifs, reference)
    assert np.allclose(origin, [[11.131, 11.057, 0.0]], atol=1e-3)
    assert np.allclose(direction, [[0.0, 1.0, -1.0]])


def test_find_best_altitude_convergent() -> None:
    origins = {"0": np.array([2.0, 0.0, 8.0]), "1": np.array([-2.0, 0.0, 8.0])}
    directions = {
        "0": np.array([-1.0, 0.0, -1.0]),
        "1": np.array([1.0, 0.0, -1.0]),
    }
    altitude = pairs_selection.find_best_altitude(origins, directions)
    assert np.allclose([altitude], [1.0], atol=1e-2)


def test_find_best_altitude_divergent() -> None:
    origins = {"0": np.array([2.0, 0.0, 8.0]), "1": np.array([-2.0, 0.0, 8.0])}
    directions = {
        "0": np.array([1.0, 0.0, -1.0]),
        "1": np.array([-1.0, 0.0, -1.0]),
    }
    altitude = pairs_selection.find_best_altitude(origins, directions)
    assert np.allclose([altitude], pairs_selection.DEFAULT_Z, atol=1e-2)


# ── sorted_pair ──────────────────────────────────────────────────────


def test_sorted_pair_already_sorted() -> None:
    assert pairs_selection.sorted_pair("a", "b") == ("a", "b")


def test_sorted_pair_reversed() -> None:
    assert pairs_selection.sorted_pair("b", "a") == ("a", "b")


def test_sorted_pair_equal() -> None:
    assert pairs_selection.sorted_pair("x", "x") == ("x", "x")


# ── has_gps_info ─────────────────────────────────────────────────────


def test_has_gps_info_complete() -> None:
    exif = {"gps": {"latitude": 1.0, "longitude": 2.0}}
    assert pairs_selection.has_gps_info(exif)


def test_has_gps_info_missing_gps_key() -> None:
    assert not pairs_selection.has_gps_info({})


def test_has_gps_info_missing_latitude() -> None:
    exif = {"gps": {"longitude": 2.0}}
    assert not pairs_selection.has_gps_info(exif)


def test_has_gps_info_empty_exif() -> None:
    assert not pairs_selection.has_gps_info({})


# ── match_candidates_by_order ────────────────────────────────────────


def test_match_candidates_by_order_basic() -> None:
    """Neighbors are selected by sequence proximity."""
    images = ["im0", "im1", "im2", "im3", "im4"]
    pairs = pairs_selection.match_candidates_by_order(
        images, images, max_neighbors=2)
    # im0 should match im1 (within window of 1 on each side)
    assert ("im0", "im1") in pairs or ("im1", "im0") in pairs


def test_match_candidates_by_order_zero() -> None:
    """max_neighbors=0 returns empty set."""
    images = ["a", "b", "c"]
    pairs = pairs_selection.match_candidates_by_order(
        images, images, max_neighbors=0)
    assert len(pairs) == 0


def test_match_candidates_by_order_window() -> None:
    """Window size controls which neighbors are included."""
    images = ["im0", "im1", "im2", "im3", "im4"]
    # max_neighbors=4 → n=2 → window [-2, +2]
    pairs = pairs_selection.match_candidates_by_order(
        images, images, max_neighbors=4)
    # im0 should reach im2
    sorted_pairs = {tuple(sorted(p)) for p in pairs}
    assert ("im0", "im2") in sorted_pairs


# ── match_candidates_by_time ────────────────────────────────────────


def test_match_candidates_by_time_basic() -> None:
    """Nearest-time images are selected."""
    images = ["a", "b", "c", "d"]
    exifs = {
        "a": {"capture_time": 0.0},
        "b": {"capture_time": 1.0},
        "c": {"capture_time": 2.0},
        "d": {"capture_time": 10.0},
    }
    pairs = pairs_selection.match_candidates_by_time(
        images, images, exifs, max_neighbors=2)
    sorted_pairs = {tuple(sorted(p)) for p in pairs}
    # "a" should match "b" and "c" (nearest 2)
    assert ("a", "b") in sorted_pairs
    assert ("a", "c") in sorted_pairs


def test_match_candidates_by_time_zero() -> None:
    """max_neighbors=0 returns empty set."""
    images = ["a", "b"]
    exifs = {"a": {"capture_time": 0}, "b": {"capture_time": 1}}
    pairs = pairs_selection.match_candidates_by_time(
        images, images, exifs, max_neighbors=0)
    assert len(pairs) == 0


# ── match_candidates_by_distance ─────────────────────────────────────


def test_match_candidates_by_distance_nearby() -> None:
    """Nearby GPS positions produce pairs."""
    reference = geo.TopocentricConverter(0, 0, 0)
    exifs = {
        "a": {"gps": {"latitude": 0.0, "longitude": 0.0}},
        "b": {"gps": {"latitude": 0.0001, "longitude": 0.0}},
        "c": {"gps": {"latitude": 1.0, "longitude": 1.0}},
    }
    pairs = pairs_selection.match_candidates_by_distance(
        ["a"], ["a", "b", "c"], exifs, reference,
        max_neighbors=2, max_distance=100.0, use_opk=False,
    )
    sorted_pairs = {tuple(sorted(p)) for p in pairs}
    assert ("a", "b") in sorted_pairs
    # "c" is too far away for max_distance=100m
    assert ("a", "c") not in sorted_pairs


def test_match_candidates_by_distance_empty_candidates() -> None:
    """Empty candidate set returns empty result."""
    reference = geo.TopocentricConverter(0, 0, 0)
    pairs = pairs_selection.match_candidates_by_distance(
        ["a"], [], {}, reference, max_neighbors=5, max_distance=100.0, use_opk=False,
    )
    assert len(pairs) == 0


# ── pairs_from_neighbors ────────────────────────────────────────────


def test_pairs_from_neighbors_same_and_other_camera() -> None:
    """Pairs are split between same-camera and other-camera neighbors."""
    exifs = {
        "ref": {"camera": "cam1"},
        "same1": {"camera": "cam1"},
        "same2": {"camera": "cam1"},
        "other1": {"camera": "cam2"},
        "other2": {"camera": "cam2"},
    }
    distances = [1.0, 2.0, 3.0, 4.0]
    order = [0, 1, 2, 3]
    other = ["same1", "same2", "other1", "other2"]
    pairs = pairs_selection.pairs_from_neighbors(
        "ref", exifs, distances, order, other, max_neighbors=2
    )
    pair_keys = set(pairs.keys())
    # Should include same-camera neighbors AND other-camera neighbors
    assert len(pair_keys) == 4  # 2 same + 2 other


def test_pairs_from_neighbors_max_neighbors_limit() -> None:
    """Only max_neighbors of each category are included."""
    exifs = {
        "ref": {"camera": "cam1"},
        "s1": {"camera": "cam1"},
        "s2": {"camera": "cam1"},
        "s3": {"camera": "cam1"},
    }
    distances = [1.0, 2.0, 3.0]
    order = [0, 1, 2]
    other = ["s1", "s2", "s3"]
    pairs = pairs_selection.pairs_from_neighbors(
        "ref", exifs, distances, order, other, max_neighbors=1
    )
    # Only 1 same-camera neighbor, 0 other-camera
    assert len(pairs) == 1


# ── construct_pairs ──────────────────────────────────────────────────


def test_construct_pairs_no_enforce() -> None:
    """Without enforce_other_cameras, pick top-N by distance."""
    results = [
        ("im1", [1.0, 2.0, 0.5], ["im2", "im3", "im4"]),
    ]
    exifs = {
        "im1": {"camera": "c1"},
        "im2": {"camera": "c1"},
        "im3": {"camera": "c1"},
        "im4": {"camera": "c1"},
    }
    pairs = pairs_selection.construct_pairs(
        results, max_neighbors=2, exifs=exifs, enforce_other_cameras=False)
    assert len(pairs) == 2
    # im4 (distance 0.5) and im1 (distance 1.0) should be closest
    sorted_keys = {tuple(sorted(k)) for k in pairs.keys()}
    assert ("im1", "im4") in sorted_keys


def test_construct_pairs_enforce_other_cameras() -> None:
    """With enforce_other_cameras, both same and other camera neighbors are included."""
    results = [
        ("im1", [1.0, 2.0, 3.0], ["im2", "im3", "im4"]),
    ]
    exifs = {
        "im1": {"camera": "c1"},
        "im2": {"camera": "c1"},
        "im3": {"camera": "c2"},
        "im4": {"camera": "c2"},
    }
    pairs = pairs_selection.construct_pairs(
        results, max_neighbors=1, exifs=exifs, enforce_other_cameras=True
    )
    sorted_keys = {tuple(sorted(k)) for k in pairs.keys()}
    # Should have 1 same-camera (im2) + 1 other-camera (im3)
    assert ("im1", "im2") in sorted_keys
    assert ("im1", "im3") in sorted_keys
