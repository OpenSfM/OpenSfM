# pyre-strict
import os
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from opensfm import dataset, matching
from opensfm.actions import detect_features, extract_metadata, match_features
from opensfm.synthetic_data import synthetic_dataset, synthetic_scene
from opensfm.test import data_generation


def _m(n: int) -> NDArray:
    return np.zeros((n, 2), dtype=np.int32)


class FakeDataSet:
    """Minimal stand-in exposing only what the matching graph code uses."""

    def __init__(self, images: List[str]) -> None:
        self.image_list = images
        self.config: Dict[str, Any] = {
            "robust_matching_min_match": 20,
            "matching_components_exhaustive_cap": 100,
            "matching_components_vlad_neighbors": 5,
            "force_match_components": False,
            "processes": 1,
        }

    def images(self) -> List[str]:
        return self.image_list

    def load_exif(self, image: str) -> Dict[str, Any]:
        return {}


# --- Unit tests: build_match_graph / image_components ---


def test_build_match_graph_thresholds_edges() -> None:
    graph = matching.build_match_graph(
        ["a", "b", "c"], {("a", "b"): _m(25), ("a", "c"): _m(5)}, 20
    )
    assert graph.number_of_nodes() == 3
    assert graph.has_edge("a", "b")
    assert not graph.has_edge("a", "c")


def test_build_match_graph_no_matches() -> None:
    graph = matching.build_match_graph(["a", "b", "c"], {}, 20)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 0
    assert len(matching.image_components(graph)) == 3


def test_image_components_sorted_largest_first() -> None:
    graph = matching.build_match_graph(
        ["a", "b", "c", "d", "e"],
        {("a", "b"): _m(25), ("a", "c"): _m(25), ("d", "e"): _m(25)},
        20,
    )
    components = matching.image_components(graph)
    assert components == [{"a", "b", "c"}, {"d", "e"}]


# --- Unit tests: _select_cross_component_pairs ---


CandidateCall = Tuple[List[str], List[str], Dict[str, Any]]


def _fake_candidates(
    monkeypatch: pytest.MonkeyPatch, calls: List[CandidateCall]
) -> None:
    """Replace pairs_selection.match_candidates_from_metadata with a fake
    returning all ref x cand pairs and recording its arguments."""

    def fake(
        ref: List[str],
        cand: List[str],
        exifs: Dict[str, Any],
        data: Any,
        override: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        calls.append((list(ref), list(cand), dict(override)))
        return [(r, c) for r in ref for c in cand], {}

    monkeypatch.setattr(
        matching.pairs_selection, "match_candidates_from_metadata", fake
    )


def test_select_pairs_exhaustive_below_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[CandidateCall] = []
    _fake_candidates(monkeypatch, calls)
    data = FakeDataSet([])
    components: List[Set[str]] = [{"a", "b", "c"}, {"d", "e"}]

    pairs, stats = matching._select_cross_component_pairs(
        data, {}, components, exhaustive_cap=100, vlad_neighbors=5
    )

    expected = {
        ("a", "d"), ("a", "e"), ("b", "d"),
        ("b", "e"), ("c", "d"), ("c", "e"),
    }
    assert pairs == expected
    assert stats == {
        "exhaustive_component_pairs": 1,
        "vlad_component_pairs": 0,
        "skipped_component_pairs": 0,
    }
    ref, cand, override = calls[0]
    assert ref == ["d", "e"]  # smaller component as ref
    assert cand == ["a", "b", "c"]
    assert override["matching_vlad_neighbors"] == 0
    assert override["matching_gps_distance"] == 0
    assert override["matching_order_neighbors"] == 0


def test_select_pairs_vlad_above_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[CandidateCall] = []
    _fake_candidates(monkeypatch, calls)
    data = FakeDataSet([])
    components: List[Set[str]] = [{"c", "d", "e"}, {"a", "b"}]

    _, stats = matching._select_cross_component_pairs(
        data, {}, components, exhaustive_cap=1, vlad_neighbors=7
    )

    assert stats["vlad_component_pairs"] == 1
    assert stats["exhaustive_component_pairs"] == 0
    ref, cand, override = calls[0]
    assert ref == ["a", "b"]
    assert cand == ["c", "d", "e"]
    assert override["matching_vlad_neighbors"] == 7
    assert override["matching_vlad_gps_distance"] == 0
    assert override["matching_vlad_gps_neighbors"] == 0
    assert override["matching_bow_neighbors"] == 0


def test_select_pairs_skips_when_vlad_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: Any) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        raise FileNotFoundError("no VLAD words")

    monkeypatch.setattr(
        matching.pairs_selection, "match_candidates_from_metadata", _raise
    )
    data = FakeDataSet([])
    components: List[Set[str]] = [{"c", "d", "e"}, {"a", "b"}]

    pairs, stats = matching._select_cross_component_pairs(
        data, {}, components, exhaustive_cap=2, vlad_neighbors=1
    )

    assert pairs == set()
    assert stats["skipped_component_pairs"] == 1


# --- Unit tests: bridge_matching_components ---


def test_bridge_early_exit_single_component() -> None:
    data = FakeDataSet(["a", "b", "c"])
    pairs_matches: Dict[Tuple[str, str], List[Tuple[int, int]]] = {
        ("a", "b"): [(0, 0)] * 25,
        ("b", "c"): [(0, 0)] * 25,
    }

    new_matches, report = matching.bridge_matching_components(
        data, {}, ["a", "b", "c"], pairs_matches
    )

    assert new_matches == {}
    assert report["num_components_before"] == 1
    assert report["num_components_after"] == 1
    assert report["num_candidate_pairs"] == 0


def test_bridge_excludes_attempted_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs_matches: Dict[Tuple[str, str], List[Tuple[int, int]]] = {
        ("a", "b"): [(0, 0)] * 25,
        ("c", "a"): [(0, 0)] * 3,  # already attempted, sub-threshold
    }
    requested: List[Tuple[str, str]] = []

    def fake_match(
        data: Any,
        override: Dict[str, Any],
        exifs: Dict[str, Any],
        pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        requested.extend(pairs)
        return {p: [(0, 0)] * 30 for p in pairs}

    def fake_select(
        data: Any,
        exifs: Dict[str, Any],
        components: List[Set[str]],
        cap: int,
        k: int,
    ) -> Tuple[Set[Tuple[str, str]], Dict[str, int]]:
        return {("a", "c"), ("b", "c")}, matching._empty_selection_stats()

    monkeypatch.setattr(matching, "match_images_with_pairs", fake_match)
    monkeypatch.setattr(matching, "_select_cross_component_pairs", fake_select)
    data = FakeDataSet(["a", "b", "c"])

    new_matches, report = matching.bridge_matching_components(
        data, {}, ["a", "b", "c"], pairs_matches, exifs={}
    )

    assert requested == [("b", "c")]
    assert set(new_matches) == {("b", "c")}
    assert report["num_components_before"] == 2
    assert report["num_components_after"] == 1
    assert ("c", "a") in pairs_matches  # input untouched


def test_bridge_two_components_integration(
    scene_synthetic: synthetic_scene.SyntheticInputData,
) -> None:
    synthetic = synthetic_dataset.SyntheticDataSet(
        scene_synthetic.reconstruction,
        scene_synthetic.exifs,
        scene_synthetic.features,
        scene_synthetic.tracks_manager,
    )

    images = sorted(synthetic.images())[:16]
    synthetic.image_list = images
    half_a, half_b = images[:8], images[8:]

    # Two disjoint components matched with the real matcher.
    pairs_matches: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    override = {
        "matching_gps_neighbors": 0,
        "matching_gps_distance": 0,
        "matching_time_neighbors": 2,
    }
    for half in (half_a, half_b):
        half_matches, _ = matching.match_images(synthetic, override, half, half)
        pairs_matches.update(half_matches)

    synthetic.config["matching_components_exhaustive_cap"] = 10_000
    min_matches = synthetic.config["robust_matching_min_match"]
    before = matching.image_components(
        matching.build_match_graph(images, pairs_matches, min_matches)
    )
    assert len(before) == 2

    snapshot = dict(pairs_matches)
    new_matches, report = matching.bridge_matching_components(
        synthetic, {}, images, pairs_matches
    )

    assert report["num_components_before"] == 2
    assert report["num_components_after"] == 1
    assert new_matches
    assert not set(new_matches) & set(pairs_matches)
    # The initial matches are untouched.
    assert set(pairs_matches) == set(snapshot)
    assert all(pairs_matches[k] is snapshot[k] for k in snapshot)


def _create_lund_test_folder(tmpdir: Any) -> dataset.DataSet:
    src = os.path.join(data_generation.DATA_PATH, "lund")
    dst = str(tmpdir.mkdir("lund"))
    for filename in ["images", "config.yaml"]:
        os.symlink(os.path.join(src, filename), os.path.join(dst, filename))
    return dataset.DataSet(dst)


def test_bridge_components_real_vlad_selection(tmpdir: Any) -> None:
    """Regression test for the real (non-monkeypatched) VLAD selection path,
    taken when a component pair is above matching_components_exhaustive_cap.
    The other bridge tests fake pairs_selection.match_candidates_from_metadata;
    this one exercises the on-disk VLAD codebook load and the
    feature_type/feature_root/hahog_normalize_to_uchar assertions in
    opensfm/bow.py, which _select_cross_component_pairs's `except OSError`
    does not catch.
    """
    data = _create_lund_test_folder(tmpdir)
    images = sorted(data.images())[:6]
    data.image_list = images
    half_a, half_b = images[:3], images[3:]

    extract_metadata.run_dataset(data)
    detect_features.run_dataset(data)

    pairs_matches: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for half in (half_a, half_b):
        half_matches, _ = matching.match_images(data, {}, half, half)
        pairs_matches.update(half_matches)

    min_matches = data.config["robust_matching_min_match"]
    before = matching.image_components(
        matching.build_match_graph(images, pairs_matches, min_matches)
    )
    assert len(before) == 2

    data.config["matching_components_exhaustive_cap"] = 0  # force the VLAD branch
    new_matches, report = matching.bridge_matching_components(
        data, {}, images, pairs_matches
    )

    assert report["vlad_component_pairs"] == 1
    assert report["exhaustive_component_pairs"] == 0
    assert report["skipped_component_pairs"] == 0
    assert report["num_candidate_pairs"] > 0
    assert isinstance(new_matches, dict)


# --- match_features integration of the bridging pass ---


def _run_match_features(
    monkeypatch: pytest.MonkeyPatch, force_match_components: bool
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], List[Tuple[int, int]]]]:
    data = FakeDataSet(["a", "b"])
    data.config["force_match_components"] = force_match_components

    monkeypatch.setattr(
        match_features.matching,
        "match_images",
        lambda d, o, r, c, exifs=None: ({("a", "b"): [(0, 0)] * 25}, {}),
    )
    called: Dict[str, Any] = {}

    def fake_bridge(
        d: Any,
        o: Dict[str, Any],
        images: List[str],
        pairs_matches: Dict[Tuple[str, str], List[Tuple[int, int]]],
        exifs: Any = None,
    ) -> Tuple[Dict[Tuple[str, str], List[Tuple[int, int]]], Dict[str, Any]]:
        called["exifs"] = exifs
        return {("a", "z"): [(1, 1)] * 30}, {"num_components_before": 2}

    monkeypatch.setattr(
        match_features.matching, "bridge_matching_components", fake_bridge
    )
    saved: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    monkeypatch.setattr(
        match_features.matching,
        "save_matches",
        lambda d, imgs, pairs_matches: saved.update(pairs_matches),
    )
    reports: Dict[str, str] = {}
    # pyre-fixme[16]: monkeypatch report saving onto the fake.
    data.save_report = lambda content, path: reports.__setitem__(path, content)

    match_features.run_dataset(data)
    return called, saved


def test_match_features_bridges_components_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called, saved = _run_match_features(monkeypatch, force_match_components=True)
    assert called["exifs"] is not None  # EXIFs passed through, not reloaded
    assert set(saved) == {("a", "b"), ("a", "z")}


def test_match_features_no_bridging_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called, saved = _run_match_features(monkeypatch, force_match_components=False)
    assert called == {}
    assert set(saved) == {("a", "b")}
