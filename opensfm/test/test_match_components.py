# pyre-strict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from opensfm import matching, tracking
from opensfm.actions import match_components, match_features
from opensfm.synthetic_data import synthetic_dataset, synthetic_scene


def _m(n: int) -> NDArray:
    return np.zeros((n, 2), dtype=np.int32)


class FakeMatchesDataSet:
    """Minimal stand-in exposing only the matches API used by the matching
    graph code."""

    def __init__(
        self, images: List[str], matches: Dict[str, Dict[str, NDArray]]
    ) -> None:
        self.image_list = images
        self.matches = matches  # per-im1 store, mutated by save_matches
        self.config: Dict[str, Any] = {
            "robust_matching_min_match": 20,
            "matching_components_exhaustive_cap": 100,
            "matching_components_vlad_neighbors": 5,
            "matching_merge_components": False,
            "processes": 1,
        }
        self.reports: Dict[str, str] = {}
        self.saved_calls: List[str] = []

    def images(self) -> List[str]:
        return self.image_list

    def matches_exists(self, image: str) -> bool:
        return image in self.matches

    def load_matches(self, image: str) -> Dict[str, NDArray]:
        if image not in self.matches:
            raise IOError(image)  # tracking.load_matches catches IOError
        return dict(self.matches[image])  # copy, like unpickling does

    def save_matches(self, image: str, matches: Dict[str, NDArray]) -> None:
        self.saved_calls.append(image)
        self.matches[image] = matches

    def load_exif(self, image: str) -> Dict[str, Any]:
        return {}

    def save_report(self, content: str, path: str) -> None:
        self.reports[path] = content


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
    data = FakeMatchesDataSet([], {})
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
    data = FakeMatchesDataSet([], {})
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
    data = FakeMatchesDataSet([], {})
    components: List[Set[str]] = [{"c", "d", "e"}, {"a", "b"}]

    pairs, stats = matching._select_cross_component_pairs(
        data, {}, components, exhaustive_cap=2, vlad_neighbors=1
    )

    assert pairs == set()
    assert stats["skipped_component_pairs"] == 1


# --- Unit tests: matching.save_matches_merging ---


def test_merge_save_preserves_existing_matches() -> None:
    data = FakeMatchesDataSet(["a"], {"a": {"b": _m(25)}})
    matching.save_matches_merging(data, {("a", "z"): [(0, 1), (2, 3)]})
    assert set(data.matches["a"].keys()) == {"b", "z"}
    assert len(data.matches["a"]["b"]) == 25


def test_merge_save_converts_lists_to_ndarray() -> None:
    data = FakeMatchesDataSet(["a"], {"a": {"b": _m(25)}})
    matching.save_matches_merging(data, {("a", "z"): [(0, 1), (2, 3)]})
    saved = data.matches["a"]["z"]
    assert isinstance(saved, np.ndarray)
    assert saved.shape == (2, 2)

    data_empty = FakeMatchesDataSet(["a"], {"a": {}})
    matching.save_matches_merging(data_empty, {("a", "z"): []})
    assert data_empty.matches["a"]["z"].shape == (0, 2)


def test_merge_save_new_image_without_existing_file() -> None:
    data = FakeMatchesDataSet(["a"], {"a": {"b": _m(25)}})
    matching.save_matches_merging(data, {("q", "r"): [(1, 1)]})
    assert set(data.matches["q"].keys()) == {"r"}


def test_merge_save_never_touches_existing() -> None:
    data = FakeMatchesDataSet(["a", "b"], {"b": {"a": _m(5), "c": _m(25)}})
    matching.save_matches_merging(data, {("a", "b"): [(0, 1)]})
    assert "a" not in data.matches
    assert set(data.matches["b"].keys()) == {"a", "c"}
    assert len(data.matches["b"]["a"]) == 5


# --- Unit tests: bridge_matching_components ---


def test_bridge_early_exit_single_component() -> None:
    data = FakeMatchesDataSet(["a", "b", "c"], {})
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
    data = FakeMatchesDataSet(["a", "b", "c"], {})

    new_matches, report = matching.bridge_matching_components(
        data, {}, ["a", "b", "c"], pairs_matches, exifs={}
    )

    assert requested == [("b", "c")]
    assert set(new_matches) == {("b", "c")}
    assert report["num_components_before"] == 2
    assert report["num_components_after"] == 1
    assert ("c", "a") in pairs_matches  # input untouched


# --- match_features integration of the bridging pass ---


def _run_match_features(
    monkeypatch: pytest.MonkeyPatch, merge_components: bool
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], List[Tuple[int, int]]]]:
    data = FakeMatchesDataSet(["a", "b"], {})
    data.config["matching_merge_components"] = merge_components

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

    match_features.run_dataset(data)
    return called, saved


def test_match_features_bridges_components_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called, saved = _run_match_features(monkeypatch, merge_components=True)
    assert called["exifs"] is not None  # EXIFs passed through, not reloaded
    assert set(saved) == {("a", "b"), ("a", "z")}


def test_match_features_no_bridging_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called, saved = _run_match_features(monkeypatch, merge_components=False)
    assert called == {}
    assert set(saved) == {("a", "b")}


# --- Standalone command tests ---


def test_run_dataset_early_exit_single_component() -> None:
    data = FakeMatchesDataSet(
        ["a", "b", "c"], {"a": {"b": _m(25)}, "b": {"c": _m(25)}}
    )

    def _fail(image: str) -> Dict[str, Any]:
        raise AssertionError("load_exif should not be called on early exit")

    # pyre-fixme[8]: monkeypatch onto the instance.
    data.load_exif = _fail

    match_components.run_dataset(data)

    assert data.saved_calls == []
    assert "match_components.json" in data.reports
    report = data.reports["match_components.json"]
    assert '"num_components_before": 1' in report
    assert '"num_components_after": 1' in report
    assert '"num_candidate_pairs": 0' in report


def test_run_dataset_bridges_two_components(
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

    store: Dict[str, Dict[str, NDArray]] = {}
    reports: Dict[str, str] = {}

    # pyre-fixme[8]: monkeypatch matches/report I/O onto the instance.
    synthetic.matches_exists = lambda im: im in store

    def _load(im: str) -> Dict[str, NDArray]:
        if im not in store:
            raise IOError(im)
        return dict(store[im])

    # pyre-fixme[8]: monkeypatch matches/report I/O onto the instance.
    synthetic.load_matches = _load
    # pyre-fixme[8]: monkeypatch matches/report I/O onto the instance.
    synthetic.save_matches = lambda im, m: store.__setitem__(im, m)
    # pyre-fixme[8]: monkeypatch matches/report I/O onto the instance.
    synthetic.save_report = lambda content, path: reports.__setitem__(path, content)

    override = {
        "matching_gps_neighbors": 0,
        "matching_gps_distance": 0,
        "matching_time_neighbors": 2,
    }
    for half in (half_a, half_b):
        pairs_matches, _ = matching.match_images(synthetic, override, half, half)
        matching.save_matches_merging(synthetic, pairs_matches)

    synthetic.config["matching_components_exhaustive_cap"] = 10_000

    min_matches = synthetic.config["robust_matching_min_match"]
    before = matching.image_components(
        matching.build_match_graph(
            images, dict(tracking.load_matches(synthetic, images)), min_matches
        )
    )
    assert len(before) == 2

    # Snapshot existing matches to verify they are untouched.
    existing_snapshot = {
        im: {other: arr.copy() for other, arr in m.items()}
        for im, m in store.items()
    }

    match_components.run_dataset(synthetic)

    after = matching.image_components(
        matching.build_match_graph(
            images, dict(tracking.load_matches(synthetic, images)), min_matches
        )
    )
    assert len(after) == 1

    for im, m in existing_snapshot.items():
        for other, arr in m.items():
            assert np.array_equal(store[im][other], arr)

    assert "match_components.json" in reports
    assert '"num_components_before": 2' in reports["match_components.json"]


def test_command_registered() -> None:
    from opensfm import commands

    assert commands.match_components in commands.opensfm_commands
    idx = commands.opensfm_commands.index(commands.match_components)
    assert idx == commands.opensfm_commands.index(commands.match_features) + 1
    assert commands.match_components.Command().name == "match_components"
