# pyre-strict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from opensfm import matching
from opensfm.actions import match_components
from opensfm.synthetic_data import synthetic_dataset, synthetic_scene


def _m(n: int) -> NDArray:
    return np.zeros((n, 2), dtype=np.int32)


class FakeMatchesDataSet:
    """Minimal stand-in exposing only the matches API used by the action."""

    def __init__(
        self, images: List[str], matches: Dict[str, Dict[str, NDArray]]
    ) -> None:
        self.image_list = images
        self.matches = matches  # per-im1 store, mutated by save_matches
        self.config: Dict[str, Any] = {
            "robust_matching_min_match": 20,
            "matching_components_exhaustive_cap": 100,
            "matching_components_vlad_neighbors": 5,
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
        raise AssertionError("load_exif should not be called on early exit")

    def save_report(self, content: str, path: str) -> None:
        self.reports[path] = content


# --- Unit tests: build_match_graph / image_components ---


def test_build_match_graph_thresholds_edges() -> None:
    data = FakeMatchesDataSet(
        ["a", "b", "c"], {"a": {"b": _m(25), "c": _m(5)}}
    )
    graph = match_components.build_match_graph(data, data.images(), 20)
    assert graph.number_of_nodes() == 3
    assert graph.has_edge("a", "b")
    assert not graph.has_edge("a", "c")


def test_build_match_graph_no_match_files() -> None:
    data = FakeMatchesDataSet(["a", "b", "c"], {})
    graph = match_components.build_match_graph(data, data.images(), 20)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 0
    assert len(match_components.image_components(graph)) == 3


def test_image_components_sorted_largest_first() -> None:
    data = FakeMatchesDataSet(
        ["a", "b", "c", "d", "e"],
        {"a": {"b": _m(25), "c": _m(25)}, "d": {"e": _m(25)}},
    )
    graph = match_components.build_match_graph(data, data.images(), 20)
    components = match_components.image_components(graph)
    assert components == [{"a", "b", "c"}, {"d", "e"}]


# --- Unit tests: select_cross_component_pairs ---


def test_select_pairs_exhaustive_below_cap() -> None:
    data = FakeMatchesDataSet([], {})
    components: List[Set[str]] = [{"a", "b", "c"}, {"d", "e"}]
    pairs, stats = match_components.select_cross_component_pairs(
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
        "fallback_component_pairs": 0,
    }


def test_select_pairs_vlad_above_cap() -> None:
    data = FakeMatchesDataSet([], {})
    components: List[Set[str]] = [{"c", "d", "e"}, {"a", "b"}]
    histograms: Dict[str, NDArray] = {
        "a": np.array([0.0], dtype=np.float32),
        "b": np.array([10.0], dtype=np.float32),
        "c": np.array([0.1], dtype=np.float32),
        "d": np.array([10.1], dtype=np.float32),
        "e": np.array([50.0], dtype=np.float32),
    }
    pairs, stats = match_components.select_cross_component_pairs(
        data,
        {},
        components,
        exhaustive_cap=1,
        vlad_neighbors=1,
        histograms=histograms,
    )
    assert pairs == {("a", "c"), ("b", "d")}
    assert stats["vlad_component_pairs"] == 1
    assert stats["exhaustive_component_pairs"] == 0
    assert stats["fallback_component_pairs"] == 0


def test_select_pairs_fallback_when_vlad_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        match_components.pairs_selection,
        "vlad_histograms",
        lambda imgs, data: {},
    )
    data = FakeMatchesDataSet([], {})
    components: List[Set[str]] = [{"c", "d", "e"}, {"a", "b"}]

    pairs, stats = match_components.select_cross_component_pairs(
        data, {}, components, exhaustive_cap=2, vlad_neighbors=1
    )
    assert stats["fallback_component_pairs"] == 1
    assert len(pairs) == 2

    pairs_again, _ = match_components.select_cross_component_pairs(
        data, {}, components, exhaustive_cap=2, vlad_neighbors=1
    )
    assert pairs == pairs_again


def test_subsampled_pairs_capped_and_deterministic() -> None:
    comp_a = {"a", "b", "c"}
    comp_b = {"d", "e", "f"}
    capped = match_components._subsampled_pairs(comp_a, comp_b, 4)
    assert len(capped) == 4
    all_cross = {(x, y) for x in comp_a for y in comp_b}
    assert capped <= all_cross
    assert capped == match_components._subsampled_pairs(comp_a, comp_b, 4)

    full = match_components._subsampled_pairs(comp_a, comp_b, 100)
    assert len(full) == 9


def test_subsampled_pairs_zero_cap() -> None:
    assert match_components._subsampled_pairs({"a", "b"}, {"c", "d"}, 0) == set()


# --- Unit tests: merge_and_save_matches ---


def test_merge_save_preserves_existing_matches() -> None:
    data = FakeMatchesDataSet(["a"], {"a": {"b": _m(25)}})
    match_components.merge_and_save_matches(data, {("a", "z"): [(0, 1), (2, 3)]})
    assert set(data.matches["a"].keys()) == {"b", "z"}
    assert len(data.matches["a"]["b"]) == 25


def test_merge_save_converts_lists_to_ndarray() -> None:
    data = FakeMatchesDataSet(["a"], {"a": {"b": _m(25)}})
    match_components.merge_and_save_matches(data, {("a", "z"): [(0, 1), (2, 3)]})
    saved = data.matches["a"]["z"]
    assert isinstance(saved, np.ndarray)
    assert saved.shape == (2, 2)

    data_empty = FakeMatchesDataSet(["a"], {"a": {}})
    match_components.merge_and_save_matches(data_empty, {("a", "z"): []})
    assert data_empty.matches["a"]["z"].shape == (0, 2)


def test_merge_save_new_image_without_existing_file() -> None:
    data = FakeMatchesDataSet(["a"], {"a": {"b": _m(25)}})
    match_components.merge_and_save_matches(data, {("q", "r"): [(1, 1)]})
    assert set(data.matches["q"].keys()) == {"r"}


# --- run_dataset tests ---


def test_run_dataset_early_exit_single_component() -> None:
    data = FakeMatchesDataSet(
        ["a", "b", "c"], {"a": {"b": _m(25)}, "b": {"c": _m(25)}}
    )
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
        matched_per_im1: Dict[str, Dict[str, NDArray]] = {}
        for (im1, im2), m in pairs_matches.items():
            matched_per_im1.setdefault(im1, {})[im2] = np.asarray(
                m, dtype=np.int32
            ).reshape(-1, 2)
        for im1, m in matched_per_im1.items():
            store[im1] = m

    synthetic.config["matching_components_exhaustive_cap"] = 10_000

    min_matches = synthetic.config["robust_matching_min_match"]
    before = match_components.image_components(
        match_components.build_match_graph(synthetic, images, min_matches)
    )
    assert len(before) == 2

    # Snapshot existing matches to verify non-destruction.
    existing_snapshot = {
        im: {other: arr.copy() for other, arr in m.items()}
        for im, m in store.items()
    }

    match_components.run_dataset(synthetic)

    after = match_components.image_components(
        match_components.build_match_graph(synthetic, images, min_matches)
    )
    assert len(after) == 1

    for im, m in existing_snapshot.items():
        for other, arr in m.items():
            assert other in store[im]
            assert np.array_equal(store[im][other], arr)

    assert "match_components.json" in reports


def test_command_registered() -> None:
    from opensfm import commands

    assert commands.match_components in commands.opensfm_commands
    idx = commands.opensfm_commands.index(commands.match_components)
    assert idx == commands.opensfm_commands.index(commands.match_features) + 1
    assert commands.match_components.Command().name == "match_components"
