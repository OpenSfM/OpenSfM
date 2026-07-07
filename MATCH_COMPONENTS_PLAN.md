# Implementation Plan: `match_components` — bridge disconnected matching-graph components

## Context

OpenSfM's pair selection (`opensfm/pairs_selection.py`) is optimized for GPS-tagged captures. On GPS-denied / no-GPS datasets, the selected pairs can leave the image matching graph split into several disjoint connected components after `match_features`, and `reconstruct` then emits multiple partial reconstructions. This plan adds an extra pipeline pass, run after `match_features`, that detects disconnected components and re-runs feature matching on cross-component image pairs so `create_tracks` can bridge them.

This document is self-contained: implement it exactly as written, without making new design decisions.

**Fixed design decisions**:
- Standalone command + action pair (`match_components`), mirroring `match_features`. `match_features` itself is untouched.
- Hybrid cross-component pair selection: exhaustive when the candidate count is under a configurable cap, VLAD-similarity-guided above it.

**Verified key facts** (rely on these):
- `matching.match_images_with_pairs(data, config_override, exifs, pairs, poses=None)` (`opensfm/matching.py:112`) matches an explicit pair list — no pair selection involved. Returns `{(im1, im2): matches}`.
- **Critical pitfall**: `DataSet.save_matches(image, matches)` (`opensfm/dataset.py:380`) pickles the whole per-image dict, fully overwriting `matches/{image}_matches.pkl.gz`. `matching.save_matches` therefore destroys pre-existing matches not present in its input. The new pass must load-merge-save.
- `tracking.load_matches(data, images)` (`opensfm/tracking.py:67`) iterates saved match files (tolerates missing files via `except IOError`), yielding `((im1, im2), matches)`.
- `networkx>=2.5` is already a dependency (`pyproject.toml`); graph + `connected_components` template exists in `opensfm/rig.py:162-179`.
- Reusable helpers: `pairs_selection.sorted_pair` (`:27`), `pairs_selection.vlad_histograms(images, data)` (`:879`), `pairs_selection.construct_pairs(results, max_neighbors, exifs, enforce_other_cameras)` (`:520`), `vlad.vlad_distances(image, other_images, histograms)` (`opensfm/vlad.py:35`; `other_images` must be a `list`, it does `other_images + [image]`).
- Edge threshold: reuse config `robust_matching_min_match` (default 20, `opensfm/config.py:213`) — same semantics ("minimum number of matches to accept matches between two images").
- Downstream needs no changes: `create_tracks` consumes match files; `incremental_reconstruction` already emits one reconstruction per graph component, so bridging components upstream directly reduces partial reconstructions.

---

## 1. New action: `opensfm/actions/match_components.py`

Header `# pyre-strict`, full type annotations. Imports: `logging`, `collections.defaultdict`, `itertools.combinations`, `timeit.default_timer as timer`, typing, `networkx as nx`, `numpy as np`, `numpy.typing.NDArray`, `from opensfm import io, matching, pairs_selection, tracking, vlad`, `from opensfm.dataset_base import DataSetBase`.

### 1.1 `run_dataset(data: DataSetBase) -> None`

Docstring: *"Match features between images belonging to different connected components of the matching graph."* Steps in order:

1. `start = timer()`; `images = data.images()`; `min_matches: int = data.config["robust_matching_min_match"]`.
2. `components = image_components(build_match_graph(data, images, min_matches))`; log count.
3. If `len(components) <= 1`: log "matching graph already connected", write report (before == after, zero pairs), return.
4. `exifs = {im: data.load_exif(im) for im in images}`.
5. `pairs, selection_stats = select_cross_component_pairs(data, exifs, components, data.config["matching_components_exhaustive_cap"], data.config["matching_components_vlad_neighbors"])`.
6. `pairs_list = sorted(pairs)` (determinism). If empty, skip to step 10.
7. `matched = matching.match_images_with_pairs(data, {}, exifs, pairs_list)` — empty `config_override`: matching reads everything else from `data.config`, and GPS is only used by pair selection, which we bypass.
8. `merge_and_save_matches(data, matched)`.
9. `matching.clear_cache()`.
10. `components_after = image_components(build_match_graph(data, images, min_matches))`; log final count.
11. `write_report(data, components, components_after, pairs_list, num_matched_pairs, selection_stats, timer() - start)` where `num_matched_pairs = sum(1 for m in matched.values() if len(m) >= min_matches)` (0 if step 7 skipped).

### 1.2 `build_match_graph(data: DataSetBase, images: List[str], min_matches: int) -> nx.Graph`

- `graph = nx.Graph()`; `graph.add_nodes_from(images)` — images without a match file must appear as singleton components.
- `for (im1, im2), m in tracking.load_matches(data, images): if len(m) >= min_matches: graph.add_edge(im1, im2)`.

### 1.3 `image_components(graph: nx.Graph) -> List[Set[str]]`

```python
return sorted(
    (set(c) for c in nx.algorithms.components.connected_components(graph)),
    key=lambda c: (-len(c), min(c)),
)
```
Largest first; ties broken by smallest image name for determinism.

### 1.4 `select_cross_component_pairs(...) -> Tuple[Set[Tuple[str, str]], Dict[str, int]]`

```python
def select_cross_component_pairs(
    data: DataSetBase,
    exifs: Dict[str, Any],
    components: List[Set[str]],
    exhaustive_cap: int,
    vlad_neighbors: int,
    histograms: Optional[Dict[str, NDArray]] = None,
) -> Tuple[Set[Tuple[str, str]], Dict[str, int]]:
```
- `histograms` defaults to `{}`; it is a shared VLAD cache across component pairs, and injectable for tests.
- `stats = {"exhaustive_component_pairs": 0, "vlad_component_pairs": 0, "fallback_component_pairs": 0}`.
- For `comp_a, comp_b in combinations(components, 2)`:
  - If `len(comp_a) * len(comp_b) <= exhaustive_cap`: add `_exhaustive_pairs(comp_a, comp_b)`; bump `exhaustive_component_pairs`.
  - Else: `vlad_pairs = _vlad_pairs(data, exifs, comp_a, comp_b, vlad_neighbors, histograms)`. If non-empty, add them and bump `vlad_component_pairs`; if empty (VLAD unavailable), log a warning with both component sizes, add `_subsampled_pairs(comp_a, comp_b, exhaustive_cap)`, bump `fallback_component_pairs`.
- Single pass over all component-pair combinations, **no iteration rounds** — with the typical 2–5 components this connects every component to every other in one shot; the command can simply be re-run (it early-exits once connected).
- All pairs are canonicalized via `pairs_selection.sorted_pair`.

### 1.5 Helpers

```python
def _exhaustive_pairs(comp_a: Set[str], comp_b: Set[str]) -> Set[Tuple[str, str]]:
    return {pairs_selection.sorted_pair(a, b) for a in comp_a for b in comp_b}
```

`_vlad_pairs(data, exifs, comp_a, comp_b, vlad_neighbors, histograms) -> Set[Tuple[str, str]]`:
1. `smaller, larger = sorted((comp_a, comp_b), key=lambda c: (len(c), min(c)))`.
2. `need = (comp_a | comp_b) - set(histograms)`; in a `try/except Exception` (VLAD words file may be missing): `histograms.update(pairs_selection.vlad_histograms(need, data))`; on exception log a warning and `return set()`. (`vlad_histograms` silently drops images whose descriptors are missing.)
3. `cand_images = sorted(im for im in larger if im in histograms)`; if empty, `return set()`.
4. `results = [vlad.vlad_distances(im, cand_images, histograms) for im in sorted(smaller) if im in histograms]` — note `cand_images` is a list, as `vlad_distances` requires.
5. `scored = pairs_selection.construct_pairs(results, vlad_neighbors, exifs, enforce_other_cameras=False)`; `return set(scored.keys())` (keys are already `sorted_pair`s).

```python
def _subsampled_pairs(comp_a: Set[str], comp_b: Set[str], cap: int) -> Set[Tuple[str, str]]:
    """Deterministic even-stride subsampling of all cross pairs down to cap pairs."""
    all_pairs = sorted(pairs_selection.sorted_pair(a, b) for a in comp_a for b in comp_b)
    if len(all_pairs) <= cap:
        return set(all_pairs)
    stride = len(all_pairs) / float(cap)
    return {all_pairs[int(i * stride)] for i in range(cap)}
```

### 1.6 `merge_and_save_matches` — the critical function

```python
def merge_and_save_matches(
    data: DataSetBase,
    matched_pairs: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> None:
    """Merge new pairwise matches into the existing per-image match files.

    DataSet.save_matches overwrites the whole per-image file, so existing
    matches are loaded first and updated.
    """
    per_im1: Dict[str, Dict[str, NDArray]] = defaultdict(dict)
    for (im1, im2), m in matched_pairs.items():
        per_im1[im1][im2] = np.asarray(m, dtype=np.int32).reshape(-1, 2)
    for im1, new_matches in per_im1.items():
        existing: Dict[str, NDArray] = (
            data.load_matches(im1) if data.matches_exists(im1) else {}
        )
        existing.update(new_matches)
        data.save_matches(im1, existing)
```
Non-negotiable notes:
- **Do NOT use `matching.save_matches`** — it would destroy pre-existing matches (full-file overwrite).
- `np.asarray(m, dtype=np.int32).reshape(-1, 2)`: values from `match_images_with_pairs` may be lists of tuples; `DataSet.find_matches` does `arr[:, [1, 0]]`, which needs a 2-D array. `reshape(-1, 2)` turns an empty list into shape `(0, 2)`.
- Keys come back in the orientation we passed (canonical `sorted_pair`), so grouping by `im1` is deterministic. Saving zero-match pairs is intentional (records "pair was attempted", like `match_features`).
- A pre-existing sub-threshold entry stored in the reverse orientation (in `im2`'s file) may end up duplicated; harmless — `create_tracks`' union-find is idempotent. No dedup needed.

### 1.7 `write_report(...)`

```python
report: Dict[str, Any] = {
    "wall_time": wall_time,
    "num_components_before": len(components_before),
    "num_components_after": len(components_after),
    "component_sizes_before": [len(c) for c in components_before],
    "component_sizes_after": [len(c) for c in components_after],
    "num_candidate_pairs": len(pairs),
    "num_matched_pairs": num_matched_pairs,
    "pairs": pairs,
}
report.update(selection_stats)
data.save_report(io.json_dumps(report), "match_components.json")
```

---

## 2. New command: `opensfm/commands/match_components.py`

Exact mirror of `opensfm/commands/match_features.py`:

```python
# pyre-strict
import argparse

from opensfm.actions import match_components
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "match_components"
    help = "Match features across disconnected components of the matching graph"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        match_components.run_dataset(dataset)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
```

Registration in `opensfm/commands/__init__.py`:
1. Add `match_components,` to the import block (alphabetical: before `match_features`).
2. Insert `match_components,` in the `opensfm_commands` list **immediately after** `match_features,` (pipeline order).

No other CLI changes needed (`command_runner.py` consumes `opensfm_commands`).

---

## 3. Config: `opensfm/config.py`

Insert after `matching_use_opk: bool = True` (line 203), before the "Params for geometric estimation" section:

```python
    # Maximum number of cross-component image pairs per pair of components matched exhaustively by the match_components command. Above this, pairs are selected by VLAD similarity
    matching_components_exhaustive_cap: int = 100
    # Number of most similar images (by VLAD distance) each image of the smaller component is paired with in the other component, when above the exhaustive cap
    matching_components_vlad_neighbors: int = 5
```

Only these two params. The graph edge threshold reuses `robust_matching_min_match` — same semantics, and a separate knob could silently disagree with the matcher's own acceptance threshold.

---

## 4. Tests: `opensfm/test/test_match_components.py`

Header `# pyre-strict`, style: many small tests, one behavior each. Fixtures come from `opensfm/test/conftest.py`; the monkeypatching pattern for `SyntheticDataSet` is in `opensfm/test/test_matching.py:86-121` (`test_match_images`), including its `# pyre-fixme[8]` comments.

### 4.1 In-file fake dataset for unit tests

```python
class FakeMatchesDataSet:
    """Minimal stand-in exposing only the matches API used by the action."""
    def __init__(self, images: List[str], matches: Dict[str, Dict[str, NDArray]]) -> None:
        self.image_list = images
        self.matches = matches  # per-im1 store, mutated by save_matches
    def images(self): return self.image_list
    def matches_exists(self, image): return image in self.matches
    def load_matches(self, image):
        if image not in self.matches:
            raise IOError(image)  # tracking.load_matches catches IOError
        return dict(self.matches[image])  # copy, like unpickling does
    def save_matches(self, image, matches): self.matches[image] = matches
```
Helper: `def _m(n: int) -> NDArray: return np.zeros((n, 2), dtype=np.int32)`.

### 4.2 Unit tests

1. `test_build_match_graph_thresholds_edges` — images a,b,c; matches `a→{b: _m(25), c: _m(5)}`; with `min_matches=20` assert 3 nodes, edge (a,b) present, (a,c) absent.
2. `test_build_match_graph_no_match_files` — empty store; assert 3 nodes, 0 edges, and `image_components` returns 3 singletons.
3. `test_image_components_sorted_largest_first` — components {a,b,c} and {d,e}; assert order `[{a,b,c}, {d,e}]`.
4. `test_select_pairs_exhaustive_below_cap` — components `[{a,b,c}, {d,e}]`, `exhaustive_cap=100`; assert exactly the 6 sorted cross pairs and `stats == {"exhaustive_component_pairs": 1, "vlad_component_pairs": 0, "fallback_component_pairs": 0}`.
5. `test_select_pairs_vlad_above_cap` — components sizes 2 and 3, `exhaustive_cap=1`, `vlad_neighbors=1`; pass precomputed `histograms` (that's what the parameter is for) as 1-D `np.float32` vectors crafted so each smaller-component image has a known nearest neighbor (e.g. `a=[0.0], b=[10.0]` vs `c=[0.1], d=[10.1], e=[50.0]`); expect `{("a","c"), ("b","d")}` and `stats["vlad_component_pairs"] == 1`. Exercises the real `vlad.vlad_distances` path with zero dataset I/O. If the C++ binding rejects the 1-D float32 vectors, fall back to monkeypatching `match_components.vlad.vlad_distances` — but try the real call first.
6. `test_select_pairs_fallback_when_vlad_unavailable` — `exhaustive_cap=2`; `monkeypatch.setattr(match_components.pairs_selection, "vlad_histograms", lambda imgs, data: {})`; assert `stats["fallback_component_pairs"] == 1`, exactly 2 pairs, identical on repeated invocation.
7. `test_subsampled_pairs_capped_and_deterministic` — `_subsampled_pairs({a,b,c}, {d,e,f}, cap=4)`: length 4, subset of the 9 cross pairs, equal across calls; with `cap=100` all 9 returned.
8. `test_merge_save_preserves_existing_matches` — store `a→{b: _m(25)}`; `merge_and_save_matches(data, {("a","z"): [(0,1),(2,3)]})`; assert `data.matches["a"]` contains BOTH `b` (untouched, 25 rows) and `z`.
9. `test_merge_save_converts_lists_to_ndarray` — assert `data.matches["a"]["z"]` is `np.ndarray` with shape `(2, 2)`; an empty-list input yields shape `(0, 2)`.
10. `test_merge_save_new_image_without_existing_file` — merge `{("q","r"): [(1,1)]}` into a store without `q`; assert the `matches_exists`-False branch creates `{"r": ...}`.

### 4.3 Integration test (real matcher, synthetic scene)

`test_run_dataset_bridges_two_components(scene_synthetic)` — no ready-made disjoint fixture exists; build one:
1. `synthetic = synthetic_dataset.SyntheticDataSet(scene_synthetic.reconstruction, scene_synthetic.exifs, scene_synthetic.features, scene_synthetic.tracks_manager)`.
2. Keep it fast: `images = sorted(synthetic.images())[:16]`; `half_a, half_b = images[:8], images[8:]` (contiguous arcs of the circle scene overlap at the boundary, so exhaustive cross-matching is guaranteed to find bridges).
3. In-memory store `store: Dict[str, Dict[str, NDArray]] = {}`; monkeypatch on the instance (pattern of `test_matching.py:99-104`): `matches_exists`, `load_matches` (raise `IOError` if absent), `save_matches` writing into `store`; also capture reports via `save_report = lambda content, path: reports.__setitem__(path, content)`.
4. Populate two disjoint components with the real matcher, per half: `override = {"matching_gps_neighbors": 0, "matching_gps_distance": 0, "matching_time_neighbors": 2}`; for each half call `matching.match_images(synthetic, override, half, half)` and write results into `store` grouped per im1.
5. Config: `synthetic.config["matching_components_exhaustive_cap"] = 10_000` (force the exhaustive branch — synthetic data has no VLAD words guarantee).
6. Pre-assert: 2 components via `build_match_graph` + `image_components`.
7. Act: `match_components.run_dataset(synthetic)`.
8. Assert: components after == 1; every pre-existing `(im1, im2)` entry still present in `store` with an equal array (`np.array_equal`) — end-to-end non-destruction; `"match_components.json" in reports`.

### 4.4 Early-exit test

`test_run_dataset_early_exit_single_component` — `FakeMatchesDataSet` with chain a-b-c (all `_m(25)`), plus a `config` dict attribute (`robust_matching_min_match: 20` + the two new params), `save_report` capturing into a dict, `load_exif` that raises `AssertionError` (proves it is never called), and `save_matches` recording calls. Run `run_dataset`; assert no `save_matches` calls, report captured with `num_components_before == num_components_after == 1` and `num_candidate_pairs == 0`.

### 4.5 Registration test

`test_command_registered` — `from opensfm import commands`; assert `commands.match_components in commands.opensfm_commands`, positioned immediately after `commands.match_features`, and `commands.match_components.Command().name == "match_components"`.

---

## 5. Verification

1. `conda activate opensfm` (once), `export LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so`.
2. New tests: `pytest opensfm/test/test_match_components.py -v`.
3. Regression: `pytest opensfm/test/test_matching.py opensfm/test/test_pairs_selection.py opensfm/test/test_commands.py -v`.
4. Optional end-to-end on the shipped `data/lund` dataset (6 images): copy to a temp dir; in `config.yaml` set `matching_gps_distance: 0`, `matching_gps_neighbors: 0`, `matching_order_neighbors: 2`; run `extract_metadata`, `detect_features`, `match_features`; delete a middle image's `matches/<image>_matches.pkl.gz` to sever the chain; run `bin/opensfm match_components <dir>`; check `reports/match_components.json` shows `num_components_before >= 2` and `num_components_after: 1`; then `create_tracks` + `reconstruct` and confirm a single reconstruction in `reconstruction.json`.

## 6. Implementation order

1. `opensfm/config.py` — two new params (§3).
2. `opensfm/actions/match_components.py` (§1).
3. `opensfm/commands/match_components.py` + registration (§2).
4. `opensfm/test/test_match_components.py` (§4).
5. Verification (§5).

## 7. Files touched

- **New**: `opensfm/actions/match_components.py`, `opensfm/commands/match_components.py`, `opensfm/test/test_match_components.py`.
- **Edited**: `opensfm/config.py` (+2 params), `opensfm/commands/__init__.py` (+1 import, +1 list entry).
- **Reused, unchanged**: `opensfm/matching.py` (`match_images_with_pairs`, `clear_cache`), `opensfm/pairs_selection.py` (`sorted_pair`, `vlad_histograms`, `construct_pairs`), `opensfm/tracking.py` (`load_matches`), `opensfm/vlad.py` (`vlad_distances`).
