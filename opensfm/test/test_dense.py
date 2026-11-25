# pyre-strict
from pathlib import Path
from types import SimpleNamespace
import pytest

import numpy as np
from opensfm import dataset, dense, geo, io, pygeometry, pymap, types, pydense
from opensfm.synthetic_data import synthetic_examples, synthetic_scene
from opensfm.config import default_config
from opensfm.dense.equalize import estimate_image_corrections, apply_equalization
from opensfm.dense import crop, fusion
from opensfm.dataset import UndistortedDataSet
from opensfm.dense.dsm_ortho import (
    _component_elongation,
    _component_thickness,
    _dsm_footprint,
    _fill_holes_2pass,
    _inject_ortho_detail,
    _smooth_fill_components,
)
from opensfm.dense.fusion import (
    _assign_footprint_holes,
    _dilate_core_xy,
    _kdtree_chunk_labels,
    _pack_coarse_cells,
    _prescan_coarse_grid,
    _select_chunk_views,
    _select_chunk_views_greedy,
    _trim_sparse_ends,
)


def test_depthmap_to_ply() -> None:
    height, width = 2, 3

    camera = pygeometry.Camera.create_perspective(0.8, 0.0, 0.0)
    camera.id = "cam1"
    camera.height = height
    camera.width = width
    r = types.Reconstruction()
    r.add_camera(camera)
    shot = r.create_shot(
        "shot1",
        camera.id,
        pygeometry.Pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    )

    image = np.zeros((height, width, 3))
    depth = np.ones((height, width))

    ply = dense.depthmap_to_ply(shot, depth, image)
    assert len(ply.splitlines()) == 16


def _one_shot_recon(width: int = 32, height: int = 32) -> types.Reconstruction:
    camera = pygeometry.Camera.create_perspective(0.8, 0.0, 0.0)
    camera.id = "cam1"
    camera.width = width
    camera.height = height
    r = types.Reconstruction()
    r.add_camera(camera)
    r.create_shot(
        "s0",
        camera.id,
        pygeometry.Pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    )
    return r


def test_prescan_coarse_grid_buckets_views() -> None:
    # One view with all-valid depth projects into a non-empty set of coarse
    # cells, each recording the (only) contributing view index 0.
    r = _one_shot_recon()
    depth = np.full((32, 32), 5.0, dtype=np.float32)
    grid = _prescan_coarse_grid(
        None, ["s0"], r, voxel_size=2.0, coarse_factor=8,
        clean_cache={"s0": (depth, depth, None)},
    )
    assert len(grid) > 0
    assert all(views == {0} for views in grid.values())


def test_kdtree_chunk_labels_partitions_all_cells() -> None:
    # A line of 100 cells split at max_cells=10 → a clean partition: every cell
    # labelled, contiguous chunk ids, each chunk within budget.
    cells = np.array([(x, 0, 0) for x in range(100)], dtype=np.int32)
    labels = _kdtree_chunk_labels(cells, max_cells=10)
    sizes = np.bincount(labels)
    assert sizes.sum() == 100          # every cell assigned
    assert (sizes > 0).all()           # no empty chunk ids
    assert sizes.max() <= 10           # GPU/coherence budget respected


class _StubDataset:
    """Minimal duck type for ``UndistortedDataSet.list_batch_indices``."""

    def __init__(self, path: str) -> None:
        self._path = path
        self.io_handler = io.IoFilesystemDefault()

    def _depthmap_path(self) -> str:
        return self._path


def test_list_batch_indices_discovers_gaps_and_skips_debug(tmp_path: Path) -> None:
    # Chunks 0 and 2 produced a fused PLY (1 produced none → a gap); a _debug
    # variant and an unrelated file must NOT be counted as batches.
    for name in (
        "fused_batch_0000.ply",
        "fused_batch_0002.ply",
        "fused_batch_0001_debug.ply",
        "mesh_batch_0000.ply",
        "reconstruction.json",
    ):
        (tmp_path / name).write_text("")
    s = _StubDataset(str(tmp_path))
    fused = UndistortedDataSet.list_batch_indices(s, "fused_batch_", ".ply")
    # gap preserved, debug skipped
    assert fused == [0, 2]
    assert UndistortedDataSet.list_batch_indices(
        s, "mesh_batch_", ".ply") == [0]
    assert UndistortedDataSet.list_batch_indices(
        s, "dsm_ortho_batch_", ".npz") == []        # none present


def test_list_batch_indices_missing_folder_is_empty() -> None:
    s = _StubDataset("/no/such/depthmap/path")
    assert UndistortedDataSet.list_batch_indices(
        s, "fused_batch_", ".ply") == []


def test_assign_footprint_holes_assigns_enclosed_gap() -> None:
    # A 7x7 ring of occupied cells with a hollow 3x3 centre: the centre is an
    # enclosed interior hole and must be handed out for completion (single chunk).
    occ = [
        (x, y, 0) for x in range(7) for y in range(7)
        if not (1 < x < 5 and 1 < y < 5)
    ]
    cells = np.array(occ, dtype=np.int64)
    labels = np.zeros(len(cells), dtype=np.int64)
    holes = _assign_footprint_holes(cells, labels, 1, close_cells=0)
    expected = {(x, y) for x in range(2, 5) for y in range(2, 5)}
    assert holes[0] == expected


def test_trim_sparse_ends_cuts_only_extremities() -> None:
    # Dense core with thin tapering ends: only the sparse extremities are cut,
    # and an interior low-count dip is preserved.  Threshold is frac * max.
    counts = np.array([1, 1, 50, 60, 3, 55, 50, 1, 1], dtype=np.int64)
    # max = 60, frac=0.1 → thr=6; ends [1,1] are < 6, idx2 (50) stops the walk.
    assert _trim_sparse_ends(counts, 0.1) == (2, 7)  # interior dip (3) kept
    # frac=0 disables trimming.
    assert _trim_sparse_ends(counts, 0.0) == (0, len(counts))
    # all-zero axis → no trim (degenerate).
    assert _trim_sparse_ends(np.zeros(5, np.int64), 0.5) == (0, 5)


def test_assign_footprint_holes_trims_sparse_outlier() -> None:
    # A 7x7 ring with a hollow 3x3 centre, plus a lone outlier far in +x.  A
    # large close_cells WITHOUT trimming bridges the whole 33-cell gap to the
    # outlier (inventing a wide sparse fringe); WITH trimming the outlier column
    # is cut, so only the ring's genuine enclosed centre is completed.
    ring = [
        (x, y, 0) for x in range(7) for y in range(7)
        if not (1 < x < 5 and 1 < y < 5)
    ]
    cells = np.array(ring + [(40, 3, 0)], dtype=np.int64)
    labels = np.zeros(len(cells), dtype=np.int64)
    center = {(x, y) for x in range(2, 5) for y in range(2, 5)}

    holes = _assign_footprint_holes(
        cells, labels, 1, close_cells=64, trim_fraction=0.5
    )
    assert center <= holes[0]                       # enclosed centre completed
    assert all(x <= 6 for (x, _y) in holes[0])      # outlier fringe cut

    # Without trimming, the large closing DOES bridge into the sparse fringe.
    holes_notrim = _assign_footprint_holes(
        cells, labels, 1, close_cells=64, trim_fraction=0.0
    )
    assert any(x > 6 for (x, _y) in holes_notrim[0])


def test_assign_footprint_holes_none_when_solid() -> None:
    cells = np.array(
        [(x, y, 0) for x in range(4) for y in range(4)], dtype=np.int64
    )
    labels = np.zeros(len(cells), dtype=np.int64)
    holes = _assign_footprint_holes(cells, labels, 1, close_cells=0)
    assert holes[0] == set()


def test_assign_footprint_holes_partition_is_disjoint_and_complete() -> None:
    # Same ring, but two chunks split by x: every enclosed empty cell is owned
    # by exactly one chunk (disjoint), and together they cover the whole hole.
    occ = [
        (x, y, 0) for x in range(7) for y in range(7)
        if not (1 < x < 5 and 1 < y < 5)
    ]
    cells = np.array(occ, dtype=np.int64)
    labels = (cells[:, 0] > 3).astype(np.int64)  # 2 chunks by x median
    holes = _assign_footprint_holes(cells, labels, 2, close_cells=0)
    expected = {(x, y) for x in range(2, 5) for y in range(2, 5)}
    assert (holes[0] | holes[1]) == expected
    assert holes[0].isdisjoint(holes[1])


def test_assign_footprint_holes_hull_completes_boundary_void() -> None:
    # An L of occupied cells leaving the diagonal corner empty.  The void touches
    # the scene boundary, so the morphological enclosure (binary_fill_holes)
    # treats it as exterior and completes NOTHING — but a hull that spans the
    # whole bbox marks those boundary-adjacent columns completable.
    occ = [
        (x, y, 0) for x in range(6) for y in range(6)
        if x < 2 or y < 2  # L-shape: the 4x4 top-right block is empty
    ]
    cells = np.array(occ, dtype=np.int64)
    labels = np.zeros(len(cells), dtype=np.int64)

    # No hull → morphological path leaves the open corner uncompleted.
    morph = _assign_footprint_holes(cells, labels, 1, close_cells=0)
    assert morph[0] == set()

    # Hull covering the full bbox (coarse_size=1.0, so cell centres are at
    # x+0.5) → every empty in-bbox column is in-hull and completed.
    hull = np.array(
        [[-1.0, -1.0], [7.0, -1.0], [7.0, 7.0], [-1.0, 7.0]], dtype=np.float64
    )
    holes = _assign_footprint_holes(
        cells, labels, 1, close_cells=0, coarse_size=1.0, crop_hull=hull
    )
    empty = {
        (x, y) for x in range(6) for y in range(6) if not (x < 2 or y < 2)
    }
    assert holes[0] == empty


def test_assign_footprint_holes_hull_rejects_outside_corner() -> None:
    # Empty columns OUTSIDE the hull stay uncompleted (the no-data exterior must
    # remain transparent — no overgrowth past the survey outline).
    occ = [(x, y, 0) for x in range(6) for y in range(6) if x < 2 or y < 2]
    cells = np.array(occ, dtype=np.int64)
    labels = np.zeros(len(cells), dtype=np.int64)
    # Hull is a tight triangle hugging the occupied L; the far corner (x,y high)
    # is outside it.
    hull = np.array(
        [[-1.0, -1.0], [6.5, -1.0], [-1.0, 6.5]], dtype=np.float64
    )
    holes = _assign_footprint_holes(
        cells, labels, 1, close_cells=0, coarse_size=1.0, crop_hull=hull
    )
    # The far corner column (5, 5) -> centre (5.5, 5.5) is outside the triangle.
    assert (5, 5) not in holes[0]


def test_dsm_footprint_is_extensive_with_large_close() -> None:
    # The padded closing must be EXTENSIVE — never erode the footprint inward
    # from the array border (the binary_closing border_value pitfall, which the
    # default border_value=0 would collapse to empty at 32 iterations).
    valid = np.zeros((60, 60), bool)
    valid[3:57, 3:57] = True
    fp = _dsm_footprint(valid, close_iters=32)
    assert fp[valid].all()
    assert int(fp.sum()) >= int(valid.sum())


def test_dsm_footprint_does_not_balloon_convex_block() -> None:
    # A solid convex block has no concavity/hole to close, so the footprint must
    # equal the block at ANY close_iters — i.e. the closing does NOT balloon out
    # to the array rectangle (the border_value=1 failure this padded closing
    # replaces).  This is what makes a large dsm_footprint_close_cells safe.
    valid = np.zeros((60, 60), bool)
    valid[3:57, 3:57] = True
    for n in (8, 32, 256):
        fp = _dsm_footprint(valid, close_iters=n)
        assert int(fp.sum()) == int(valid.sum()), f"ballooned at n={n}"


def test_dsm_footprint_bridges_narrow_gap() -> None:
    # Two blocks separated by a 4-wide gap (<= 2*close_iters) are bridged — the
    # intended effect of close_iters — while the data extent is preserved.
    valid = np.zeros((40, 40), bool)
    valid[5:35, 8:19] = True    # left block (cols 8..18)
    valid[5:35, 23:34] = True   # right block (cols 23..33); gap cols 19..22
    closed = _dsm_footprint(valid, close_iters=4)
    # interior of the gap is filled (assert away from the open top/bottom ends)
    assert closed[10:30, 19:23].all()
    assert int(closed.sum()) > int(valid.sum())


def test_smooth_fill_components_harmonic_cg() -> None:
    # The colour inpaint solves Laplace's equation matrix-free with CG (no splu
    # fill-in).  Between two constant-colour borders the harmonic solution is a
    # linear ramp: the hole centre is ~the border average, the row is monotone,
    # values stay in [0, 255], and no cell is left NaN.
    from scipy import ndimage as ndi

    grid = np.zeros((30, 80, 3), np.float32)
    grid[:, :20] = (200.0, 100.0, 50.0)
    grid[:, 60:] = (50.0, 150.0, 250.0)
    valid = np.zeros((30, 80), bool)
    valid[:, :20] = True
    valid[:, 60:] = True
    hole = np.zeros((30, 80), bool)
    hole[:, 20:60] = True
    labels, n = ndi.label(hole)

    _smooth_fill_components(grid, labels, np.arange(1, n + 1), valid)

    assert np.isfinite(grid[hole]).all()
    assert grid.min() >= 0.0 and grid.max() <= 255.0
    centre = grid[15, 40]
    assert np.allclose(centre, (125.0, 125.0, 150.0), atol=8.0)
    row_r = grid[15, 20:60, 0]
    assert np.all(np.diff(row_r) <= 0.5)  # monotone non-increasing ramp


def test_fill_holes_2pass_mops_up_ringless_component() -> None:
    # A fillable component whose ring holds no valid sample makes the low_flat
    # filler bail (`ring_vals.size == 0`), leaving it NaN — the source of
    # coarse-grid void squares.  The guaranteed mop-up must close it from the
    # nearest filled cell, WITHOUT touching the genuine exterior (outside the
    # hole mask), which stays no-data so the ortho remains transparent there.
    g = np.full((40, 40), np.nan, np.float32)
    g[0:10, 0:10] = 5.0
    valid = ~np.isnan(g)
    hole_a = np.zeros((40, 40), bool)
    hole_a[0:14, 11:30] = True  # 266 cells, ring touches the valid block
    hole_b = np.zeros((40, 40), bool)
    hole_b[23:40, 23:40] = True  # 289 cells, ring is all exterior → bails
    hole_mask = hole_a | hole_b

    out, _extrap = _fill_holes_2pass(
        g, sample_valid=valid, hole_mask=hole_mask,
        small_area_max=256, diffuse_iters=8, kappa=0.5, dt=0.2,
        large_fill="low_flat", low_percentile=20.0, occlusion_drop=0.0,
    )

    # No fillable cell may survive as no-data.
    assert not (hole_mask & ~np.isfinite(out)).any()
    # The genuine exterior (NaN, outside the hole mask) is left untouched.
    exterior = (~valid) & (~hole_mask)
    assert np.isnan(out[exterior]).all()


def test_select_chunk_views_coverage_before_quality() -> None:
    # 6 chunk cells along x. Two high-weight views (A, B) redundantly cover cells
    # 0-3; a LOW-weight view C is the sole observer of cells 4,5.  Naive top-2 by
    # weight = [A, B] would leave 4,5 unfused (→ holes).  Coverage-first must
    # include C, then fill remaining budget with the best skipped views.
    cells = np.array([[i, 0, 0] for i in range(6)], dtype=np.int64)
    origin = cells.min(0)
    span = cells.max(0) - origin + 1
    cpk = np.sort(_pack_coarse_cells(cells, origin, span))

    def uk(idxs: list) -> np.ndarray:
        return np.array([[i, 0, 0] for i in idxs], dtype=np.int64)

    view_ukey = {
        "A": uk([0, 1, 2, 3]),
        "B": uk([0, 1, 2, 3]),
        "C": uk([4, 5]),
        "D": uk([0, 1]),
    }
    weighted = ["A", "B", "D", "C"]  # weight desc; C is worst

    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 2)
    assert "C" in views and n_unc == 0  # sole observer of 4,5 kept

    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 4)
    assert set(views) == {"A", "B", "C", "D"} and n_unc == 0
    assert views[0] == "A" and views[1] == "C"  # coverage picks come first

    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 1)
    assert views == ["A"] and n_unc == 2  # budget-limited, reported


def test_select_chunk_views_min_two_observers() -> None:
    # Ensure each cell gets >= 2 observers.  A covers all 4 cells; B covers 0,1;
    # C (low weight) is the only OTHER view of cells 2,3.  Naive top-2 = [A, B]
    # leaves cells 2,3 with a single observer (A) — which the SVO would reject.
    # min_obs=2 must pull in C so 2,3 reach two observers.
    cells = np.array([[i, 0, 0] for i in range(4)], dtype=np.int64)
    origin = cells.min(0)
    span = cells.max(0) - origin + 1
    cpk = np.sort(_pack_coarse_cells(cells, origin, span))

    def uk(idxs: list) -> np.ndarray:
        return np.array([[i, 0, 0] for i in idxs], dtype=np.int64)

    view_ukey = {
        "A": uk([0, 1, 2, 3]),
        "B": uk([0, 1]),
        "C": uk([2, 3]),
    }
    weighted = ["A", "B", "C"]  # C is worst

    # min_obs=1: A alone covers everything once.
    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 3, min_obs=1
    )
    assert views[0] == "A" and n_unc == 0

    # min_obs=2: A then B (0,1 → 2 obs) then C (2,3 → 2 obs); all satisfied.
    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 3, min_obs=2
    )
    assert set(views) == {"A", "B", "C"} and n_unc == 0

    # min_obs=2, budget=2: only A,B fit → cells 2,3 stay single-observer.
    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 2, min_obs=2
    )
    assert views == ["A", "B"] and n_unc == 2

    # A cell with a single observer in the whole candidate set cannot reach 2.
    view_ukey2 = {"A": uk([0, 1, 2, 3]), "B": uk([0, 1, 2])}  # cell 3: only A
    views, n_unc = _select_chunk_views(
        cpk, weighted[:2], view_ukey2, origin, span, 5, min_obs=2
    )
    assert n_unc == 1  # cell 3 can never get a 2nd observer


def test_select_chunk_views_greedy_cover_prefers_broad_views() -> None:
    # 6 chunk cells along x.  High-weight views A, B, C each cover a narrow
    # 2-cell strip; LOW-weight view D covers everything.  The single-pass
    # selection (_select_chunk_views) walks weight order and spends the whole
    # budget on A, B, C; greedy maximum-coverage (_select_chunk_views_greedy,
    # behind depthmap_fusion_adaptive_chunking) picks D first — one view
    # covers all.
    cells = np.array([[i, 0, 0] for i in range(6)], dtype=np.int64)
    origin = cells.min(0)
    span = cells.max(0) - origin + 1
    cpk = np.sort(_pack_coarse_cells(cells, origin, span))

    def uk(idxs: list) -> np.ndarray:
        return np.array([[i, 0, 0] for i in idxs], dtype=np.int64)

    view_ukey = {
        "A": uk([0, 1]),
        "B": uk([2, 3]),
        "C": uk([4, 5]),
        "D": uk([0, 1, 2, 3, 4, 5]),
    }
    weighted = ["A", "B", "C", "D"]  # D is worst

    # Budget 1: single pass keeps A (4 cells unfused); greedy takes D (0 unfused).
    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 1)
    assert views == ["A"] and n_unc == 4
    views, n_unc = _select_chunk_views_greedy(
        cpk, weighted, view_ukey, origin, span, 1
    )
    assert views == ["D"] and n_unc == 0

    # Budget 3: greedy covers with D, then fills with the best skipped views.
    views, n_unc = _select_chunk_views_greedy(
        cpk, weighted, view_ukey, origin, span, 3
    )
    assert views == ["D", "A", "B"] and n_unc == 0

    # min_obs=2 with budget 3: D + A + B → cells 4,5 stay single-observer.
    views, n_unc = _select_chunk_views_greedy(
        cpk, weighted, view_ukey, origin, span, 3, min_obs=2
    )
    assert views == ["D", "A", "B"] and n_unc == 2


def test_adaptive_chunking_covers_cells_the_legacy_budget_left_as_holes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # THE hole scenario of depthmap_fusion_adaptive_chunking: 6 coarse cells
    # in a row, 3 views (A, B, C) each observing a 2-cell strip, but a
    # per-chunk budget of max_views=2.  Every cell HAS an observer, so no
    # cell is data-limited — yet the legacy fixed chunk can only afford two
    # of the three views, leaving one strip with no selected observer: a
    # budget-limited hole.  The adaptive builder must bisect the chunk so
    # each half fits the budget and nothing is left unfused.
    def cells_of(*xs: int) -> np.ndarray:
        return np.array([[x, 0, 0] for x in xs], dtype=np.int64)

    prescan = {
        "A": cells_of(0, 1),
        "B": cells_of(2, 3),
        "C": cells_of(4, 5),
    }
    observers = {
        sid: {tuple(int(v) for v in row) for row in cells}
        for sid, cells in prescan.items()
    }

    def fake_prescan(data, sid, shot, inv_coarse, depth_factor, subsample):
        cells = prescan[sid]
        return sid, cells, np.ones(len(cells))

    monkeypatch.setattr(fusion, "_prescan_view_weights", fake_prescan)

    data = SimpleNamespace(config={"processes": 1})
    reconstruction = SimpleNamespace(shots={sid: None for sid in prescan})
    kwargs = dict(
        voxel_size=1.0, coarse_factor=16, max_chunk_cells=100, max_views=2,
        depth_factor=2.0, footprint_close=1, min_observers=1,
    )

    def n_holey_cells(units) -> int:
        # Cells with fewer than min_observers SELECTED views — unfused holes.
        return sum(
            1
            for unit in units
            for cell in unit.cells
            if sum(cell in observers[v] for v in unit.views) < 1
        )

    # Legacy: one chunk, budget 2 → one strip (2 cells) left without a view.
    legacy_units, _ = fusion._build_global_chunks(
        data, reconstruction, list(prescan), **kwargs
    )
    assert len(legacy_units) == 1
    assert n_holey_cells(legacy_units) == 2

    # Adaptive: the chunk splits until coverage fits the budget — no holes.
    adaptive_units, _ = fusion._build_global_chunks_adaptive(
        data, reconstruction, list(prescan), **kwargs
    )
    assert len(adaptive_units) == 2
    assert n_holey_cells(adaptive_units) == 0
    # ...and the finer partition stays disjoint and complete.
    all_cells = [cell for unit in adaptive_units for cell in unit.cells]
    assert len(all_cells) == len(set(all_cells)) == 6


def test_dilate_core_xy_grows_into_observed_neighbours() -> None:
    # A 3x3 core plus observed cells one ring out (same Z) and a far cell.
    # margin=1 grows the core by one XY ring (into the observed neighbours) but
    # never reaches the far cell, and ignores Z (2.5D plan-view dilation).
    core = {(x, y, 0) for x in range(3) for y in range(3)}
    ring = {(-1, 1, 0), (3, 1, 5), (1, -1, 0)}  # adjacent, varied Z
    far = {(20, 20, 0)}
    observed = core | ring | far
    out = _dilate_core_xy(observed, core, margin=1)
    assert core <= out                 # core always kept
    assert ring <= out                 # one-ring neighbours pulled in
    assert far.isdisjoint(out)         # far cell never reached
    assert out <= observed             # only ever observed cells


def test_dilate_core_xy_zero_margin_is_core_only() -> None:
    core = {(0, 0, 0), (1, 0, 0)}
    observed = core | {(2, 0, 0)}
    assert _dilate_core_xy(observed, core, margin=0) == core


def test_kdtree_chunk_labels_separates_far_blobs() -> None:
    # Two compact blobs far apart; with a budget that fits each, the split keeps
    # each blob whole and in its own chunk (spatially coherent, not scattered).
    a = np.array([(x, y, 0) for x in range(5)
                 for y in range(5)], dtype=np.int32)
    b = a + np.array([1000, 0, 0], dtype=np.int32)
    cells = np.vstack([a, b])
    labels = _kdtree_chunk_labels(cells, max_cells=25)
    assert len(set(labels[:25].tolist())) == 1   # blob A is one chunk
    assert len(set(labels[25:].tolist())) == 1   # blob B is one chunk
    assert labels[0] != labels[25]               # different chunks


# ── Radiometric equalization (dense.equalize) ─────────────────────────────


def _synthetic_equalize_scene(
    n_shots: int, n_tracks: int, gains: "np.ndarray", vign: "np.ndarray",
    width: int = 400, height: int = 300, seed: int = 0,
) -> "tuple[types.Reconstruction, pymap.TracksManager, np.ndarray]":
    """Build a reconstruction + tracks where each (shot, track) colour is exactly
    ``albedo * gain_shot * vignette_shot(rho)`` for known per-shot gains/vignette
    and known per-track albedos.  Returns (reconstruction, tracks, rmax)."""
    rng = np.random.default_rng(seed)
    rec = types.Reconstruction()
    cam = pygeometry.Camera.create_perspective(0.8, 0.0, 0.0)
    cam.id = "cam"
    cam.width = width
    cam.height = height
    rec.add_camera(cam)
    pose = pygeometry.Pose(np.zeros(3), np.zeros(3))
    for i in range(n_shots):
        rec.create_shot(f"shot_{i}", cam.id, pose)
    rmax = 0.5 * np.sqrt(width ** 2 + height ** 2) / max(width, height)

    albedos = rng.uniform(40.0, 200.0, size=(n_tracks, 3))
    tm = pymap.TracksManager()
    for t in range(n_tracks):
        k = min(int(rng.integers(3, 6)), n_shots)
        shots = rng.choice(n_shots, size=k, replace=False)
        for i in shots:
            rho = float(rng.uniform(0.0, 1.0))
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            pt = rho * rmax * np.array([np.cos(theta), np.sin(theta)])
            v = np.exp(vign[i, :, 0] * rho ** 2 + vign[i, :, 1] * rho ** 4)
            col = np.clip(albedos[t] * gains[i] * v, 1.0, 254.0)
            ci = np.round(col).astype(int)
            obs = pymap.Observation(
                float(pt[0]), float(pt[1]), 0.01,
                int(ci[0]), int(ci[1]), int(ci[2]), t)
            tm.add_observation(f"shot_{i}", str(t), obs)
    return rec, tm, rmax


def test_equalize_recovers_per_image_gains() -> None:
    # Known per-shot per-channel gains + vignetting baked into track colours must
    # be recovered (up to the global scale gauge, so compare gain RATIOS between
    # images, which are gauge-invariant).
    n_shots = 6
    rng = np.random.default_rng(1)
    gains = rng.uniform(0.7, 1.4, size=(n_shots, 3))
    vign = np.zeros((n_shots, 3, 2))  # no vignetting in this test
    rec, tm, _ = _synthetic_equalize_scene(n_shots, 400, gains, vign)

    corr = estimate_image_corrections(tm, rec, default_config())

    g = np.array([corr[f"shot_{i}"]["gain"] for i in range(n_shots)])
    # Gauge-invariant check: every pairwise gain ratio matches the truth.
    for c in range(3):
        for i in range(n_shots):
            for j in range(i + 1, n_shots):
                got = g[i, c] / g[j, c]
                want = gains[i, c] / gains[j, c]
                assert abs(got - want) < 0.05 * want, (
                    f"gain ratio ch{c} {i}/{j}: {got:.3f} vs {want:.3f}")
    # Gauge: geometric mean of recovered gains is ~1 per channel.
    assert np.allclose(np.exp(np.log(g).mean(axis=0)), 1.0, atol=0.02)


def test_equalize_recovers_vignetting() -> None:
    # A radial darkening (negative log-coeffs) baked per image must be recovered
    # in the vignette coefficients (these carry no gauge ambiguity).
    n_shots = 4
    gains = np.ones((n_shots, 3))
    rng = np.random.default_rng(2)
    vign = np.zeros((n_shots, 3, 2))
    vign[:, :, 0] = rng.uniform(-0.35, -0.1, size=(n_shots, 3))  # rho^2 term
    rec, tm, _ = _synthetic_equalize_scene(n_shots, 1000, gains, vign)

    # Small ridge so the test checks unbiased recovery, not the regulariser.
    cfg = default_config()
    cfg["equalize_vignette_regularization"] = 0.02

    corr = estimate_image_corrections(tm, rec, cfg)

    for i in range(n_shots):
        got = np.array(corr[f"shot_{i}"]["vignette"])  # (3, order)
        assert np.allclose(got[:, 0], vign[i, :, 0], atol=0.05), (
            f"shot {i} vignette rho^2: {got[:, 0]} vs {vign[i, :, 0]}")
    # And edge darkening really is recovered (V(1) < 1).
    g0 = np.array(corr["shot_0"]["vignette"])
    assert float(np.exp(g0[:, 0].sum())) < 1.0


def test_equalize_identity_when_no_mismatch() -> None:
    # If every image already agrees (unit gain, no vignette), the correction is
    # ~identity (gains ~1, vignette ~0).
    n_shots = 5
    gains = np.ones((n_shots, 3))
    vign = np.zeros((n_shots, 3, 2))
    rec, tm, _ = _synthetic_equalize_scene(n_shots, 300, gains, vign)

    corr = estimate_image_corrections(tm, rec, default_config())

    for i in range(n_shots):
        assert np.allclose(corr[f"shot_{i}"]["gain"], 1.0, atol=0.03)
        assert np.allclose(corr[f"shot_{i}"]["vignette"], 0.0, atol=0.03)


def test_apply_equalization_flattens_known_falloff() -> None:
    # Forward-apply a known gain + radial vignetting to a flat-albedo image, then
    # apply_equalization with the matching correction must recover the flat
    # albedo (gain divided out, vignette flattened across the frame).

    w, h = 120, 90
    albedo = 130.0
    corr = {
        "gain": [1.25, 1.0, 0.8],
        "vignette": [[-0.30, 0.0], [-0.20, 0.0], [-0.12, 0.0]],
        "pp": [0.0, 0.0],
        "rmax": 0.5 * np.sqrt(w * w + h * h) / max(w, h),
    }
    # Build the "measured" image = albedo * gain * V(rho) (what the camera saw).
    size = float(max(w, h))
    xs = (np.arange(w) + 0.5 - 0.5 * w) / size
    ys = (np.arange(h) + 0.5 - 0.5 * h) / size
    rho2 = (xs[None, :] ** 2 + ys[:, None] ** 2) / corr["rmax"] ** 2
    measured = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        v = np.exp(corr["vignette"][c][0] * rho2)
        measured[..., c] = albedo * corr["gain"][c] * v
    measured_u8 = np.clip(measured, 0, 255).astype(np.uint8)

    out = apply_equalization(measured_u8, corr)

    assert out.dtype == np.uint8
    # Flat and centred on the albedo (small spatial spread, no residual falloff).
    for c in range(3):
        assert abs(float(out[..., c].mean()) - albedo) < 3.0
        assert float(out[..., c].std()) < 3.0
    # Vignetting really was undone: corner is no longer darker than centre.
    centre = out[h // 2, w // 2].astype(int)
    corner = out[1, 1].astype(int)
    assert abs(int(centre[0]) - int(corner[0])) <= 3


def test_apply_equalization_preserves_highlights() -> None:
    # A brightening gain (gain < 1) lifts midtones AND already-bright pixels; the
    # highlight roll-off must keep the bright region ~as-is (no flat 255 burn)
    # while the midtone still gets the full correction.
    corr = {
        "gain": [0.7, 0.7, 0.7],      # factor ~1.43 (brightening)
        "vignette": [[0.0], [0.0], [0.0]],
        "pp": [0.0, 0.0],
        "rmax": 0.64,
    }
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :10] = 120                 # midtone (well below knee)
    img[:, 10:] = 248                 # near-white (would clip if multiplied)

    out = apply_equalization(img, corr, highlight_knee=235.0)
    mid = float(out[:, :10].mean())
    bright = float(out[:, 10:].mean())
    # midtone fully corrected (~171)
    assert abs(mid - 120 / 0.7) < 4.0
    assert bright < 252.0                       # NOT burned to flat white
    assert abs(bright - 248.0) < 6.0            # ~kept its original value

    # Disabling the roll-off (knee=255) burns the bright region to white.
    burned = apply_equalization(img, corr, highlight_knee=255.0)
    assert float(burned[:, 10:].mean()) > 254.0


# ── Ortho detail injection (dsm_ortho._inject_ortho_detail) ───────────────


def test_inject_ortho_detail_adds_texture_not_exposure_step() -> None:
    # The sharp source carries BOTH high-frequency texture and a low-frequency
    # exposure step (left half brighter — what a single-source switch looks
    # like).  Detail injection must add the texture back onto the flat blend
    # while the exposure step is high-passed away (does NOT appear) — that is the
    # whole point: sharper, but no patchiness.
    h, w = 64, 64
    cfg = {"ortho_detail_sigma": 2.0, "ortho_detail_strength": 1.0}

    blend = np.full((h, w, 3), 100, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (20 * ((xx + yy) % 2 == 0) - 10).astype(np.int16)  # ±10 hi-freq
    # lo-freq step
    step = np.where(xx < w // 2, 40, 0).astype(np.int16)
    sharp1 = np.clip(100 + checker + step, 0, 255).astype(np.uint8)
    sharp = np.repeat(sharp1[..., None], 3, axis=2)  # (h, w, 3)
    baked = np.ones((h, w), bool)

    out = _inject_ortho_detail(blend, sharp, baked, cfg).astype(np.int16)

    # Texture injected: the checkerboard variance shows up in the output.
    assert out[8:56, 8:56].std() > 5.0
    # Exposure step removed: interior left vs right means match (no 40 step),
    # sampling away from the central boundary where the step's own edge lives.
    left = out[8:56, 8:24].mean()
    right = out[8:56, 40:56].mean()
    assert abs(
        left - right) < 4.0, f"exposure step leaked: {left:.1f} vs {right:.1f}"
    # And the base level still comes from the blend (~100), not sharp's 140.
    assert abs(left - 100) < 4.0


def test_inject_ortho_detail_leaves_unbaked_cells_untouched() -> None:
    # Cells with no real bake (baked_mask False) must keep the blend exactly —
    # no detail is invented where there is no sharp source.
    h, w = 32, 32
    cfg = {"ortho_detail_sigma": 2.0, "ortho_detail_strength": 1.0}
    rng = np.random.default_rng(0)
    blend = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    sharp = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    baked = np.ones((h, w), bool)
    baked[:8, :] = False  # top strip unbaked

    out = _inject_ortho_detail(blend, sharp, baked, cfg)
    assert np.array_equal(out[:8, :], blend[:8, :])


def test_inject_ortho_detail_disabled_is_identity() -> None:
    blend = np.full((10, 10, 3), 120, np.uint8)
    sharp = np.zeros((10, 10, 3), np.uint8)
    baked = np.ones((10, 10), bool)
    out = _inject_ortho_detail(
        blend, sharp, baked,
        {"ortho_detail_sigma": 0.0, "ortho_detail_strength": 1.0})
    assert np.array_equal(out, blend)


def test_bake_colors_sharp_returns_sharpest_inlier_color() -> None:
    # GPU end-to-end: three nadir cameras at different heights; the SHARPEST
    # (lowest camera, A) carries a slightly distinct but in-gate colour.
    # bake_colors(with_sharp=True) must return A's RAW colour as the sharp source
    # (not the blended/averaged colour).

    if not pydense.SVOFuser.is_gpu_available():
        pytest.skip("no GPU/OpenCL device for the SVO bake")

    h = w = 64
    f, cx, cy = 80.0, w / 2.0, h / 2.0
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    color_a = (160, 90, 40)   # sharpest view (height 8)
    color_bc = (150, 95, 45)  # the other two (in-gate, close to A)

    fuser = pydense.SVOFuser()
    fuser.set_voxel_size(0.05)
    fuser.set_trunc_factor(4.0)
    fuser.set_min_weight(0.0)
    fuser.set_min_count(1)
    fuser.set_device(0)
    fuser.set_bbox(np.array([-2, -2, -1], np.float32),
                   np.array([2, 2, 1], np.float32))
    for height, col in [(8.0, color_a), (10.0, color_bc), (12.0, color_bc)]:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        t = -R @ np.array([0.0, 0.0, height])
        depth = np.full((h, w), height, dtype=np.float32)
        normal = np.zeros((h, w, 3), np.float32)
        normal[..., 2] = 1.0
        cimg = np.zeros((h, w, 3), np.uint8)
        cimg[:] = col
        mask = np.full((h, w), 255, np.uint8)
        fuser.add_view(K, R, t, depth, normal, cimg, mask)
    fuser.count_voxels()
    fuser.fuse_only()

    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    nrm = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    out = fuser.bake_colors(pts, nrm, n_final=3, irls_iters=3, with_sharp=True)

    assert isinstance(out, tuple) and len(out) == 2
    colors, sharp = out
    assert sharp.shape == (1, 3)
    # Sharp = A's RAW colour (sharpest inlier), NOT the blend of A/B/C.
    assert np.all(np.abs(sharp[0].astype(int) - color_a) <= 2)
    # The blend, by contrast, is pulled toward the B/C majority (not exactly A).
    assert not np.array_equal(colors[0], np.array(color_a, np.uint8))


# ───────────────────────── SfM-hull crop ─────────────────────────


def test_hull_contains_square() -> None:
    # Axis-aligned 2×2 square, counter-clockwise.
    hull = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    x = np.array([1.0, 3.0, -0.5, 0.5])
    y = np.array([1.0, 1.0, 1.0, 0.5])
    inside = crop.hull_contains(hull, x, y)
    assert inside.tolist() == [True, False, False, True]


def test_hull_contains_broadcasts_to_grid() -> None:
    hull = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    xs = np.array([-1.0, 1.0, 3.0, 5.0])
    ys = np.array([-1.0, 2.0, 5.0])
    inside = crop.hull_contains(hull, xs[None, :], ys[:, None])
    assert inside.shape == (3, 4)
    # Only (x=1 or 3, y=2) land inside the square.
    assert inside[1, 1] and inside[1, 2]
    assert not inside[0, 0] and not inside[2, 3]


def test_crop_hull_from_points_trims_outliers() -> None:
    rng = np.random.default_rng(0)
    # Dense ground cluster in [-1, 1]² with a little height noise.
    square = rng.uniform(-1.0, 1.0, size=(4000, 2))
    z = rng.uniform(0.0, 0.1, size=(4000, 1))
    inliers = np.hstack([square, z])
    # A few far stray points (well under the 2% trim fraction).
    outliers = np.array([[100.0, 0.0, 0.0], [0.0, -80.0, 0.0],
                         [60.0, 60.0, 0.0]])
    pts = np.vstack([inliers, outliers])

    hull = crop.crop_hull_from_points(pts, percentile=2.0)
    assert hull is not None
    # Outliers are trimmed → outside the hull.
    assert not crop.hull_contains(
        hull, outliers[:, 0], outliers[:, 1]).any()
    # The cluster centre is inside; the hull stays tight (~[-1, 1]).
    assert crop.hull_contains(hull, np.array([0.0]), np.array([0.0]))[0]
    assert hull[:, 0].max() < 1.5 and hull[:, 1].max() < 1.5


def test_crop_hull_follows_oriented_frame() -> None:
    # A long, thin strip rotated 30°: the percentile trim must act along the
    # strip's own axes, not the world axes, so the hull keeps the full length.
    rng = np.random.default_rng(1)
    u = rng.uniform(-10.0, 10.0, size=(5000, 1))   # along the strip
    v = rng.uniform(-0.5, 0.5, size=(5000, 1))     # across the strip
    a = np.deg2rad(30.0)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = np.hstack([u, v]) @ rot.T
    pts = np.hstack([xy, np.zeros((len(xy), 1))])

    hull = crop.crop_hull_from_points(pts, percentile=2.0)
    assert hull is not None
    # Endpoints near the strip extremities survive the trim (long axis kept).
    end = np.array([9.0, 0.0]) @ rot.T
    assert crop.hull_contains(
        hull, np.array([end[0]]), np.array([end[1]]))[0]


def test_crop_hull_too_few_points_returns_none() -> None:
    assert crop.crop_hull_from_points(
        np.zeros((2, 3)), percentile=2.0) is None
    # Collinear points are degenerate → no crop.
    line = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    assert crop.crop_hull_from_points(line, percentile=2.0) is None


# ────────────── DSM hole-fill routing by hole shape (elongation) ─────────────


def test_fill_holes_2pass_area_split_when_no_aspect() -> None:
    # max_aspect = 0 (off) keeps the legacy AREA split: a large hole is sent to
    # low_flat (filled flat at the single border altitude), not diffused.
    grid = np.full((30, 30), 5.0, np.float32)
    hole = np.zeros((30, 30), bool)
    hole[5:25, 5:25] = True                   # 400-cell hole
    grid[hole] = np.nan

    out, _extrap = _fill_holes_2pass(
        grid, sample_valid=~np.isnan(grid), hole_mask=hole,
        small_area_max=16, diffuse_iters=8, kappa=0.5, dt=0.2,
        large_fill="low_flat", low_percentile=20.0, max_aspect=0.0,
    )
    assert np.allclose(out[hole], 5.0, atol=1e-3)


def test_component_elongation_compact_vs_elongated() -> None:
    from scipy import ndimage

    holes = np.zeros((40, 40), bool)
    holes[5:11, 5:11] = True            # 6x6 compact blob   → aspect ~1
    holes[20:22, 5:35] = True           # 2x30 long strip    → aspect large
    labels, n = ndimage.label(holes)
    elong = _component_elongation(labels, n)
    # Identify the two components by their bbox.
    blob = labels[7, 7]
    strip = labels[20, 20]
    assert elong[blob] < 2.0
    assert elong[strip] > 8.0


def test_component_elongation_orientation_invariant() -> None:
    from scipy import ndimage

    # A diagonal strip (square bbox) must still read as elongated.
    holes = np.zeros((40, 40), bool)
    for i in range(30):
        holes[5 + i, 5 + i] = True
        holes[5 + i, 6 + i] = True
    labels, n = ndimage.label(holes)
    elong = _component_elongation(labels, n)
    assert elong[1] > 6.0


def test_fill_holes_2pass_compact_step_still_flat() -> None:
    # With the elongation guard active, a COMPACT stepped hole is still flat.
    grid = np.zeros((24, 24), np.float32)
    grid[:, 12:] = 10.0
    hole = np.zeros((24, 24), bool)
    hole[9:15, 9:15] = True              # compact, straddles the step
    grid[hole] = np.nan
    out, _e = _fill_holes_2pass(
        grid, sample_valid=~np.isnan(grid), hole_mask=hole,
        small_area_max=4, diffuse_iters=8, kappa=0.5, dt=0.2,
        large_fill="low_flat", low_percentile=20.0, max_aspect=4.0,
    )
    assert out[hole].max() < 3.0          # flat at ground, not slanted to roof


def test_component_thickness_blob_vs_feather() -> None:
    from scipy import ndimage

    holes = np.zeros((60, 60), bool)
    holes[5:25, 5:25] = True                 # 20x20 chunky blob → thick
    # A wiggly, BRANCHING thin tendril (1-2 px wide): near-round bbox (low
    # aspect) but thin everywhere.
    for i in range(30):
        r = 35 + int(3 * np.sin(i / 3.0))
        holes[r, 10 + i] = True
        holes[r + 1, 10 + i] = True
    holes[40:55, 25] = True                  # a branch off the tendril
    labels, n = ndimage.label(holes)
    thick = _component_thickness(holes, labels, n)
    blob = labels[15, 15]
    feather = labels[35, 12]
    assert thick[blob] > 6.0                 # big inscribed disk
    assert thick[feather] < 2.5              # thin everywhere


def test_component_thickness_feather_has_low_aspect() -> None:
    # The branching feather that thickness catches is NOT caught by aspect — its
    # bounding box is near-round.  This is exactly why the thickness gate exists.
    from scipy import ndimage

    holes = np.zeros((60, 60), bool)
    for i in range(30):
        r = 30 + int(10 * np.sin(i / 4.0))   # wiggles across 20 rows
        holes[r, 10 + i] = True
        holes[r + 1, 10 + i] = True
    labels, n = ndimage.label(holes)
    aspect = _component_elongation(labels, n)
    thick = _component_thickness(holes, labels, n)
    # near-round bbox → "looks compact"
    assert aspect[1] < 4.0
    assert thick[1] < 2.5                     # but thin → thickness rejects it


def test_fill_holes_2pass_thickness_diffuses_feather() -> None:
    # A thin feather on a slope must be diffused (follows the slope), NOT flat-
    # filled into a canyon — even though its aspect ratio is modest.
    H, W = 50, 50
    grid = np.tile(np.linspace(0.0, 30.0, W).astype(np.float32), (H, 1))
    feather = np.zeros((H, W), bool)
    for i in range(40):
        r = 25 + int(8 * np.sin(i / 5.0))
        feather[r, 5 + i] = True
        feather[r + 1, 5 + i] = True
    grid_in = grid.copy()
    grid_in[feather] = np.nan

    out, _e = _fill_holes_2pass(
        grid_in, sample_valid=~np.isnan(grid_in), hole_mask=feather,
        small_area_max=4, diffuse_iters=16, kappa=0.5, dt=0.2,
        large_fill="low_flat", low_percentile=20.0,
        max_aspect=8.0, min_thickness=3.0,
    )
    filled = out[feather]
    assert np.all(np.isfinite(filled))
    # Follows the slope (spread), not collapsed to one low altitude (canyon).
    assert filled.max() - filled.min() > 10.0


def test_run_clustering_skips_shots_missing_from_tracks(tmp_path: Path) -> None:
    # A panorama cube face that sees no track is in the undistorted
    # reconstruction but missing from the undistorted tracks manager.
    np.random.seed(42)
    reconstruction = synthetic_examples.synthetic_cube_scene().get_reconstruction()
    tracks_manager = synthetic_scene.SyntheticInputData(
        reconstruction,
        geo.TopocentricConverter(47.0, 6.0, 0),
        40, 0.0, 0.0, 0.0, (0.0, 0.0), False,
    ).tracks_manager
    template_shot = next(iter(reconstruction.shots.values()))
    face_without_tracks = "pano_perspective_view_top.jpg"
    reconstruction.create_shot(
        face_without_tracks, template_shot.camera.id, template_shot.pose
    )
    (tmp_path / "image_list.txt").write_text("\n".join(reconstruction.shots))
    udata = UndistortedDataSet(
        dataset.DataSet(str(tmp_path)), str(tmp_path / "undistorted")
    )

    dense.run_clustering(udata, tracks_manager, reconstruction)

    assert face_without_tracks not in udata.load_depth_ranges()
