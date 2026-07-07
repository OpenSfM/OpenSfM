# pyre-strict
import logging
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from opensfm import io, matching, pairs_selection, tracking, vlad
from opensfm.dataset_base import DataSetBase

logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(data: DataSetBase) -> None:
    """Match features between images belonging to different connected
    components of the matching graph."""

    start = timer()
    images = data.images()
    min_matches: int = data.config["robust_matching_min_match"]

    components = image_components(build_match_graph(data, images, min_matches))
    logger.info("Found %d connected components in the matching graph", len(components))

    if len(components) <= 1:
        logger.info("Matching graph already connected, nothing to do")
        write_report(data, components, components, [], 0, {}, timer() - start)
        return

    exifs = {im: data.load_exif(im) for im in images}
    pairs, selection_stats = select_cross_component_pairs(
        data,
        exifs,
        components,
        data.config["matching_components_exhaustive_cap"],
        data.config["matching_components_vlad_neighbors"],
    )
    pairs_list = sorted(pairs)

    num_matched_pairs = 0
    if pairs_list:
        matched = matching.match_images_with_pairs(data, {}, exifs, pairs_list)
        merge_and_save_matches(data, matched)
        matching.clear_cache()
        num_matched_pairs = sum(1 for m in matched.values() if len(m) >= min_matches)

    components_after = image_components(build_match_graph(data, images, min_matches))
    logger.info(
        "Matching graph has %d connected components after merging",
        len(components_after),
    )

    write_report(
        data,
        components,
        components_after,
        pairs_list,
        num_matched_pairs,
        selection_stats,
        timer() - start,
    )


def build_match_graph(
    data: DataSetBase, images: List[str], min_matches: int
) -> nx.Graph:
    """Build a graph with one node per image and an edge for each saved pair
    having at least min_matches matches."""
    graph = nx.Graph()
    graph.add_nodes_from(images)
    for (im1, im2), m in tracking.load_matches(data, images):
        if len(m) >= min_matches:
            graph.add_edge(im1, im2)
    return graph


def image_components(graph: nx.Graph) -> List[Set[str]]:
    """Connected components as sets of image names, largest first (ties broken
    by smallest image name for determinism)."""
    return sorted(
        (set(c) for c in nx.algorithms.components.connected_components(graph)),
        key=lambda c: (-len(c), min(c)),
    )


def select_cross_component_pairs(
    data: DataSetBase,
    exifs: Dict[str, Any],
    components: List[Set[str]],
    exhaustive_cap: int,
    vlad_neighbors: int,
    histograms: Optional[Dict[str, NDArray]] = None,
) -> Tuple[Set[Tuple[str, str]], Dict[str, int]]:
    """Select image pairs across every pair of components.

    Component pairs with at most exhaustive_cap candidate pairs are matched
    exhaustively; larger ones are pruned with VLAD similarity (top
    vlad_neighbors candidates per image of the smaller component). Falls back
    to deterministic subsampling when VLAD histograms are unavailable.
    Pre-computed VLAD histograms can be passed in (mainly for tests); the
    dict is reused as a cache across component pairs.
    """
    if histograms is None:
        histograms = {}

    pairs: Set[Tuple[str, str]] = set()
    stats = {
        "exhaustive_component_pairs": 0,
        "vlad_component_pairs": 0,
        "fallback_component_pairs": 0,
    }

    for comp_a, comp_b in combinations(components, 2):
        if len(comp_a) * len(comp_b) <= exhaustive_cap:
            pairs |= _exhaustive_pairs(comp_a, comp_b)
            stats["exhaustive_component_pairs"] += 1
            continue

        vlad_pairs = _vlad_pairs(
            data, exifs, comp_a, comp_b, vlad_neighbors, histograms
        )
        if vlad_pairs:
            pairs |= vlad_pairs
            stats["vlad_component_pairs"] += 1
        else:
            logger.warning(
                "VLAD unavailable for components of size %d and %d, "
                "falling back to subsampled matching",
                len(comp_a),
                len(comp_b),
            )
            pairs |= _subsampled_pairs(comp_a, comp_b, exhaustive_cap)
            stats["fallback_component_pairs"] += 1

    return pairs, stats


def _exhaustive_pairs(
    comp_a: Set[str], comp_b: Set[str]
) -> Set[Tuple[str, str]]:
    return {pairs_selection.sorted_pair(a, b) for a in comp_a for b in comp_b}


def _vlad_pairs(
    data: DataSetBase,
    exifs: Dict[str, Any],
    comp_a: Set[str],
    comp_b: Set[str],
    vlad_neighbors: int,
    histograms: Dict[str, NDArray],
) -> Set[Tuple[str, str]]:
    """Top-vlad_neighbors most similar cross-component pairs per image of the
    smaller component. Returns an empty set if VLAD data is unavailable."""
    smaller, larger = sorted((comp_a, comp_b), key=lambda c: (len(c), min(c)))

    need = (comp_a | comp_b) - set(histograms)
    if need:
        try:
            histograms.update(pairs_selection.vlad_histograms(need, data))
        except Exception as e:
            logger.warning("Could not compute VLAD histograms: %s", e)
            return set()

    cand_images = sorted(im for im in larger if im in histograms)
    if not cand_images:
        return set()

    results = [
        vlad.vlad_distances(im, cand_images, histograms)
        for im in sorted(smaller)
        if im in histograms
    ]
    scored = pairs_selection.construct_pairs(
        results, vlad_neighbors, exifs, enforce_other_cameras=False
    )
    return set(scored.keys())


def _subsampled_pairs(
    comp_a: Set[str], comp_b: Set[str], cap: int
) -> Set[Tuple[str, str]]:
    """Deterministic even-stride subsampling of all cross pairs down to cap
    pairs."""
    all_pairs = sorted(
        pairs_selection.sorted_pair(a, b) for a in comp_a for b in comp_b
    )
    if len(all_pairs) <= cap:
        return set(all_pairs)
    stride = len(all_pairs) / float(cap)
    return {all_pairs[int(i * stride)] for i in range(cap)}


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


def write_report(
    data: DataSetBase,
    components_before: List[Set[str]],
    components_after: List[Set[str]],
    pairs: List[Tuple[str, str]],
    num_matched_pairs: int,
    selection_stats: Dict[str, int],
    wall_time: float,
) -> None:
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
