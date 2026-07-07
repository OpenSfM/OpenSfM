# pyre-strict
import logging
from itertools import combinations
from timeit import default_timer as timer
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from opensfm import io, matching, pairs_selection, tracking
from opensfm.dataset_base import DataSetBase

logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(data: DataSetBase) -> None:
    """Match features between images belonging to different connected
    components of the matching graph."""

    start = timer()
    images = data.images()
    min_matches: int = data.config["robust_matching_min_match"]

    graph = build_match_graph(data, images, min_matches)
    components = image_components(graph)
    logger.info("Found %d connected components in the matching graph", len(components))

    if len(components) <= 1:
        logger.info("Matching graph already connected, nothing to do")
        write_report(
            data, components, components, [], 0, _empty_selection_stats(), timer() - start
        )
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
        matching.save_matches_merging(data, matched)
        matching.clear_cache()
        num_matched_pairs = sum(1 for m in matched.values() if len(m) >= min_matches)
        # The saved graph gained exactly the matched pairs above the
        # threshold, so update the in-memory graph instead of re-reading
        # every match file.
        for (im1, im2), m in matched.items():
            if len(m) >= min_matches:
                graph.add_edge(im1, im2)

    components_after = image_components(graph)
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


# Overrides deactivating every pair selection strategy: this makes
# match_candidates_from_metadata return all ref x cand pairs.
_ALL_STRATEGIES_OFF: Dict[str, Any] = {
    "matching_gps_distance": 0,
    "matching_gps_neighbors": 0,
    "matching_graph_rounds": 0,
    "matching_time_neighbors": 0,
    "matching_order_neighbors": 0,
    "matching_bow_neighbors": 0,
    "matching_vlad_neighbors": 0,
}


def select_cross_component_pairs(
    data: DataSetBase,
    exifs: Dict[str, Any],
    components: List[Set[str]],
    exhaustive_cap: int,
    vlad_neighbors: int,
) -> Tuple[Set[Tuple[str, str]], Dict[str, int]]:
    """Select image pairs across every pair of components.

    Component pairs with at most exhaustive_cap candidate pairs are matched
    exhaustively; larger ones are pruned with VLAD similarity (top
    vlad_neighbors candidates per image of the smaller component), and
    skipped with a warning when the VLAD data is unavailable.
    """
    pairs: Set[Tuple[str, str]] = set()
    stats = _empty_selection_stats()

    for comp_a, comp_b in combinations(components, 2):
        smaller, larger = sorted((comp_a, comp_b), key=lambda c: (len(c), min(c)))
        override = dict(_ALL_STRATEGIES_OFF)
        if len(comp_a) * len(comp_b) <= exhaustive_cap:
            stats_key = "exhaustive_component_pairs"
        else:
            override["matching_vlad_neighbors"] = vlad_neighbors
            override["matching_vlad_gps_distance"] = 0
            override["matching_vlad_gps_neighbors"] = 0
            stats_key = "vlad_component_pairs"

        try:
            new_pairs, _ = pairs_selection.match_candidates_from_metadata(
                sorted(smaller), sorted(larger), exifs, data, override
            )
        except OSError as e:
            logger.warning(
                "Skipping matching between components of size %d and %d: %s. "
                "Provide VLAD data or raise matching_components_exhaustive_cap.",
                len(comp_a),
                len(comp_b),
                e,
            )
            stats["skipped_component_pairs"] += 1
            continue

        pairs.update(pairs_selection.sorted_pair(im1, im2) for im1, im2 in new_pairs)
        stats[stats_key] += 1

    return pairs, stats


def _empty_selection_stats() -> Dict[str, int]:
    return {
        "exhaustive_component_pairs": 0,
        "vlad_component_pairs": 0,
        "skipped_component_pairs": 0,
    }


def write_report(
    data: DataSetBase,
    components_before: List[Set[str]],
    components_after: List[Set[str]],
    pairs: List[Tuple[str, str]],
    num_matched_pairs: int,
    selection_stats: Dict[str, int],
    wall_time: float,
) -> None:
    """Save the component counts and matched pairs to the report folder."""
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
