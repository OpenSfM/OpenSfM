# pyre-strict
from timeit import default_timer as timer
from typing import Any, Dict, List, Tuple

from opensfm import io, matching
from opensfm.dataset_base import DataSetBase


def run_dataset(data: DataSetBase) -> None:
    """Match features between image pairs."""

    images = data.images()

    start = timer()
    exifs = {im: data.load_exif(im) for im in images}
    pairs_matches, preport = matching.match_images(data, {}, images, images, exifs)
    if data.config["force_match_components"]:
        new_matches, creport = matching.bridge_matching_components(
            data, {}, images, pairs_matches, exifs
        )
        pairs_matches.update(new_matches)
        preport.update(creport)
    matching.save_matches(data, images, pairs_matches)
    matching.clear_cache()
    end = timer()
    write_report(data, preport, list(pairs_matches.keys()), end - start)


def write_report(
    data: DataSetBase,
    preport: Dict[str, Any],
    pairs: List[Tuple[str, str]],
    wall_time: float,
) -> None:
    report = {
        "wall_time": wall_time,
        "num_pairs": len(pairs),
        "pairs": pairs,
    }
    report.update(preport)
    data.save_report(io.json_dumps(report), "matches.json")
