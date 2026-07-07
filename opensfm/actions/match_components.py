# pyre-strict
import logging
from timeit import default_timer as timer

from opensfm import io, matching, tracking
from opensfm.dataset_base import DataSetBase

logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(data: DataSetBase) -> None:
    """Match features across disconnected components of the matching graph
    of an already matched dataset."""

    start = timer()
    images = data.images()
    pairs_matches = dict(tracking.load_matches(data, images))
    new_matches, report = matching.bridge_matching_components(
        data, {}, images, pairs_matches
    )
    matching.save_matches_merging(data, new_matches)
    matching.clear_cache()

    report["wall_time"] = timer() - start
    data.save_report(io.json_dumps(report), "match_components.json")
