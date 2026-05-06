# pyre-strict
import logging
from typing import Tuple

from opensfm import types
from opensfm.dataset import DataSet

logger = logging.getLogger(__name__)


def run_dataset(dataset: DataSet, n: int, shift: Tuple[float, float]) -> None:
    reconstructions = dataset.load_reconstruction()
    tracks_manager = dataset.load_tracks_manager()
    cropped_reconstructions = []

    for rec in reconstructions:
        shot_ids = tuple(rec.shots.keys())
        if not shot_ids:
            cropped_reconstructions.append(rec)
            continue

        origins = []
        for shot_id in shot_ids:
            origins.append((shot_id, rec.shots[shot_id].pose.get_origin()))

        xs = [o[1][0] for o in origins]
        ys = [o[1][1] for o in origins]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        cx = (max_x + min_x) / 2.0 + shift[0] * (max_x - min_x) / 2.0
        cy = (max_y + min_y) / 2.0 + shift[1] * (max_y - min_y) / 2.0

        distances = []
        for shot_id, origin in origins:
            dist = (origin[0] - cx)**2 + (origin[1] - cy)**2
            distances.append((dist, shot_id))

        distances.sort()
        kept_shot_ids = set([s_id for _, s_id in distances[:n]])

        # re-create a new reconstruction with only the kept shots
        rec_cropped = types.Reconstruction()
        rec_cropped.cameras = rec.cameras

        # only add kept shots
        rec_cropped.shots = {s_id: rec.shots[s_id] for s_id in kept_shot_ids}

        # retrieved the seen points by the kept shots
        seen_points = set()
        for shot_id in kept_shot_ids:
            for track_id in tracks_manager.get_shot_observations(shot_id).keys():
                if track_id in rec.points:
                    seen_points.add(track_id)
        for point_id in seen_points:
            rec_cropped.create_point(
                point_id, rec.points[point_id].coordinates)

        # set point colors
        for point_id in seen_points:
            rec_cropped.points[point_id].color = rec.points[point_id].color
        cropped_reconstructions.append(rec_cropped)

    dataset.save_reconstruction(
        cropped_reconstructions, "reconstruction_cropped.json")
