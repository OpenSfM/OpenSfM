# pyre-strict

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# Each correction function takes (xmp_tags, geo) and returns a YPR array
# and a boolean indicating if pitch offset should be applied.
# xmp_tags is xmp[0] from the EXIF object, geo is the geo dict.
CorrectionFn = Callable[[Dict[str, Any],
                         Dict[str, Any]], Tuple[NDArray[Any], bool]]


def _fix_dji_fc7303(
    xmp_tags: Dict[str, Any], geo: Dict[str, Any]
) -> Tuple[NDArray[Any], bool]:
    """
    The FC7303 has all 0's Gimbal, so we use FlightXXX.

    It also doesn't require the 90 degree offset in pitch
    that other DJI drones do.
    """
    if "latitude" in geo and "longitude" in geo:
        return np.array(
            [
                float(xmp_tags["@drone-dji:FlightYawDegree"]),
                float(xmp_tags["@drone-dji:FlightPitchDegree"]),
                float(xmp_tags["@drone-dji:FlightRollDegree"]),
            ]
        ), False
    return np.array([None, None, None]), False


# Map from camera_id ("{make}_{model}", lowercased) to correction function.
# Add new entries here when a camera needs special YPR handling.
ypr_corrections: Dict[str, CorrectionFn] = {
    "dji_fc7303": _fix_dji_fc7303,
}
