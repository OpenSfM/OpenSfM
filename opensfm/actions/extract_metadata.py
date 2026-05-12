# pyre-strict
import copy
import logging
from functools import partial
from typing import Any, Dict, Tuple

from opensfm import context, exif
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)
logging.getLogger("exifread").setLevel(logging.WARNING)


def _load_or_extract_exif(
    image: str,
    data: DataSetBase,
    force: bool,
    exif_overrides: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], bool]:
    if not force and data.exif_exists(image):
        logger.info("Loading existing EXIF for %s", image)
        return image, data.load_exif(image), False

    logger.info("Extracting EXIF for %s", image)
    metadata = _extract_exif(image, data)

    if image in exif_overrides:
        metadata.update(exif_overrides[image])

    return image, metadata, True


def run_dataset(data: DataSetBase, force: bool = False) -> None:
    """Extract metadata from images' EXIF tag."""

    exif_overrides: Dict[str, Dict[str, Any]] = {}
    if data.exif_overrides_exists():
        exif_overrides = data.load_exif_overrides()

    camera_models = {}
    image_metadata = context.parallel_map(
        partial(
            _load_or_extract_exif,
            data=data,
            force=force,
            exif_overrides=exif_overrides,
        ),
        data.images(),
        data.config["processes"],
        backend="multiprocessing",
    )
    for image, d, should_save in image_metadata:
        if should_save:
            data.save_exif(image, d)

        if d["camera"] not in camera_models:
            camera = exif.camera_from_exif_metadata(d, data)
            camera_models[d["camera"]] = camera

    # Override any camera specified in the camera models overrides file.
    if data.camera_models_overrides_exists():
        overrides = data.load_camera_models_overrides()
        if "all" in overrides:
            for key in camera_models:
                camera_models[key] = copy.copy(overrides["all"])
                camera_models[key].id = key
        else:
            for key, value in overrides.items():
                camera_models[key] = value
    data.save_camera_models(camera_models)
    data.init_reference()


def _extract_exif(image: str, data: DataSetBase) -> Dict[str, Any]:
    with data.open_image_file(image) as fp:
        d = exif.extract_exif_from_file(
            fp,
            partial(data.image_size, image),
            data.config["use_exif_size"],
            data.config["default_projection_type"],
            name=image,
        )

    if data.config["unknown_camera_models_are_different"] and (
        not d["model"] or d["model"] == "unknown"
    ):
        d["model"] = f"unknown_{image}"

    d["camera"] = exif.camera_id(d)

    return d
