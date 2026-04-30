# pyre-strict
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
from numpy.typing import NDArray
from opensfm import exif as oexif, geo, geometry, multiview, pygeometry, pymap, rig, types
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)


def guess_gravity_up_from_orientation_tag(orientation: int) -> NDArray:
    """Guess upward vector in camera coordinates given the orientation tag.

    Assumes camera is looking towards the horizon and horizon is horizontal
    on the image when taking in to account the orientation tag.
    """
    # See http://sylvana.net/jpegcrop/exif_orientation.html
    if orientation == 1:
        return np.array([0, -1, 0])
    if orientation == 2:
        return np.array([0, -1, 0])
    if orientation == 3:
        return np.array([0, 1, 0])
    if orientation == 4:
        return np.array([0, 1, 0])
    if orientation == 5:
        return np.array([-1, 0, 0])
    if orientation == 6:
        return np.array([-1, 0, 0])
    if orientation == 7:
        return np.array([1, 0, 0])
    if orientation == 8:
        return np.array([1, 0, 0])
    raise RuntimeError(f"Error: Unknown orientation tag: {orientation}")


def shot_gravity_up_in_image_axis(shot: pymap.Shot) -> Optional[NDArray]:
    """Get or guess shot's gravity up direction."""
    if shot.metadata.gravity_down.has_value:
        return -shot.metadata.gravity_down.value

    if not shot.metadata.orientation.has_value:
        return None

    orientation = shot.metadata.orientation.value
    if not 1 <= orientation <= 8:
        logger.error(
            "Unknown orientation tag {} for image {}".format(
                orientation, shot.id)
        )
        orientation = 1
    return guess_gravity_up_from_orientation_tag(orientation)


def rotation_from_shot_metadata(shot: pymap.Shot) -> Optional[NDArray]:
    rotation = rotation_from_angles(shot)
    if rotation is None:
        rotation = rotation_from_orientation_compass(shot)
    return rotation


def rotation_from_orientation_compass(shot: pymap.Shot) -> Optional[NDArray]:
    up_vector = shot_gravity_up_in_image_axis(shot)
    if up_vector is None:
        return None
    if shot.metadata.compass_angle.has_value:
        angle = shot.metadata.compass_angle.value
    else:
        angle = 0.0
    return multiview.rotation_matrix_from_up_vector_and_compass(list(up_vector), angle)


def rotation_from_angles(shot: pymap.Shot) -> Optional[NDArray]:
    if not shot.metadata.opk_angles.has_value:
        return None
    opk_degrees = shot.metadata.opk_angles.value
    opk_rad = map(math.radians, opk_degrees)
    return geometry.rotation_from_opk(*opk_rad)


def reconstruction_from_metadata(
    data: DataSetBase, images: Iterable[str]
) -> types.Reconstruction:
    """Initialize a reconstruction by using EXIF data for constructing shot poses and cameras."""
    data.init_reference()
    rig_assignments = rig.rig_assignments_per_image(
        data.load_rig_assignments())
    rig_camera_priors = data.load_rig_cameras()

    reconstruction = types.Reconstruction()
    reconstruction.reference = data.load_reference()
    reconstruction.cameras = data.load_camera_models()

    for rig_camera_id, rig_camera in rig_camera_priors.items():
        reconstruction.add_rig_camera(rig_camera)

    shot_poses: Dict[str, pygeometry.Pose] = {}
    all_images: Set[str] = set()
    for image in images:
        all_images.add(image)
        if image in rig_assignments:
            _, _, instance_shots = rig_assignments[image]
            all_images.update(instance_shots)

    for image in all_images:
        camera_id = data.load_exif(image)["camera"]

        if image in rig_assignments:
            rig_instance_id, rig_camera_id, _ = rig_assignments[image]
        else:
            rig_instance_id = image
            rig_camera_id = camera_id

        if rig_camera_id not in reconstruction.rig_cameras:
            reconstruction.add_rig_camera(
                pymap.RigCamera(pygeometry.Pose(), rig_camera_id)
            )

        if rig_instance_id not in reconstruction.rig_instances:
            reconstruction.add_rig_instance(pymap.RigInstance(rig_instance_id))

        shot = reconstruction.create_shot(
            shot_id=image,
            camera_id=camera_id,
            rig_camera_id=rig_camera_id,
            rig_instance_id=rig_instance_id,
        )

        shot.metadata = get_image_metadata(data, image)

        if not shot.metadata.gps_position.has_value:
            reconstruction.remove_shot(image)
            continue
        gps_pos = shot.metadata.gps_position.value

        pose = pygeometry.Pose()
        rotation = rotation_from_shot_metadata(shot)
        if rotation is not None:
            pose.set_rotation_matrix(rotation)
        pose.set_origin(gps_pos)
        shot.scale = 1.0
        shot_poses[image] = pose

    for rig_instance in reconstruction.rig_instances.values():
        average_translation = np.zeros(3)
        rotations = []
        for shot_id in rig_instance.shots:
            if shot_id in shot_poses:
                rig_instance.update_instance_pose_with_shot(
                    shot_id, shot_poses[shot_id]
                )
                average_translation += rig_instance.pose.get_origin()
                rotations.append(rig_instance.pose.rotation)

        if len(rotations) > 0:
            rig_instance.pose.rotation = geometry.average_rotation(rotations)
            rig_instance.pose.set_origin(average_translation / len(rotations))

    return reconstruction


def exif_to_metadata(
    exif: Dict[str, Any], use_altitude: bool, reference: types.TopocentricConverter
) -> pymap.ShotMeasurements:
    """Construct a metadata object from raw EXIF tags (as a dict)."""
    metadata = pymap.ShotMeasurements()

    gps = exif.get("gps")
    if gps and "latitude" in gps and "longitude" in gps:
        lat, lon = gps["latitude"], gps["longitude"]
        if use_altitude:
            alt = min([oexif.maximum_altitude, gps.get("altitude", 2.0)])
        else:
            alt = 2.0  # Arbitrary value used to align the reconstruction
        x, y, z = reference.to_topocentric(lat, lon, alt)
        metadata.gps_position.value = np.array([x, y, z])
        if "latitude_std" in gps and "longitude_std" in gps and "altitude_std" in gps:
            gps_std = np.array(
                [gps["longitude_std"], gps["latitude_std"], gps["altitude_std"]]
            )
        elif "dop" in gps and gps["dop"] > 0:
            gps_std = np.full(3, gps["dop"])
        else:
            gps_std = geo.DEFAULT_GPS_STD.copy()
        gps_std = np.maximum(gps_std, 1e-3)
        metadata.gps_accuracy.value = gps_std

    opk = exif.get("opk")
    if opk and "omega" in opk and "phi" in opk and "kappa" in opk:
        omega, phi, kappa = opk["omega"], opk["phi"], opk["kappa"]
        metadata.opk_angles.value = np.array([omega, phi, kappa])
        metadata.opk_accuracy.value = opk.get("accuracy", 1.0)

    metadata.orientation.value = exif.get("orientation", 1)

    if "accelerometer" in exif:
        logger.warning(
            "'accelerometer' EXIF tag is deprecated in favor of 'gravity_down', which expresses "
            "the gravity down direction in the image coordinate frame."
        )

    if "gravity_down" in exif:
        metadata.gravity_down.value = exif["gravity_down"]

    if "compass" in exif:
        metadata.compass_angle.value = exif["compass"]["angle"]
        if exif["compass"].get("accuracy") is not None:
            metadata.compass_accuracy.value = exif["compass"]["accuracy"]

    if "capture_time" in exif:
        metadata.capture_time.value = exif["capture_time"]

    if "skey" in exif:
        metadata.sequence_key.value = exif["skey"]

    if "relative_altitude" in exif:
        metadata.relative_altitude.value = exif["relative_altitude"]

    return metadata


def get_image_metadata(data: DataSetBase, image: str) -> pymap.ShotMeasurements:
    """Get image metadata as a ShotMetadata object."""
    exif = data.load_exif(image)
    reference = data.load_reference()
    return exif_to_metadata(exif, data.config["use_altitude_tag"], reference)
