# pyre-strict
import logging
from typing import Dict, List

from opensfm import pymap, reconstruction_helpers as helpers, rig, types
from opensfm.dataset import DataSet, DataSetBase
from opensfm.types import Reconstruction


logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(
    data: DataSet, 
    method: str, 
    calibration_type: str,
    definition: Dict[str, str], 
    output_debug: bool
) -> None:
    """Given a dataset that contains rigs, construct rig data files.

    Args:
        data: dataset object
        method : 'pattern` will create instances based on a REGEX pattern (see below)
                 'assignments` will create instances based on the rig_assignments.json
        calibration_type: 'sfm' will run incremental SfM to estimate the rig camera poses
                          'metadata' will use the image metadata (GPS and orientation) to 
                          estimate the rig camera poses
        definition : For the `pattern` method, a JSON dict (one for each RigCamera) with values as :
                    - (.*) for `pattern` method where the part outside
                        of parenthesis defines a RigCamera instance
                    - a camera model ID for the `camera` method
        output_debug : output a debug JSON reconstruction `rig_instances.json` with rig instances
    """
    if method == "pattern":
        rig.create_rigs_with_pattern(data, calibration_type, definition)
    elif method == "assignments":
        rig.create_rigs_with_assignments(data, calibration_type)
    else:
        raise ValueError(f"Unsupported method {method}")
    if output_debug:
        reconstructions = _reconstruction_from_rigs_and_assignments(data)
        data.save_reconstruction(reconstructions, "rig_instances.json")


def _reconstruction_from_rigs_and_assignments(
    data: DataSetBase,
) -> List[Reconstruction]:
    assignments = data.load_rig_assignments()
    rig_cameras = data.load_rig_cameras()

    data.init_reference()

    reconstruction = types.Reconstruction()
    reconstruction.cameras = data.load_camera_models()
    for rig_instance_id, instance in assignments.items():
        for image, rig_camera_id in instance:
            rig_camera = rig_cameras[rig_camera_id]
            reconstruction.add_rig_camera(
                pymap.RigCamera(rig_camera.pose, rig_camera_id)
            )

            instance_obj = reconstruction.add_rig_instance(
                pymap.RigInstance(rig_instance_id)
            )
            instance_obj.pose.set_origin(
                helpers.get_image_metadata(data, image).gps_position.value
            )

            d = data.load_exif(image)
            shot = reconstruction.create_shot(
                image,
                camera_id=d["camera"],
                rig_camera_id=rig_camera_id,
                rig_instance_id=rig_instance_id,
            )
            shot.metadata = helpers.get_image_metadata(data, image)
    return [reconstruction]
