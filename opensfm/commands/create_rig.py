# pyre-strict
import argparse
import json

from opensfm.actions import create_rig
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "create_rig"
    help = "Create rig by creating `rig_cameras.json` and `rig_assignments.json` files."

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        create_rig.run_dataset(dataset, args.method, args.calibration_type, json.loads(args.definition), True)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--method",
            choices=["assignments", "pattern", "camera"],
            help=(
                "Method for creating the rigs",
                "`assignments` will create rigs based on the rig_assignments.json, "
                "`pattern` will create rigs based on a REGEX pattern (see below), "
                "`camera` will create rigs based on a camera model ID"
            ),
        )
        parser.add_argument(
            "--calibration-type",
            choices=["metadata", "sfm"],
            help=(
                "Method for calibrating the rig cameras. "
                "`metadata` will use the image metadata to estimate the rig camera poses, "
                "`sfm` will run incremental SfM to estimate the rig camera poses"
            ),
        )
        parser.add_argument(
            "--definition",
            help=(
                "Defines each RigCamera as a JSON string dict with the form `{camera_id: definition, ...}`"
                "For `pattern`, the definition is expected to be a REGEX with the form (.*)"
                "For `camera`, the definition is expected to be a camera model ID"
            ),
        )
