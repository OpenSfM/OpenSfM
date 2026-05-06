# pyre-strict
import argparse
from typing import Tuple

from opensfm.actions import crop_reconstruction
from opensfm.dataset import DataSet
from . import command

class Command(command.CommandBase):
    name = "crop_reconstruction"
    help = "Crop reconstruction to N images around a shifted center"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        crop_reconstruction.run_dataset(dataset, args.n, tuple(args.shift))

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-n", "--n",
            help="Number of images to keep",
            type=int,
            default=50,
        )
        parser.add_argument(
            "--shift",
            help="Shift of the center in X and Y (e.g. -1.0 -1.0)",
            type=float,
            nargs=2,
            default=[0.0, 0.0],
        )

