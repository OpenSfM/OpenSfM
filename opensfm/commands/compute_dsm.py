# pyre-strict
import argparse

from opensfm.actions import compute_dsm
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "compute_dsm"
    help = "Compute a Digital Surface Model from the fused point cloud"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        compute_dsm.run_dataset(dataset, args.subfolder)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--subfolder",
            help="undistorted subfolder where to load data",
            default="undistorted",
        )
