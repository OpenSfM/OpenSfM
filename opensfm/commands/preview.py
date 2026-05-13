# pyre-strict
import argparse

from opensfm.actions import preview
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "preview"
    help = "Run fast incremental reconstruction for visual preview"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        preview.run_dataset(dataset)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
