# pyre-strict
import argparse

from opensfm.actions import match_components
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "match_components"
    help = "Match features across disconnected components of the matching graph"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        match_components.run_dataset(dataset)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
