# pyre-strict
"""Benchmark configuration loading and validation."""

import json
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    root: str
    datasets: List[str]
    output_dir: str = "./benchmark_runs"


def load_config(path: str) -> BenchmarkConfig:
    """Load and validate a benchmark configuration from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    root = data.get("root")
    if not root:
        raise ValueError("Config must specify 'root' directory")
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise ValueError(f"Root directory does not exist: {root}")

    datasets = data.get("datasets")
    if not datasets or not isinstance(datasets, list):
        raise ValueError("Config must specify a non-empty 'datasets' list")

    for name in datasets:
        ds_path = os.path.join(root, name)
        if not os.path.isdir(ds_path):
            raise ValueError(f"Dataset directory does not exist: {ds_path}")
        images_path = os.path.join(ds_path, "images")
        if not os.path.isdir(images_path):
            raise ValueError(f"Dataset has no images/ directory: {ds_path}")

    output_dir = data.get("output_dir", "./benchmark_runs")
    output_dir = os.path.abspath(output_dir)

    return BenchmarkConfig(root=root, datasets=datasets, output_dir=output_dir)
