# pyre-strict
"""Benchmark configuration loading and validation."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BenchmarkConfig:
    root: str
    datasets: Dict[str, str]  # dataset_name -> config_name
    configs_dir: str  # path to benchmark/configs/ with predefined config.yaml files
    output_dir: str = "./benchmark_runs"

    def dataset_names(self) -> List[str]:
        """Ordered list of dataset names."""
        return list(self.datasets.keys())


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
    if not datasets or not isinstance(datasets, dict):
        raise ValueError(
            "Config must specify 'datasets' as a dict of {name: config_name}"
        )

    # Resolve configs directory (configs/ at repo root, sibling of benchmark/)
    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
    )
    if not os.path.isdir(configs_dir):
        raise ValueError(f"Configs directory does not exist: {configs_dir}")

    for name, config_name in datasets.items():
        ds_path = os.path.join(root, name)
        if not os.path.isdir(ds_path):
            raise ValueError(f"Dataset directory does not exist: {ds_path}")
        images_path = os.path.join(ds_path, "images")
        if not os.path.isdir(images_path):
            raise ValueError(f"Dataset has no images/ directory: {ds_path}")
        config_file = os.path.join(configs_dir, f"{config_name}.yaml")
        if not os.path.isfile(config_file):
            raise ValueError(
                f"Config file not found for dataset '{name}': {config_file}"
            )

    output_dir = data.get("output_dir", "./benchmark_runs")
    output_dir = os.path.abspath(output_dir)

    return BenchmarkConfig(
        root=root, datasets=datasets, configs_dir=configs_dir, output_dir=output_dir
    )
