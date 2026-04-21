# pyre-strict
"""SfM pipeline execution for benchmarking."""

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from benchmark.config import BenchmarkConfig

logger: logging.Logger = logging.getLogger(__name__)

PIPELINE_STEPS: List[str] = [
    "extract_metadata",
    "detect_features",
    "match_features",
    "create_tracks",
    "reconstruct",
    "compute_statistics",
    "export_report",
]


def run_pipeline(
    opensfm_bin: str,
    dataset_path: str,
) -> Dict[str, Any]:
    """Run the SfM pipeline on a single dataset.

    Returns a dict with per-step timings and success/failure status.
    """
    result: Dict[str, Any] = {
        "success": True,
        "steps": {},
        "failed_step": None,
    }

    for step in PIPELINE_STEPS:
        logger.info("  [%s] %s ...", os.path.basename(dataset_path), step)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [opensfm_bin, step, dataset_path],
                capture_output=True,
                text=True,
                check=True,
            )
            elapsed = time.monotonic() - t0
            result["steps"][step] = {
                "wall_time": round(elapsed, 2),
                "success": True,
            }
            logger.info("  [%s] %s done (%.1fs)",
                        os.path.basename(dataset_path), step, elapsed)
        except subprocess.CalledProcessError as e:
            elapsed = time.monotonic() - t0
            result["steps"][step] = {
                "wall_time": round(elapsed, 2),
                "success": False,
                "stderr": e.stderr[-2000:] if e.stderr else "",
            }
            result["success"] = False
            result["failed_step"] = step
            logger.error(
                "  [%s] %s FAILED (%.1fs)\n%s",
                os.path.basename(dataset_path),
                step,
                elapsed,
                e.stderr[-500:] if e.stderr else "",
            )
            break

    return result


def run_all_datasets(
    opensfm_bin: str,
    run_dir: str,
    config: BenchmarkConfig,
    commit_hash: str,
) -> Dict[str, Any]:
    """Run the pipeline on all datasets and write run_meta.json.

    Returns the run metadata dict.
    """
    run_meta: Dict[str, Any] = {
        "commit": commit_hash,
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "root": config.root,
            "datasets": config.datasets,
            "output_dir": config.output_dir,
        },
        "datasets": {},
    }

    total_t0 = time.monotonic()

    for dataset_name in config.datasets:
        dataset_path = os.path.join(run_dir, dataset_name)
        logger.info("Running pipeline on %s", dataset_name)
        pipeline_result = run_pipeline(opensfm_bin, dataset_path)
        run_meta["datasets"][dataset_name] = pipeline_result

    run_meta["total_wall_time"] = round(time.monotonic() - total_t0, 2)

    meta_path = os.path.join(run_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("Run metadata written to %s", meta_path)

    return run_meta
