#!/usr/bin/env python3
# pyre-strict
"""Benchmark CLI entry point.

Usage:
    python -m benchmark.run --config benchmark.json --commit abc1234
    python -m benchmark.run --config benchmark.json --commit abc1234 --reference prev_run_dir
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from benchmark.compare import (
    find_reference_run,
    generate_comparison_html,
    load_run_meta,
    load_run_stats,
)
from benchmark.config import load_config
from benchmark.pipeline import run_all_datasets
from benchmark.workspace import (
    build_in_worktree,
    cleanup_worktree,
    setup_dataset,
    setup_worktree,
    _resolve_commit,
)

logger: logging.Logger = logging.getLogger("benchmark")


def _find_repo_root() -> str:
    """Find the git repository root from this file's location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenSfM benchmarks at a specific git commit."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark JSON config file.",
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="Git commit hash (or ref) to benchmark.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference run: path to a previous run directory, or a commit hash prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load and validate config
    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = os.path.abspath(args.output_dir)

    repo_root = _find_repo_root()

    # Resolve commit to full hash
    full_hash = _resolve_commit(args.commit, repo_root)
    short_hash = full_hash[:8]
    logger.info("Benchmarking commit %s (%s)", short_hash, full_hash)

    # Create run directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{short_hash}_{timestamp}"
    run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # Setup worktree and build
    worktree_path = setup_worktree(args.commit, repo_root)
    try:
        build_in_worktree(worktree_path)

        # Setup datasets
        for dataset_name in config.datasets:
            source_dir = os.path.join(config.root, dataset_name)
            target_dir = os.path.join(run_dir, dataset_name)
            setup_dataset(source_dir, target_dir)
            logger.info("Dataset prepared: %s", dataset_name)

        # Run pipeline
        opensfm_bin = os.path.join(worktree_path, "bin", "opensfm")
        run_meta = run_all_datasets(opensfm_bin, run_dir, config, full_hash)
    finally:
        cleanup_worktree(worktree_path, repo_root)

    # Find reference and generate comparison
    ref_run_dir = find_reference_run(
        config.output_dir, run_dir, args.reference)
    current_stats = load_run_stats(run_dir)
    reference_stats = load_run_stats(ref_run_dir) if ref_run_dir else None
    reference_meta = load_run_meta(ref_run_dir) if ref_run_dir else None

    if ref_run_dir:
        logger.info("Comparing against reference: %s", ref_run_dir)
    else:
        logger.info(
            "No reference run found — report will show current results only.")

    output_path = generate_comparison_html(
        current_stats, reference_stats, run_meta, reference_meta, run_dir
    )

    # Summary
    total_datasets = len(config.datasets)
    succeeded = sum(
        1 for d in run_meta.get("datasets", {}).values() if d.get("success")
    )
    logger.info(
        "Benchmark complete: %d/%d datasets succeeded (%.1fs total)",
        succeeded,
        total_datasets,
        run_meta.get("total_wall_time", 0),
    )
    logger.info("Report: %s", output_path)


if __name__ == "__main__":
    main()
