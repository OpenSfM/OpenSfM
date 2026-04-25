#!/usr/bin/env python3
# pyre-strict
"""Benchmark CLI entry point.

Usage:
    # New run:
    python -m benchmark.run --config benchmark.json --commit abc1234

    # Resume an interrupted run (skip completed steps):
    python -m benchmark.run --config benchmark.json --resume /path/to/run_dir

    # Re-run from a specific step in an existing run dir:
    python -m benchmark.run --config benchmark.json --resume /path/to/run_dir --from-step reconstruct

    # New run bootstrapped from an existing run (reuse features/matches, re-run from reconstruct):
    python -m benchmark.run --config benchmark.json --commit abc1234 --from-step reconstruct
    python -m benchmark.run --config benchmark.json --commit abc1234 --from-step reconstruct --bootstrap /path/to/run_dir

    # Compare against an explicit reference:
    python -m benchmark.run --config benchmark.json --commit abc1234 --reference /path/to/ref_run
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from benchmark.compare import (
    find_reference_run,
    generate_comparison_html,
    load_run_meta,
    load_run_stats,
)
from benchmark.config import load_config
from benchmark.pipeline import PIPELINE_STEPS, run_all_datasets, save_run_meta, protect_self_from_oom
from benchmark.workspace import (
    build_in_worktree,
    cleanup_worktree,
    setup_conda_env,
    setup_dataset,
    setup_worktree,
    _resolve_commit,
)

logger: logging.Logger = logging.getLogger("benchmark")


def _find_repo_root() -> str:
    """Find the git repository root from this file's location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_bootstrap_run(output_dir: str, commit_hash: str) -> Optional[str]:
    """Auto-detect the most recent complete run for a given commit hash."""
    short = commit_hash[:8]
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if not name.startswith(short):
            continue
        run_path = os.path.join(output_dir, name)
        meta = load_run_meta(run_path)
        if meta and meta.get("status") == "complete":
            candidates.append((name, run_path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenSfM benchmarks at a specific git commit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark JSON config file.",
    )

    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument(
        "--commit",
        help="Git commit hash (or ref) to benchmark. Starts a new run.",
    )
    run_mode.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help=(
            "Resume or re-run from an existing run directory. "
            "The commit hash is read from run_meta.json inside RUN_DIR."
        ),
    )

    parser.add_argument(
        "--from-step",
        choices=PIPELINE_STEPS,
        default=None,
        metavar="STEP",
        help=(
            "Start execution from this step, skipping earlier steps. "
            "With --resume: re-run from this step in the same directory. "
            "With --commit: creates a new run and bootstraps prior step outputs "
            "from an existing run (auto-detected or specified via --bootstrap). "
            f"Choices: {', '.join(PIPELINE_STEPS)}"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        metavar="RUN_DIR",
        default=None,
        help=(
            "Only valid with --commit --from-step. Path to an existing run directory "
            "to symlink/copy prior step outputs from. If omitted, the most recent "
            "run for the same commit is used automatically."
        ),
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference run: path to a previous run directory, or a commit hash prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config (ignored when --resume is used).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    # Protect this orchestrator process from the OOM killer — pipeline
    # subprocesses are given a high OOM score instead, so the kernel
    # kills them first under memory pressure rather than us.
    protect_self_from_oom(-500)

    if args.bootstrap and not (args.commit and args.from_step):
        parser.error("--bootstrap requires --commit and --from-step")
    if args.from_step and args.resume and args.bootstrap:
        parser.error("--bootstrap is only used with --commit, not --resume")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load and validate config
    config = load_config(args.config)

    repo_root = _find_repo_root()
    is_resume = args.resume is not None

    # -----------------------------------------------------------------------
    # Determine run directory and commit hash
    # -----------------------------------------------------------------------
    if is_resume:
        run_dir = os.path.abspath(args.resume)
        if not os.path.isdir(run_dir):
            logger.error("Resume directory does not exist: %s", run_dir)
            sys.exit(1)

        existing_meta = load_run_meta(run_dir) or {}
        full_hash: str = existing_meta.get("commit", "")

        # Fall back to parsing the commit hash from the folder name (e.g. "cff716cd_20260422_062256")
        if not full_hash:
            folder_name = os.path.basename(run_dir)
            short_from_name = folder_name.split("_")[0]
            if short_from_name:
                try:
                    full_hash = _resolve_commit(short_from_name, repo_root)
                    logger.info(
                        "Inferred commit %s from run directory name.", full_hash[:8]
                    )
                except Exception:
                    pass

        if not full_hash:
            logger.error(
                "Cannot determine commit hash from directory name or run_meta.json in %s.",
                run_dir,
            )
            sys.exit(1)
        short_hash = full_hash[:8]
        mode_desc = f"Resuming run at {run_dir}"
        if args.from_step:
            mode_desc += f" from step '{args.from_step}'"
        logger.info("%s (commit %s)", mode_desc, short_hash)
    else:
        if args.output_dir:
            config.output_dir = os.path.abspath(args.output_dir)
        full_hash = _resolve_commit(args.commit, repo_root)
        short_hash = full_hash[:8]
        logger.info("Benchmarking commit %s (%s)", short_hash, full_hash)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{short_hash}_{timestamp}"
        run_dir = os.path.join(config.output_dir, run_name)
        old_umask = os.umask(0o000)
        os.makedirs(run_dir, exist_ok=True, mode=0o777)
        os.umask(old_umask)
        logger.info("Run directory: %s", run_dir)
        existing_meta = None

    # -----------------------------------------------------------------------
    # Setup worktree, conda env, and build  (always needed — env was cleaned up)
    # -----------------------------------------------------------------------
    worktree_path = setup_worktree(full_hash, repo_root)
    conda_env = None
    try:
        conda_env = setup_conda_env(worktree_path, full_hash)
        build_in_worktree(worktree_path, conda_env)

        # Dataset setup: idempotent — skips datasets that already have image_list.txt
        for dataset_name, config_name in config.datasets.items():
            target_dir = os.path.join(run_dir, dataset_name)
            source_dir = os.path.join(config.root, dataset_name)
            config_file = os.path.join(
                config.configs_dir, f"{config_name}.yaml")
            setup_dataset(source_dir, target_dir, config_file)
            logger.info("Dataset prepared: %s (config: %s)",
                        dataset_name, config_name)

        # Write initial run_meta.json before pipeline starts (crash-safe)
        if not is_resume:
            initial_meta = {
                "commit": full_hash,
                "date": datetime.now(timezone.utc).isoformat(),
                "status": "in_progress",
                "config": {
                    "root": config.root,
                    "datasets": config.datasets,
                    "output_dir": config.output_dir,
                },
                "datasets": {},
            }
            save_run_meta(initial_meta, run_dir)
            existing_meta = initial_meta

        opensfm_bin = os.path.join(worktree_path, "bin", "opensfm")

        # Resolve bootstrap source for --commit --from-step
        bootstrap_run_dir: Optional[str] = None
        if args.from_step and not is_resume:
            if args.bootstrap:
                bootstrap_run_dir = os.path.abspath(args.bootstrap)
                if not os.path.isdir(bootstrap_run_dir):
                    raise ValueError(
                        f"Bootstrap directory does not exist: {bootstrap_run_dir}")
            else:
                bootstrap_run_dir = _find_bootstrap_run(
                    config.output_dir, full_hash)
                if bootstrap_run_dir:
                    logger.info("Auto-detected bootstrap source: %s",
                                bootstrap_run_dir)
                else:
                    logger.warning(
                        "No complete previous run found for commit %s to bootstrap from. "
                        "Steps before '%s' will be run from scratch.",
                        short_hash, args.from_step,
                    )

        run_meta = run_all_datasets(
            opensfm_bin,
            run_dir,
            config,
            full_hash,
            conda_env,
            resume=is_resume,
            from_step=args.from_step,
            existing_meta=existing_meta,
            bootstrap_run_dir=bootstrap_run_dir,
        )
    finally:
        cleanup_worktree(worktree_path, repo_root, conda_env)

    # -----------------------------------------------------------------------
    # HTML comparison report
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
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
