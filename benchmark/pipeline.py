# pyre-strict
"""SfM pipeline execution for benchmarking."""

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

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

# For each step, the path relative to dataset_path that indicates completion.
# Directories are considered complete when non-empty.
STEP_OUTPUT_FILES: Dict[str, str] = {
    "extract_metadata": "exif",
    "detect_features": "features",
    "match_features": "reports/matches.json",
    "create_tracks": "tracks.csv",
    "reconstruct": "reconstruction.json",
    "compute_statistics": "stats/stats.json",
    "export_report": "stats/report.pdf",
}

# All files/directories a step produces that a subsequent step may need.
# Used to bootstrap a new run from an existing one.
STEP_OUTPUTS: Dict[str, List[str]] = {
    "extract_metadata": ["exif", "camera_models.json"],
    "detect_features": ["features"],
    "match_features": ["matches", "reports/matches.json"],
    "create_tracks": ["tracks.csv", "reports/tracks.json"],
    "reconstruct": ["reconstruction.json", "reports/reconstruction.json"],
    "compute_statistics": ["stats"],
    "export_report": [],
}


def _set_oom_score_adj(score: int) -> None:
    """Write an OOM score adjustment to /proc/self/oom_score_adj.

    Called as preexec_fn in pipeline subprocesses so the kernel preferentially
    kills them under memory pressure rather than the benchmark orchestrator.
    """
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(score))
    except OSError:
        pass


def _make_oom_preexec(score: int = 500) -> Callable[[], None]:
    def _preexec() -> None:
        _set_oom_score_adj(score)
    return _preexec


def protect_self_from_oom(score: int = -500) -> None:
    """Lower the current process's OOM score so it survives memory pressure."""
    _set_oom_score_adj(score)


def _refresh_disk_cache(dataset_path: str) -> None:
    """
    Force NFS attribute-cache refresh on key dataset files.
    """
    for name in ("image_list.txt", "config.yaml"):
        p = os.path.join(dataset_path, name)
        try:
            os.stat(p)
            # Touch-read a few bytes to force the NFS client to revalidate
            with open(p, "rb") as f:
                f.read(1)
        except OSError:
            pass
    # Also stat the directory itself
    try:
        os.listdir(dataset_path)
    except OSError:
        pass


def bootstrap_dataset(source_dataset_dir: str, target_dataset_dir: str, from_step: str) -> None:
    """Populate target_dataset_dir with outputs of all steps before from_step.

    Directories are symlinked; individual files are copied.
    The target dataset directory must already exist (image_list.txt etc. in place).
    """
    from_idx = PIPELINE_STEPS.index(from_step)
    steps_to_copy = PIPELINE_STEPS[:from_idx]

    old_umask = os.umask(0o000)
    try:
        for step in steps_to_copy:
            for rel_path in STEP_OUTPUTS.get(step, []):
                src = os.path.join(source_dataset_dir, rel_path)
                dst = os.path.join(target_dataset_dir, rel_path)

                if not os.path.exists(src):
                    continue

                # Ensure parent directory exists
                parent = os.path.dirname(dst)
                if parent:
                    os.makedirs(parent, exist_ok=True, mode=0o777)

                if os.path.lexists(dst):
                    continue  # already bootstrapped

                if os.path.isdir(src):
                    os.symlink(src, dst)
                    logger.info("  bootstrap symlink: %s -> %s", dst, src)
                else:
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info("  bootstrap copy:    %s", rel_path)
    finally:
        os.umask(old_umask)


def is_step_complete(step: str, dataset_path: str) -> bool:
    """Return True if the step's output already exists in the dataset directory."""
    rel = STEP_OUTPUT_FILES.get(step)
    if not rel:
        return False
    full_path = os.path.join(dataset_path, rel)
    if os.path.isdir(full_path):
        return bool(os.listdir(full_path))
    return os.path.isfile(full_path)


def save_run_meta(run_meta: Dict[str, Any], run_dir: str) -> None:
    meta_path = os.path.join(run_dir, "run_meta.json")
    old_umask = os.umask(0o000)
    try:
        with open(meta_path, "w") as f:
            json.dump(run_meta, f, indent=2)
        os.chmod(meta_path, 0o666)
    finally:
        os.umask(old_umask)


def run_pipeline(
    opensfm_bin: str,
    dataset_path: str,
    conda_env: str,
    resume: bool = False,
    from_step: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the SfM pipeline on a single dataset.

    Args:
        resume:    If True and from_step is None, skip steps whose outputs
                   already exist (crash recovery).
        from_step: If set, steps *before* this step are skipped unconditionally
                   (assumed bootstrapped or already complete); this step and all
                   subsequent steps are always run.

    Returns a dict with per-step timings and success/failure status.
    """
    from_idx = PIPELINE_STEPS.index(from_step) if from_step else 0

    result: Dict[str, Any] = {
        "success": True,
        "steps": {},
        "failed_step": None,
    }

    for step in PIPELINE_STEPS:
        step_idx = PIPELINE_STEPS.index(step)
        ds_name = os.path.basename(dataset_path)

        # Steps before from_step: skip unconditionally
        if from_step and step_idx < from_idx:
            logger.info(
                "  [%s] %s SKIPPED (before --from-step)", ds_name, step)
            result["steps"][step] = {
                "skipped": True, "reason": "before_from_step"}
            continue

        # Steps at/after from_step (or all steps when no from_step):
        # if resume and no from_step, also skip already-complete steps
        if resume and from_step is None and is_step_complete(step, dataset_path):
            logger.info("  [%s] %s SKIPPED (already complete)", ds_name, step)
            result["steps"][step] = {
                "skipped": True, "reason": "already_complete"}
            continue

        logger.info("  [%s] %s ...", ds_name, step)
        t0 = time.monotonic()

        # Force disk attribute-cache refresh
        _refresh_disk_cache(dataset_path)

        try:
            subprocess.run(
                [
                    "conda", "run", "--name", conda_env,
                    "bash", "-c",
                    f"umask 0000 && {opensfm_bin} {step} {dataset_path}",
                ],
                capture_output=True,
                text=True,
                check=True,
                preexec_fn=_make_oom_preexec(500),
            )
            elapsed = time.monotonic() - t0
            result["steps"][step] = {
                "wall_time": round(elapsed, 2),
                "success": True,
            }
            logger.info("  [%s] %s done (%.1fs)", ds_name, step, elapsed)
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
                ds_name,
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
    conda_env: str,
    resume: bool = False,
    from_step: Optional[str] = None,
    existing_meta: Optional[Dict[str, Any]] = None,
    bootstrap_run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the pipeline on all datasets, saving run_meta.json after each one.

    Args:
        bootstrap_run_dir: When from_step is set on a fresh run, copy/symlink
                           outputs of steps before from_step from this directory.

    Returns the final run metadata dict.
    """
    run_meta: Dict[str, Any] = existing_meta or {
        "commit": commit_hash,
        "date": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "config": {
            "root": config.root,
            "datasets": config.datasets,
            "output_dir": config.output_dir,
        },
        "datasets": {},
    }
    # Ensure status is reset when re-running
    run_meta["status"] = "in_progress"
    save_run_meta(run_meta, run_dir)

    total_t0 = time.monotonic()

    for dataset_name in config.datasets:
        dataset_path = os.path.join(run_dir, dataset_name)
        logger.info("Running pipeline on %s", dataset_name)

        if from_step and bootstrap_run_dir:
            src_ds = os.path.join(bootstrap_run_dir, dataset_name)
            if os.path.isdir(src_ds):
                logger.info("  Bootstrapping %s from %s", dataset_name, src_ds)
                bootstrap_dataset(src_ds, dataset_path, from_step)
        pipeline_result = run_pipeline(
            opensfm_bin, dataset_path, conda_env,
            resume=resume,
            from_step=from_step,
        )
        run_meta["datasets"][dataset_name] = pipeline_result
        # Persist after each dataset so a crash is recoverable
        save_run_meta(run_meta, run_dir)

    run_meta["total_wall_time"] = round(time.monotonic() - total_t0, 2)
    run_meta["status"] = "complete"
    save_run_meta(run_meta, run_dir)
    logger.info("Run metadata written to %s/run_meta.json", run_dir)

    return run_meta
