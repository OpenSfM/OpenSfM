# pyre-strict
"""Git worktree management and dataset directory setup."""

import logging
import os
import platform
import shutil
import subprocess
from typing import List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)

CONDA_ENV_PREFIX = "opensfm-bench-"

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "tif", "tiff", "pgm", "pnm", "gif"}

COPYABLE_FILES = [
    "gcp_list.txt",
    "ground_control_points.json",
    "camera_models_overrides.json",
]

COPYABLE_DIRS = [
    "masks",
]


def _resolve_commit(commit: str, repo_root: str) -> str:
    """Resolve a commit reference to a full hash."""
    result = subprocess.run(
        ["git", "rev-parse", commit],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def setup_worktree(commit: str, repo_root: str) -> str:
    """Create a git worktree for the given commit. Returns the worktree path."""
    full_hash = _resolve_commit(commit, repo_root)
    worktree_dir = os.path.join(repo_root, "benchmark", ".worktrees")
    os.makedirs(worktree_dir, exist_ok=True)
    worktree_path = os.path.join(worktree_dir, full_hash[:12])

    if os.path.isdir(worktree_path):
        logger.info("Removing existing worktree at %s", worktree_path)
        subprocess.run(
            ["git", "worktree", "remove", worktree_path, "--force"],
            cwd=repo_root,
            check=True,
        )

    logger.info("Creating worktree for %s at %s", full_hash[:8], worktree_path)
    subprocess.run(
        ["git", "worktree", "add", "--detach", worktree_path, full_hash],
        cwd=repo_root,
        check=True,
    )

    # Initialize submodules (e.g. pybind11) in the worktree
    logger.info("Initializing submodules in worktree")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=worktree_path,
        check=True,
    )

    return worktree_path


def _conda_env_name(commit_hash: str) -> str:
    """Generate a unique conda environment name for a benchmark run."""
    return f"{CONDA_ENV_PREFIX}{commit_hash[:12]}"


def _detect_lock_file(worktree_path: str) -> Optional[str]:
    """Check for conda lock files in the worktree. Returns the path if found."""
    system = platform.system().lower()  # linux, darwin
    arch = platform.machine()           # x86_64, aarch64
    # Try platform-specific lock file first, then generic
    candidates = [
        f"conda-{system}-{arch}.lock",
        f"conda-{system}-64.lock",
        "conda-lock.yml",
    ]
    for name in candidates:
        path = os.path.join(worktree_path, name)
        if os.path.isfile(path):
            return path
    return None


def setup_conda_env(worktree_path: str, commit_hash: str) -> str:
    """Create a conda environment for the worktree's dependencies.

    Checks for lock files first (newer setup), falls back to conda.yml.
    Returns the conda environment name.
    """
    env_name = _conda_env_name(commit_hash)

    # Remove existing env if present
    subprocess.run(
        ["conda", "env", "remove", "--name", env_name, "--yes"],
        capture_output=True,
        check=False,
    )

    lock_file = _detect_lock_file(worktree_path)
    conda_yml = os.path.join(worktree_path, "conda.yml")

    if lock_file:
        logger.info("Creating conda env '%s' from lock file: %s",
                    env_name, lock_file)
        subprocess.run(
            ["conda", "create", "--name", env_name, "--file", lock_file, "--yes"],
            check=True,
        )
    elif os.path.isfile(conda_yml):
        logger.info("Creating conda env '%s' from conda.yml", env_name)
        subprocess.run(
            ["conda", "env", "create", "--file",
                conda_yml, "--name", env_name, "--yes"],
            check=True,
        )
    else:
        raise FileNotFoundError(
            f"No conda lock file or conda.yml found in worktree {worktree_path}"
        )

    return env_name


def build_in_worktree(worktree_path: str, conda_env: str) -> None:
    """Build OpenSfM in the worktree within the dedicated conda env."""
    logger.info("Building OpenSfM in worktree %s (env: %s)",
                worktree_path, conda_env)
    subprocess.run(
        ["conda", "run", "--name", conda_env,
         "pip", "install", "-e", "."],
        cwd=worktree_path,
        check=True,
    )


def cleanup_worktree(worktree_path: str, repo_root: str, conda_env: Optional[str] = None) -> None:
    """Remove the worktree and its conda environment."""
    logger.info("Removing worktree %s", worktree_path)
    subprocess.run(
        ["git", "worktree", "remove", worktree_path, "--force"],
        cwd=repo_root,
        check=False,
    )

    if conda_env:
        logger.info("Removing conda env '%s'", conda_env)
        subprocess.run(
            ["conda", "env", "remove", "--name", conda_env, "--yes"],
            capture_output=True,
            check=False,
        )


def _list_images(images_dir: str) -> List[str]:
    """List image files in a directory, sorted."""
    files = []
    for name in sorted(os.listdir(images_dir)):
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(images_dir, name))
    return files


def setup_dataset(source_dir: str, target_dir: str, config_file: Optional[str] = None) -> None:
    """Create a lightweight benchmark dataset directory.

    Generates image_list.txt pointing to the source images via absolute paths,
    copies ancillary files (gcp, etc.) from the source, and installs the
    benchmark config file with the machine's CPU count as 'processes'.
    """
    # Use permissive umask so files are readable/writable across processes
    # (needed for NAS mounts where conda run may have different effective user)
    old_umask = os.umask(0o000)
    try:
        _setup_dataset_inner(source_dir, target_dir, config_file)
    finally:
        os.umask(old_umask)


def _setup_dataset_inner(source_dir: str, target_dir: str, config_file: Optional[str] = None) -> None:
    os.makedirs(target_dir, exist_ok=True, mode=0o777)

    # Generate image_list.txt with absolute paths to source images
    images_dir = os.path.join(source_dir, "images")
    image_paths = _list_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    image_list_path = os.path.join(target_dir, "image_list.txt")
    with open(image_list_path, "w") as f:
        for img_path in image_paths:
            f.write(img_path + "\n")

    # Copy the benchmark config and append processes count
    if config_file and os.path.isfile(config_file):
        target_config = os.path.join(target_dir, "config.yaml")
        shutil.copy2(config_file, target_config)
        ncpus = os.cpu_count() or 1
        with open(target_config, "a") as f:
            f.write(f"\nprocesses: {ncpus}\n")
        logger.info("Config %s installed with processes=%d",
                    config_file, ncpus)

    # Copy ancillary files
    for filename in COPYABLE_FILES:
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(target_dir, filename))

    # Copy ancillary directories
    for dirname in COPYABLE_DIRS:
        src = os.path.join(source_dir, dirname)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(
                target_dir, dirname), dirs_exist_ok=True)
