# pyre-strict
"""Git worktree management and dataset directory setup."""

import logging
import os
import shutil
import subprocess
from typing import List

logger: logging.Logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "tif", "tiff", "pgm", "pnm", "gif"}

COPYABLE_FILES = [
    "gcp_list.txt",
    "ground_control_points.json",
    "config.yaml",
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


def build_in_worktree(worktree_path: str) -> None:
    """Build OpenSfM in the worktree via pip install -e ."""
    logger.info("Building OpenSfM in worktree %s", worktree_path)
    subprocess.run(
        ["pip", "install", "-e", "."],
        cwd=worktree_path,
        check=True,
    )


def cleanup_worktree(worktree_path: str, repo_root: str) -> None:
    """Remove the worktree and restore the main checkout's editable install."""
    logger.info("Removing worktree %s", worktree_path)
    subprocess.run(
        ["git", "worktree", "remove", worktree_path, "--force"],
        cwd=repo_root,
        check=False,
    )
    logger.info("Restoring editable install from %s", repo_root)
    subprocess.run(
        ["pip", "install", "-e", "."],
        cwd=repo_root,
        check=True,
    )


def _list_images(images_dir: str) -> List[str]:
    """List image files in a directory, sorted."""
    files = []
    for name in sorted(os.listdir(images_dir)):
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(images_dir, name))
    return files


def setup_dataset(source_dir: str, target_dir: str) -> None:
    """Create a lightweight benchmark dataset directory.

    Generates image_list.txt pointing to the source images via absolute paths,
    and copies ancillary files (gcp, config, etc.) from the source.
    """
    os.makedirs(target_dir, exist_ok=True)

    # Generate image_list.txt with absolute paths to source images
    images_dir = os.path.join(source_dir, "images")
    image_paths = _list_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    image_list_path = os.path.join(target_dir, "image_list.txt")
    with open(image_list_path, "w") as f:
        for img_path in image_paths:
            f.write(img_path + "\n")

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
