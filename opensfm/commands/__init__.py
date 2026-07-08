# pyre-strict

from types import ModuleType
from typing import List

from . import (
    align_submodels,
    bundle,
    compute_depthmaps,
    compute_statistics,
    convert_gcp,
    create_rig,
    create_submodels,
    create_tracks,
    crop_reconstruction,
    dense_clustering,
    dense_equalize,
    dense_merging,
    detect_features,
    export_bundler,
    export_colmap,
    export_geocoords,
    export_openmvs,
    export_ply,
    export_pmvs,
    export_report,
    export_visualsfm,
    extend_reconstruction,
    extract_geolocation,
    extract_metadata,
    fuse_depthmaps,
    match_features,
    mesh,
    reconstruct,
    reconstruct_from_prior,
    correct_rolling_shutter,
    undistort,
)
from .command_runner import command_runner


opensfm_commands: List[ModuleType] = [
    extract_geolocation,
    extract_metadata,
    detect_features,
    match_features,
    create_rig,
    create_tracks,
    convert_gcp,
    reconstruct,
    crop_reconstruction,
    reconstruct_from_prior,
    correct_rolling_shutter,
    bundle,
    mesh,
    undistort,
    dense_equalize,
    dense_clustering,
    compute_depthmaps,
    fuse_depthmaps,
    dense_merging,
    compute_statistics,
    export_ply,
    export_openmvs,
    export_visualsfm,
    export_pmvs,
    export_bundler,
    export_colmap,
    export_geocoords,
    export_report,
    extend_reconstruction,
    create_submodels,
    align_submodels,
]

try:
    from . import export_rerun
    opensfm_commands.append(export_rerun)
except ImportError:
    pass
