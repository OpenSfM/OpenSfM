# Pipeline Commands

> **New to OpenSfM?** See the [quickstart](quickstart.md) for a step-by-step first reconstruction.

The application `bin/opensfm` runs all pipeline commands. The first argument is the command, the second is the dataset path.

```
usage: opensfm [-h] command ...

positional arguments:
  command            Command to run
    extract_metadata     Extract metadata from images' EXIF tags
    detect_features      Compute features for all images
    match_features       Match features between image pairs
    create_rig           Create rig cameras and assignments
    create_tracks        Link pairwise matches into tracks
    reconstruct          Compute the reconstruction
    crop_reconstruction  Crop reconstruction to N images around a center
    reconstruct_from_prior  Reconstruct from a prior reconstruction
    bundle               Bundle a reconstruction
    mesh                 Add Delaunay meshes to the reconstruction
    undistort            Save undistorted images
    compute_depthmaps    Compute depthmaps and fuse into a dense point cloud
    compute_dsm          Compute a Digital Surface Model from fused point cloud
    compute_statistics   Compute statistics and save them in the stats folder
    export_ply           Export reconstruction to PLY format
    export_openmvs       Export reconstruction to openMVS format
    export_visualsfm     Export reconstruction to NVM_V3 format from VisualSfM
    export_pmvs          Export reconstruction to PMVS
    export_bundler       Export reconstruction to bundler format
    export_colmap        Export reconstruction to COLMAP format
    export_geocoords     Export reconstructions in geographic coordinates
    export_report        Export a PDF report from previously computed statistics
    export_rerun         Export reconstruction to Rerun format (requires rerun-sdk)
    extend_reconstruction  Extend an existing reconstruction with new images
    create_submodels     Split the dataset into smaller submodels
    align_submodels      Align submodel reconstructions
```

### `extract_metadata`

Extracts EXIF metadata from images and stores them in the `exif/` folder and `camera_models.json`.

Data extracted per image:

- `width` and `height`: image size in pixels
- `gps` latitude, longitude, altitude, dop: GPS coordinates and Dilution Of Precision
- `capture_time`: capture time (UNIX timestamp)
- `camera orientation`: EXIF orientation tag (1, 3, 6, or 8)
- `projection_type`: determined from GPano metadata. Supported types:
  - `perspective`: perspective projection with 2nd and 4th order radial distortion
  - `radial`: same with principal point and aspect ratio
  - `simple_radial`: perspective with 2nd order radial, principal point, aspect ratio
  - `brown`: perspective with 2nd/4th/6th radial, two tangential, principal point, aspect ratio
  - `spherical` (or `equirectangular`): spherical projection for 360 images
  - `fisheye`: fisheye with 2nd and 4th order radial distortion
  - `fisheye_opencv`: fisheye identical to OpenCV's fisheye
  - `fisheye62`: fisheye with 1–6th order radial, 2 tangential, principal point, aspect ratio
  - `fisheye624`: fisheye62 plus 4 thin-prism distortion coefficients
  - `dual`: linear interpolation between fisheye and perspective models
- `focal_ratio`: EXIF focal length divided by sensor width
- `make` and `model`: camera make and model
- `camera`: camera ID string

Camera model parameters are chosen by this priority:
1. `camera_models_overrides.json` if the camera ID is present
2. Internal calibration database if the camera ID is present
3. Inferred from EXIF metadata

#### Providing Additional Metadata

When EXIF metadata is missing or incorrect, override it via `exif_overrides.json`:

```json
{
    "image_name.jpg": {
        "gps": {
            "latitude": 52.51891,
            "longitude": 13.40029,
            "altitude": 27.0,
            "dop": 5.0
        }
    }
}
```

Rerun `extract_metadata` after writing this file.

#### Providing Your Own Camera Parameters

Put custom camera parameters in `camera_models_overrides.json` in the project folder (same structure as `camera_models.json`).

Use `"all"` as camera ID to override all cameras:

```json
{
    "all": {
        "projection_type": "perspective",
        "width": 1920,
        "height": 1080,
        "focal": 0.9,
        "k1": 0.0,
        "k2": 0.0
    }
}
```

For fisheye:

```json
{
    "all": {
        "projection_type": "fisheye",
        "width": 1920,
        "height": 1080,
        "focal": 0.5,
        "k1": 0.0,
        "k2": 0.0
    }
}
```

For equirectangular 360:

```json
{
    "all": {
        "projection_type": "equirectangular",
        "width": 2000,
        "height": 1000
    }
}
```

### `detect_features`

Detects feature points in images and stores them in the `features/` folder.

Key config: `feature_type` (`HAHOG`/`SIFT`/`SURF`/`AKAZE`/`ORB`), `feature_process_size`, `feature_min_frames`. See [configuration](configuration.md#features).

### `match_features`

Matches feature points between images and stores results in the `matches/` folder. Determines image pairs to run first, then runs matching for each pair.

Matching can be restricted (and sped up) by GPS distance, capture time, or file name order.

Key config: `matching_gps_distance`, `matching_gps_neighbors`, `matching_time_neighbors`, `matching_order_neighbors`, `lowes_ratio`, `matcher_type`. See [configuration](configuration.md#pair-selection).

### `create_tracks`

Links pairwise matches into feature point tracks stored in `tracks.csv`. A track is a set of feature points from different images recognized as the same physical point.

Key config: `min_track_length`. See [configuration](configuration.md#tracks).

Format of `tracks.csv`:

```
image_name  track_id  feature_index  normalized_x  normalized_y  size  R  G  B
02.jpg      1479      2594           0.0379803     -0.0481853    0.00155505  155  145  143
```

### `reconstruct`

Runs the incremental reconstruction process to find 3D positions of tracks (structure) and camera positions (motion). Output is stored in `reconstruction.json`.

Key config: `five_point_algo_threshold`, `triangulation_threshold`, `bundle_new_points_ratio`, `bundle_interval`, `local_bundle_radius`. See [configuration](configuration.md#incremental-reconstruction).

### `mesh`

Computes a rough triangular mesh of the scene seen by each image, used for smooth motion simulation in the viewer. Output is `reconstruction.meshed.json`.

### `undistort`

Creates undistorted versions of the reconstruction, tracks, and images. Required before `compute_depthmaps`.

Key config: `undistorted_image_format`, `undistorted_image_max_size`. See [configuration](configuration.md#undistortion).

### `compute_depthmaps`

Computes a dense point cloud by computing and merging depth maps. Requires an undistorted reconstruction. Output is in the `depthmaps/` folder; the merged cloud is at `undistorted/depthmaps/merged.ply`.

Key config: `depthmap_resolution`, `depthmap_num_matching_views`, `depthmap_max_image_size`. See [configuration](configuration.md#depth-estimation-patchmatch-opencl).

### `compute_dsm`

Computes a Digital Surface Model (GeoTIFF) from the fused dense point cloud. Requires `compute_depthmaps` to have been run first.

```bash
bin/opensfm compute_dsm path/to/dataset
```

Key config parameters: `dsm_gsd`, `dsm_outlier_threshold`, `dsm_min_count`. See [configuration](configuration.md) for all DSM options.

### `compute_statistics`

Computes statistics stored in JSON at `stats/stats.json`, along with diagram images (`heatmap`, `matchgraph`, `topview`, `residuals`).

### `export_report`

Exports `stats/stats.json` and diagrams as a PDF report at `stats/report.pdf`.

### `export_rerun`

Exports the reconstruction to [Rerun](https://rerun.io/) `.rrd` format for interactive 3D visualization. Requires the `rerun-sdk` Python package.

```bash
pip install rerun-sdk
bin/opensfm export_rerun --output model.rrd path/to/dataset
```

Options: `--reconstruction-index` (default 0), `--proj` (use GCP coordinate system).

### `extend_reconstruction`

Extends an existing reconstruction by resecting new images and triangulating new points. Useful for adding images to a previously computed reconstruction without re-running from scratch.

```bash
bin/opensfm extend_reconstruction --input reconstruction.json --output extended.json path/to/dataset
```

See `opensfm/actions/extend_reconstruction.py`.


## Configuration

SfM options are set in `DATASET_PATH/config.yaml`. Any key present overrides the default value from `opensfm/config.py`.

See the full [configuration reference](configuration.md) for all parameters, defaults, and descriptions.


## Ground Control Points

See [ground_control_points.md](ground_control_points.md) for GCP setup (JSON and TXT formats, supported projections).
