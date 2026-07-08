# Pipeline Commands

> **New to OpenSfM?** See the [quickstart](quickstart.md) for a step-by-step first reconstruction.

The application `bin/opensfm` runs all pipeline commands. The first argument is the command, the second is the dataset path.

```
usage: opensfm [-h] command ...

positional arguments:
  command            Command to run
    extract_geolocation  Import geolocation/orientation from a CSV/TXT file
    extract_metadata     Extract metadata from images' EXIF tags
    detect_features      Compute features for all images
    match_features       Match features between image pairs
    create_rig           Create rig cameras and assignments
    create_tracks        Link pairwise matches into tracks
    convert_gcp          Convert GCPs between gcp_list.txt and JSON
    reconstruct          Compute the reconstruction
    crop_reconstruction  Crop reconstruction to N images around a center
    reconstruct_from_prior  Reconstruct from a prior reconstruction
    correct_rolling_shutter  Trim/correct features for rolling shutter distortion
    bundle               Bundle a reconstruction
    mesh                 Add Delaunay meshes to the reconstruction
    undistort            Save undistorted images
    dense_equalize       Estimate per-image radiometric corrections (gain + vignetting) from tracks
    dense_clustering     Dense stage 1: clusters, neighbours and depth ranges
    compute_depthmaps    Dense stage 2: compute and clean depthmaps
    fuse_depthmaps       Dense stage 3: fuse cleaned depthmaps per cluster
    dense_merging        Dense stage 4: merge into the dense cloud, mesh, DSM and ortho
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

### `extract_geolocation`

Imports per-image geolocation (and optionally orientation) from an external CSV/TXT file and writes/updates `exif_overrides.json`. Run it **before** `extract_metadata`, then (re)run `extract_metadata` so the imported values flow into the reconstruction. Imported GPS replaces any GPS read from the image EXIF.

```bash
bin/opensfm extract_geolocation --geotag-file geotags.txt --crs EPSG:2056 path/to/dataset
```

Options:
- `--geotag-file` (required) — path to the CSV/TXT file with image geolocations.
- `--crs` (default `WGS84`) — coordinate reference system of the position columns, e.g. `WGS84` or `EPSG:2056`.

The format is line-based and tolerant: each line must contain a token matching a dataset image name; the delimiter (comma, tab or space) is auto-detected and `#`/blank lines are ignored. Three consecutive numeric tokens are read as the position — `latitude longitude altitude` for `WGS84`, or `easting northing altitude` for a projected `--crs`. Optional further triplets are recognised by magnitude: a triplet of small values (all `< 0.5`) is taken as position standard deviations (m), and another triplet as `yaw pitch roll` angles in degrees (stored as an OPK orientation prior). Example:

```
img_0001.jpg, 2681192.57, 1250342.89, 605.29, 105.98, 13.89, -6.94, 0.03, 0.05, 0.08
#               easting      northing     alt    yaw     pitch   roll  σx    σy    σz
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

On GPS-denied or no-GPS captures, pair selection can leave the matching graph split into disconnected components, which `reconstruct` turns into separate partial reconstructions. Set `force_match_components: true` to run an extra pass at the end of `match_features` that matches image pairs across the components: exhaustively when a component pair has few candidate pairs (`matching_components_exhaustive_cap`), otherwise pruned by VLAD image similarity (`matching_components_vlad_neighbors`). The component counts before/after are added to `reports/matches.json`.

Key config: `matching_gps_distance`, `matching_gps_neighbors`, `matching_time_neighbors`, `matching_order_neighbors`, `lowes_ratio`, `matcher_type`, `force_match_components`. See [configuration](configuration.md#pair-selection).

### `create_rig`

Creates the rig model (`rig_cameras.json`) and the image-to-rig assignments (`rig_assignments.json`) for multi-camera rigs. Optional; only needed for rig-mounted captures. See [rig.md](rig.md) for the full workflow (`pattern` vs `assignments` methods, `sfm` vs `metadata` calibration).

### `create_tracks`

Links pairwise matches into feature point tracks stored in `tracks.csv`. A track is a set of feature points from different images recognized as the same physical point.

Key config: `min_track_length`. See [configuration](configuration.md#tracks).

Format of `tracks.csv`:

```
image_name  track_id  feature_index  normalized_x  normalized_y  size  R  G  B
02.jpg      1479      2594           0.0379803     -0.0481853    0.00155505  155  145  143
```

### `convert_gcp`

Converts ground control points between the two supported formats, backing up the existing target file (`.bak`) first.

```bash
bin/opensfm convert_gcp --from txt  path/to/dataset   # gcp_list.txt → ground_control_points.json
bin/opensfm convert_gcp --from json path/to/dataset   # ground_control_points.json → gcp_list.txt
```

See [ground_control_points.md](ground_control_points.md) for the two formats and their coordinate conventions.

### `reconstruct`

Runs the incremental reconstruction process to find 3D positions of tracks (structure) and camera positions (motion). Output is stored in `reconstruction.json`.

Key config: `five_point_algo_threshold`, `triangulation_threshold`, `bundle_new_points_ratio`, `bundle_interval`, `local_bundle_radius`. See [configuration](configuration.md#incremental-reconstruction).

### `bundle`

Runs a standalone bundle-adjustment pass over an existing reconstruction (intrinsics, poses, points, and GPS/GCP/rig priors). Useful to re-optimize after editing a reconstruction or changing bundle config.

```bash
bin/opensfm bundle --input reconstruction.json --output reconstruction.bundled.json path/to/dataset
```

### `reconstruct_from_prior`

Grows a reconstruction starting from a prior one (a partial or externally supplied reconstruction) instead of bootstrapping from scratch — useful for guided / aerotriangulation-style reconstruction.

```bash
bin/opensfm reconstruct_from_prior --input prior.json --output reconstruction.json path/to/dataset
```

### `crop_reconstruction`

Keeps the `N` shots closest to the (optionally shifted) center of the reconstruction and writes the result to `reconstruction_cropped.json`. Handy for extracting a manageable region from a large reconstruction.

```bash
bin/opensfm crop_reconstruction -n 50 --shift -1.0 -1.0 path/to/dataset
```

Options: `-n/--n` (images to keep, default 50), `--shift` (X Y shift of the center, default `0 0`).

### `correct_rolling_shutter`

Corrects the detected features of an aerial reconstruction for rolling-shutter distortion, using per-image speed estimates (from EXIF speed information, or computed from the camera motion of an existing reconstruction).

```bash
bin/opensfm correct_rolling_shutter --rolling-shutter-readout 30 path/to/dataset
```

Options: `--rolling-shutter-readout` — sensor readout time in milliseconds (default 30).

### `mesh`

Computes a rough triangular mesh of the scene seen by each image, used for smooth motion simulation in the viewer. Output is `reconstruction.meshed.json`.

### `undistort`

Creates undistorted versions of the reconstruction, tracks, and images. Required before `compute_depthmaps`.

Key config: `undistorted_image_format`, `undistorted_image_max_size`. See [configuration](configuration.md#undistortion).

### `dense_equalize`

Estimates per-image radiometric corrections — a per-channel gain (exposure/white balance) and a radial vignetting falloff — from the track correspondences (the same 3D point seen by several images should have the same colour), and writes `equalization.json` in the undistorted subfolder for the dense stages to divide out. Optional; run after `undistort`.

Options: `--subfolder` — undistorted subfolder where to load and store data (default `undistorted`).

### Dense reconstruction: `dense_clustering`, `compute_depthmaps`, `fuse_depthmaps`, `dense_merging`

Dense reconstruction runs as four stages on an undistorted reconstruction, handed off through the `undistorted/depthmaps/` folder:

```bash
bin/opensfm dense_clustering  path/to/dataset
bin/opensfm compute_depthmaps path/to/dataset
bin/opensfm fuse_depthmaps    path/to/dataset
bin/opensfm dense_merging     path/to/dataset   # add --georeferenced for LAS/LAZ + DSM/ortho in the output CRS
```

- **`dense_clustering`** — groups covisible shots into clusters and computes per-shot neighbours and depth ranges (`clusters.json`, `neighbors_*.json`, `depth_ranges.json`, ...).
- **`compute_depthmaps`** — GPU PatchMatch depthmaps followed by a consistency/visibility cleaning pass; writes per-shot `*.clean.npz`. Requires `dense_clustering` to have run first.
- **`fuse_depthmaps`** — sparse-voxel-octree TSDF fusion per cluster → `fused_batch_*.ply` (plus `mesh_batch_*.ply` and DSM/ortho tiles when enabled).
- **`dense_merging`** — merges the batches into the final products: `fused.ply`, `mesh.ply` (Surface Nets), `dsm.tif`, `ortho.tif`, optional `fused.las` / `fused.laz`, and the `point_cloud/` octree tiles. Pass `--georeferenced` to write the LAS/LAZ and DSM/ortho in the output coordinate system (the projected GCP CRS if one is given, otherwise a UTM zone derived from the reference position).

Key config: [Depth Estimation](configuration.md#depth-estimation-patchmatch-opencl), [Fusion](configuration.md#fusion), [DSM and Orthophoto](configuration.md#dsm-and-orthophoto) and [Octree Tiling](configuration.md#octree-tiling).

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

### `export_geocoords`

Reprojects reconstruction outputs from the internal topocentric frame into a geographic/projected coordinate system given as a PROJ.4 string. `--proj` is required; choose at least one export toggle.

```bash
bin/opensfm export_geocoords --proj '+proj=utm +zone=32 +north +datum=WGS84' \
    --reconstruction --image-positions path/to/dataset
```

Options:
- `--proj` (required) — target PROJ.4 projection string.
- `--reconstruction` — write `reconstruction.geocoords.json`.
- `--image-positions` — write `image_geocoords.tsv` (one row per image).
- `--transformation` — write the 4×4 transformation matrix to `geocoords_transformation.txt`.
- `--dense` — reproject the dense cloud to `undistorted/depthmaps/merged.geocoords.ply`.
- `--output` — override the output file path.

> For georeferenced **dense** products (point cloud, mesh, DSM, orthophoto, LAS/LAZ), prefer `dense_merging --georeferenced`, which targets the current `fused.ply` pipeline and the derived output CRS. `export_geocoords --dense` operates on the legacy `merged.ply`.

### Other exporters

Convert a reconstruction to third-party formats:

- **`export_ply`** — `reconstruction.ply` (sparse cloud + camera positions). `--depthmaps` additionally exports per-image depthmaps as point clouds.
- **`export_openmvs`** — OpenMVS scene at `undistorted/openmvs/scene.mvs` (run `undistort` first).
- **`export_visualsfm`** — VisualSFM NVM_V3 file at `undistorted/reconstruction.nvm`.
- **`export_pmvs`** — PMVS workspace under `pmvs/`.
- **`export_bundler`** — Bundler `list.txt` + `bundle.out` under `bundler/`.
- **`export_colmap`** — COLMAP database + model (cameras/images/points3D) under `colmap_export/`.


## Configuration

SfM options are set in `DATASET_PATH/config.yaml`. Any key present overrides the default value from `opensfm/config.py`.

See the full [configuration reference](configuration.md) for all parameters, defaults, and descriptions.

### Workflow presets (`configs/`)

The repository ships ready-made `config.yaml` presets, tuned for common capture types, in the `configs/` folder. A preset is just a normal `config.yaml`: copy the one that matches your data into the dataset, then tweak individual keys as needed.

```bash
cp configs/aerial.yaml path/to/dataset/config.yaml
```

| Preset | Capture type | Tuned for |
| ------ | ------------ | --------- |
| `aerial.yaml` | Nadir / oblique drone & aerial mapping | GPS-tagged flights: `vertical` orientation prior, GPS-preempted VLAD pair selection, triangulation-based matching rounds, DSM/ortho enabled |
| `terrestrial.yaml` | Ground-level / walk-around capture | `horizontal` orientation prior, SfM planar prior, more sequential (time/order) neighbours, DSM/ortho enabled |
| `object.yaml` | Close-range single object / turntable | Unordered all-around matching (no GPS preemption), planar prior, TSDF photometric refinement, DSM/ortho disabled |

All three share large feature budgets (`feature_min_frames: 20000`, `feature_process_size: 4096`) and the `brown` default projection; they differ mainly in pair selection, alignment prior, and the dense/DSM settings above. See the [configuration reference](configuration.md) for every parameter they set.


## Ground Control Points

See [ground_control_points.md](ground_control_points.md) for GCP setup (JSON and TXT formats, supported projections).
