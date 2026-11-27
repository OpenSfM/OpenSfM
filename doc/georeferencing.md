# Georeferencing & GIS Outputs

OpenSfM reconstructs in a **local topocentric frame** but can ingest geospatial inputs (GPS, external geolocation, ground control points in any CRS) and emit **georeferenced** products (LAS/LAZ, DSM and orthophoto GeoTIFFs). This page explains how coordinates flow through the pipeline and walks through a full large-scale aerial example.

If you only need a quick dense cloud, see [Dense reconstruction & 2D maps](dense.md). This page is for the GIS / mapping user.

---

## The internal frame: topocentric ENU

Reconstructions live in a local **East-North-Up (ENU)** Euclidean frame whose origin is stored in `reference_lla.json` (`{latitude, longitude, altitude}`):

- The origin is chosen automatically from the input GPS (a precision-weighted average of the image positions), or from the GCPs if there is no GPS. The origin **altitude is forced to 0**, so heights are relative to that reference, not to the ellipsoid.
- All shot poses and points in `reconstruction.json` are expressed in this ENU frame. The `TopocentricConverter` (see `opensfm/geo.py`) converts ENU ↔ latitude/longitude/altitude ↔ ECEF.

See [geometry.md](geometry.md) for the full coordinate-system reference.

---

## The output coordinate system

Georeferenced products (LAS/LAZ, DSM/ortho, and the report's coordinate row) are written in an **output CRS that OpenSfM derives automatically**:

1. **If your GCPs are given in a _projected_ CRS, that CRS is used verbatim.**
2. **Otherwise, OpenSfM falls back to the UTM zone** containing the reconstruction's reference point (`EPSG:326xx` in the northern hemisphere, `EPSG:327xx` in the southern).

A geographic GCP CRS (plain `WGS84` / `EPSG:4326`, i.e. lat-lon degrees) counts as **not projected** and therefore triggers the UTM fallback — you can't make a metric raster directly in degrees.

> **To control the output CRS, give your GCPs in the projected CRS you want** (even a handful is enough to set it). With no projected GCP CRS, you get the local UTM zone, which is the right default for most aerial datasets.

This single rule (implemented in `DataSet.output_coordinate_system()` → `geo.decide_output_crs`) is used by all georeferenced exports, so they are always mutually consistent.

CRS strings are parsed by PROJ and accept `WGS84`, `EPSG:xxxx`, compound EPSG (e.g. `EPSG:4979+5773` for a vertical datum), or a raw `+proj=...` string. Missing datum/geoid grids are fetched from the PROJ CDN when `proj_cdn_enabled` is set (with an optional local cache, `proj_grid_cache_dir`).

---

## Supplying inputs in a custom CRS

### External geolocation from a text file

When GPS is not in the image EXIF (or you have a better RTK/PPK solution), import it from a CSV/TXT file with [`extract_geolocation`](using.md#extract_geolocation):

```bash
bin/opensfm extract_geolocation --geotag-file geotags.txt --crs EPSG:2056 path/to/dataset
bin/opensfm extract_metadata path/to/dataset      # rerun so the values reach the reconstruction
```

`--crs` is the CRS of the position columns (e.g. `EPSG:2056` Swiss LV95, or `WGS84`). The file can also carry per-image position standard deviations and yaw/pitch/roll (stored as an OPK orientation prior). The imported values land in `exif_overrides.json` and **override any EXIF GPS**. See the [command reference](using.md#extract_geolocation) for the exact format.

### Ground control points in a custom CRS

GCPs are the most accurate way to georeference, and they also set the output CRS (rule 1 above). Both file formats carry their own CRS:

- `gcp_list.txt` — the first line is the CRS (e.g. `WGS84`, `WGS84 UTM 32N`, `EPSG:2056`, or any proj4 string); each subsequent line is `x y z pixel_x pixel_y image_name`.
- `ground_control_points.json` — a top-level `"crs"` plus points with `position` and normalized-image `observations`.

See [ground_control_points.md](ground_control_points.md) for the full schemas and the two-step way GCPs enter the pipeline (global alignment, then bundle adjustment). Use [`convert_gcp`](using.md#convert_gcp) to convert between the two formats.

---

## Georeferenced outputs

### Dense products — `dense_merging --georeferenced`

The recommended path for georeferenced **dense** products:

```bash
bin/opensfm dense_merging --georeferenced path/to/dataset
```

This writes the LAS/LAZ point cloud and the DSM/ortho GeoTIFFs in the **derived output CRS** (with the CRS embedded in the files). The DSM/ortho are reprojected from the topocentric grid to a north-up grid via GDAL. The `fused.ply` cloud, the `mesh.ply` mesh, and the `point_cloud/` octree always remain topocentric. See [Dense reconstruction & 2D maps](dense.md).

### Sparse products — `export_geocoords --proj`

To reproject the **sparse** reconstruction and/or image positions into an explicit CRS you pass directly:

```bash
bin/opensfm export_geocoords --proj '+proj=utm +zone=32 +north +datum=WGS84' \
    --reconstruction --image-positions path/to/dataset
```

Unlike `dense_merging`, `export_geocoords` takes the CRS from its `--proj` argument rather than deriving it. It writes `reconstruction.geocoords.json`, `image_geocoords.tsv`, and/or the transformation matrix. See the [command reference](using.md#export_geocoords).

### Quality report

`compute_statistics` + `export_report` embed the output CRS and (when present) DSM/ortho thumbnails in `stats/report.pdf`. See [quality_report.md](quality_report.md).

---

## Pre-calibrated cameras and rigs

Aerial and survey workflows often use lab-calibrated cameras and fixed multi-camera rigs. OpenSfM lets you inject and **freeze** that calibration.

### Lab-calibrated intrinsics

Put the known intrinsics in `camera_models_overrides.json` (same schema as `camera_models.json`; the special key `"all"` applies one calibration to every camera). To **hold them fixed** during reconstruction instead of refining them, set in `config.yaml`:

```yaml
optimize_camera_parameters: false
```

With this, the camera parameter blocks are added to bundle adjustment as constant. See [extract_metadata → Providing Your Own Camera Parameters](using.md#providing-your-own-camera-parameters).

### Rigs with fixed relative poses

Define the rig with `rig_cameras.json` (each camera's pose in the rig frame) and `rig_assignments.json` (which shots belong to which rig instance), or generate them with [`create_rig`](rig.md). The relative rig-camera poses are **kept constant by default** (`optimize_rig_parameters: false`); each rig instance's world pose is still optimized against GPS and image observations. See [rig.md](rig.md) for the model and the pattern/metadata calibration methods.

---

## End-to-end: large aerial dataset with GCPs and a custom CRS

A worked example combining everything: a multi-camera rig, lab calibration, RTK geotags in a projected CRS, GCPs, and georeferenced outputs in EPSG:2056.

**1. Lay out the dataset and config.** In `config.yaml`:

```yaml
optimize_camera_parameters: false   # trust the lab calibration
five_point_reversal_check: true     # robustness for near-nadir aerial geometry
processes: 8
```

Place files at the dataset root:
- `images/` — the imagery.
- `camera_models_overrides.json` — lab calibration (use `"all"` or per-camera).
- `rig_cameras.json` + `rig_assignments.json` — the rig model (or create them in step 3).
- `geotags.txt` — per-image RTK positions in EPSG:2056.
- `ground_control_points.json` (or `gcp_list.txt`) — GCPs **with CRS `EPSG:2056`** (this sets the output CRS).

**2. Import geolocation and metadata.**

```bash
bin/opensfm extract_geolocation --geotag-file geotags.txt --crs EPSG:2056 DATA
bin/opensfm extract_metadata DATA
```

**3. Features, matching, rig, tracks.**

```bash
bin/opensfm detect_features DATA
bin/opensfm match_features DATA
bin/opensfm create_rig DATA --method=pattern --calibration-type=metadata --definition='{...}'   # if not pre-written
bin/opensfm create_tracks DATA
```

**4. Reconstruct (GPS + GCP constrained) and densify.**

```bash
bin/opensfm reconstruct DATA
bin/opensfm undistort DATA
bin/opensfm dense_clustering DATA
bin/opensfm dense_equalize DATA
bin/opensfm compute_depthmaps DATA
bin/opensfm fuse_depthmaps DATA
bin/opensfm dense_merging --georeferenced DATA
```

**5. Georeferenced results** (all in EPSG:2056, from the GCP CRS):
- `undistorted/depthmaps/fused.las` / `fused.laz` — dense cloud.
- `undistorted/depthmaps/dsm.tif` / `ortho.tif` — DSM and orthophoto.

**6. Reports and sparse export (optional).**

```bash
bin/opensfm compute_statistics DATA
bin/opensfm export_report DATA
bin/opensfm export_geocoords --proj 'EPSG:2056' --reconstruction --image-positions DATA
```

---

## See also

- [Dense reconstruction & 2D maps](dense.md)
- [Ground control points](ground_control_points.md)
- [Rig models](rig.md)
- [Camera models & coordinate systems](geometry.md)
- [Pipeline commands](using.md) — `extract_geolocation`, `convert_gcp`, `export_geocoords`, `dense_merging`
- [Configuration reference](configuration.md)
