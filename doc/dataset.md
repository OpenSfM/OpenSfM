# Dataset Structure

A dataset is a folder with the following layout. **Input** files are provided by the user; **output** files are created by pipeline commands.

```
project/
│
│── config.yaml                      # Input: pipeline configuration overrides
│── image_list.txt                   # Input (optional): explicit list of images to process
│── images/                          # Input: source images
│   └── image_filename
│── masks/                           # Input (optional): binary masks (0 = ignore region)
│   └── image_filename.png
│── segmentations/                   # Input (optional): semantic segmentation maps
│   └── image_filename.png
│── instances/                       # Input (optional): instance segmentation maps
│   └── image_filename.png
│── gcp_list.txt                     # Input (optional): ground control points (TXT format)
│── ground_control_points.json       # Input (optional): ground control points (JSON format)
│── camera_models_overrides.json     # Input (optional): override camera calibrations
│── exif_overrides.json              # Input (optional): override EXIF metadata per image
│── rig_cameras.json                 # Input (optional): rig camera definitions
│── rig_assignments.json             # Input (optional): image-to-rig-camera assignments
│
│── exif/                            # Output of extract_metadata
│── camera_models.json               # Output of extract_metadata
│── reference_lla.json               # Output of extract_metadata (topocentric origin)
│── features/                        # Output of detect_features
│── matches/                         # Output of match_features
│── tracks.csv                       # Output of create_tracks
│── reconstruction.json              # Output of reconstruct
│── reconstruction.meshed.json       # Output of mesh
│── reports/                         # Output: per-command JSON reports
│   ├── features.json
│   ├── matches.json
│   ├── tracks.json
│   └── reconstruction.json
│── undistorted/                     # Output of undistort
│   ├── images/
│   ├── masks/
│   ├── tracks.csv
│   ├── reconstruction.json
│   └── depthmaps/                   # Output of compute_depthmaps
│       └── merged.ply
│── stats/                           # Output of compute_statistics / export_report
│   ├── stats.json
│   ├── report.pdf
│   ├── topview.png
│   ├── matchgraph.png
│   ├── heatmap_XXX.png
│   └── residuals_XXX.png
│── colmap_export/                   # Output of export_colmap
└── profile.log                      # Profiling log (appended by various commands)
```

> **Note:** Previous versions of OpenSfM used a different folder structure where undistorted data was not grouped into a single folder. Use `bin/migrate_undistort.sh` to port old datasets to the new structure.

See also: [configuration reference](configuration.md) for all `config.yaml` options.

## Reconstruction File Format

The main output of OpenSfM is `reconstruction.json`, containing estimated camera parameters, camera positions, and a sparse set of 3D points. The file is a JSON array of one or more reconstructions (disconnected components).

```
reconstruction.json: [RECONSTRUCTION, ...]

RECONSTRUCTION: {
    "cameras": {
        CAMERA_ID: CAMERA,
        ...
    },
    "shots": {
        SHOT_ID: SHOT,
        ...
    },
    "points": {
        POINT_ID: POINT,
        ...
    },
    "rig_cameras": {              # Present when rigs are used
        RIG_CAMERA_ID: RIG_CAMERA,
        ...
    },
    "rig_instances": {            # Present when rigs are used
        RIG_INSTANCE_ID: RIG_INSTANCE,
        ...
    }
}
```

### Camera

All camera models share `projection_type`, `width`, and `height`. Other parameters depend on the model. See [geometric models](geometry.md) for the projection equations.

| `projection_type`               | Parameters                                                       |
| ------------------------------- | ---------------------------------------------------------------- |
| `perspective`                   | `focal`, `k1`, `k2`                                              |
| `simple_radial`                 | `focal_x`, `focal_y`, `c_x`, `c_y`, `k1`                         |
| `radial`                        | `focal_x`, `focal_y`, `c_x`, `c_y`, `k1`, `k2`                   |
| `brown`                         | `focal_x`, `focal_y`, `c_x`, `c_y`, `k1`, `k2`, `k3`, `p1`, `p2` |
| `fisheye`                       | `focal`, `k1`, `k2`                                              |
| `fisheye_opencv`                | `focal`, `c_x`, `c_y`, `k1`, `k2`, `k3`, `k4`                    |
| `fisheye62`                     | `focal`, `c_x`, `c_y`, `k1`–`k6`, `p1`, `p2`                     |
| `fisheye624`                    | `focal`, `c_x`, `c_y`, `k1`–`k6`, `p1`, `p2`, `s0`–`s3`          |
| `spherical` / `equirectangular` | *(none beyond width/height)*                                     |
| `dual`                          | `focal`, `k1`, `k2`, `transition`                                |

Example (perspective):
```json
{
    "projection_type": "perspective",
    "width": 4000,
    "height": 3000,
    "focal": 0.85,
    "k1": -0.02,
    "k2": 0.001
}
```

### Shot

```
SHOT: {
    "camera": CAMERA_ID,
    "rotation": [X, Y, Z],      # Rotation as angle-axis vector (world → camera)
    "translation": [X, Y, Z],   # Translation (world → camera)
    "gps_position": [X, Y, Z],  # GPS position in the reconstruction reference frame
    "gps_dop": METERS,          # GPS accuracy in meters
    "orientation": NUMBER,      # EXIF orientation tag (1, 3, 6, or 8)
    "capture_time": SECONDS     # Capture time as a UNIX timestamp
}
```

The camera origin in world coordinates is $-R^T t$. See [camera coordinates](geometry.md#camera-coordinates) for the full convention.

### Point

```
POINT: {
    "coordinates": [X, Y, Z],   # Position in world coordinates
    "color": [R, G, B]          # Color (0–255)
}
```
