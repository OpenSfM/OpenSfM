# Ground Control Points

When EXIF data contains GPS location, OpenSfM uses it to georeference the reconstruction. Ground control points (GCPs) can be used for additional accuracy.

GCPs are landmarks visible in images with known geospatial positions (latitude, longitude, altitude). A single GCP can be observed in one or more images.

OpenSfM uses GCPs in two steps:
- **Alignment**: globally moves the reconstruction to align observed GCPs with their GPS positions. Requires at least two observations per GCP.
- **Bundle adjustment**: uses GCP observations as constraints to refine the reconstruction. No minimum number of observations required.

Key config parameters: `bundle_use_gcp`, `gcp_horizontal_sd`, `gcp_vertical_sd`, `gcp_global_weight`, `gcp_annealing_steps`. See [configuration reference](configuration.md#bundle-adjustment).

## JSON Format

Add a file named `ground_control_points.json` at the dataset root:

```json
{
  "points": [
    {
      "id": "gcp_id",
      "position": {
        "latitude": 52.519,
        "longitude": 13.400,
        "altitude": 14.946
      },
      "observations": [
        {
          "shot_id": "image.jpg",
          "projection": [0.1, -0.2]
        }
      ]
    }
  ]
}
```

Coordinates are in [WGS84](https://en.wikipedia.org/wiki/World_Geodetic_System). Altitude is optional. Projections are in [normalized image coordinates](geometry.md#normalized-image-coordinates).

## TXT Format

Add a file named `gcp_list.txt` at the dataset root:

- First line: projection name
- Following lines: one observation per line:

  ```
  <geo_x> <geo_y> <geo_z> <im_x> <im_y> <image_name>
  ```

  Where `<im_x> <im_y>` are pixel coordinates and `<geo_z>` can be `NaN` if altitude is unknown.

**Supported projections:**

- `WGS84`: `geo_x = longitude`, `geo_y = latitude`, `geo_z = altitude`
- `WGS84 UTM 32N` (or any UTM zone): `geo_x = E`, `geo_y = N`, `geo_z = altitude`
- Any valid [proj4](http://proj4.org/) string, e.g. `+proj=utm +zone=32 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs`

**Example `gcp_list.txt`:**

```
WGS84
13.400740745 52.519134104 12.0792090446 2335.0 1416.7 01.jpg
13.400740745 52.519134104 12.0792090446 2639.1 938.0  02.jpg
13.400502446 52.519251158 16.7021233002 766.0  1133.1 01.jpg
```
