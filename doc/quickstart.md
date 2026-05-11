# Quickstart

## Reconstructing the Example Dataset

An example dataset is available at `data/berlin`. Reconstruct it by running:

```bash
bin/opensfm_run_all data/berlin
```

This runs the entire SfM pipeline and produces `data/berlin/reconstruction.meshed.json` as output.

## Running in Docker

First, build the OpenSfM Docker image as described in [building](building.md).

Start a Docker container, mounting the `data/` folder:

```bash
docker run -it -p 8080:8080 -v ${PWD}/data/:/data/ opensfm.ubuntu24 /bin/bash
```

Inside the container, run the reconstruction:

```bash
bin/opensfm_run_all /data/berlin/
```

When done, exit with Ctrl+d. The model will be available in the `data/` directory.

## Viewer

A web-based viewer is included. Start it with:

```bash
python viewer/server.py path/to/dataset
```

## Dense Point Clouds

For a denser point cloud:

```bash
bin/opensfm undistort data/berlin
bin/opensfm compute_depthmaps data/berlin
```

This runs dense multiview stereo and produces a denser point cloud at `data/berlin/undistorted/depthmaps/merged.ply`. Visualize it with [MeshLab](http://www.meshlab.net/) or any viewer supporting [PLY](http://paulbourke.net/dataformats/ply/) files.

## Reconstructing Your Own Images

1. Put images in `data/DATASET_NAME/images/`
2. Optionally copy `data/berlin/config.yaml` to `data/DATASET_NAME/config.yaml`
3. Run the full pipeline:

```bash
bin/opensfm_run_all data/DATASET_NAME
```

Or run steps individually (see [pipeline commands](using.md)):

```bash
bin/opensfm extract_metadata data/DATASET_NAME
bin/opensfm detect_features  data/DATASET_NAME
bin/opensfm match_features   data/DATASET_NAME
bin/opensfm create_tracks    data/DATASET_NAME
bin/opensfm reconstruct      data/DATASET_NAME
```

See [dataset structure](dataset.md) for the expected folder layout and [configuration reference](configuration.md) for tuning options.
