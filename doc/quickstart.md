# Quickstart

## Reconstructing the Example Dataset

An example dataset is available at `data/berlin`. Reconstruct it by running:

```bash
bin/opensfm_run_all data/berlin
```

This runs the entire SfM pipeline followed by the [dense stages](#dense-point-clouds). It produces `data/berlin/reconstruction.meshed.json` and the dense outputs in `data/berlin/undistorted/depthmaps/`. Dense reconstruction requires an OpenCL GPU.

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
python viewer/server.py -d path/to/dataset
```

## Dense Point Clouds

`opensfm_run_all` runs these steps for you. To run them individually after `reconstruct`:

```bash
bin/opensfm undistort data/berlin
bin/opensfm dense_clustering data/berlin
bin/opensfm dense_equalize data/berlin
bin/opensfm compute_depthmaps data/berlin
bin/opensfm fuse_depthmaps data/berlin
bin/opensfm dense_merging data/berlin
```

This runs dense multi-view stereo and produces a dense point cloud at `data/berlin/undistorted/depthmaps/fused.ply` (along with a `mesh.ply`, a DSM and an orthophoto). Visualize the cloud with [MeshLab](http://www.meshlab.net/) or any viewer supporting [PLY](http://paulbourke.net/dataformats/ply/) files.

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
