OpenSfM
=======
[![Conda](https://github.com/OpenSfM/OpenSfM/actions/workflows/conda.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/conda.yml) [![Docker Ubuntu 20.04](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu20.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu20.yml) [![Docker Ubuntu 24.04](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu24.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu24.yml) [![Coverage](./badges/coverage.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/coverage.yml)

## Overview
OpenSfM is a Structure from Motion library written in Python. The library serves as a processing pipeline for reconstructing camera poses and 3D scenes from multiple images. It consists of basic modules for Structure from Motion (feature detection/matching, minimal solvers) with a focus on building a robust and scalable reconstruction pipeline. It also integrates external sensor (e.g. GPS, accelerometer) measurements for geographical alignment and robustness. A JavaScript viewer is provided to preview the models and debug the pipeline.


## Getting Started

Install using conda lock files (see [building instructions](doc/building.md)):

**Linux:**
```bash
conda create --name opensfm --file conda-linux-64.lock --yes
conda activate opensfm && pip install -e .
```

**macOS (Apple Silicon):**
```bash
conda create --name opensfm --file conda-osx-arm64.lock --yes
conda activate opensfm && pip install -e .
```

Then reconstruct a dataset:
```bash
conda activate opensfm
./bin/opensfm reconstruct path/to/dataset   # Linux/macOS
bin\opensfm.bat reconstruct path\to\dataset  # Windows
```
## Documentation

**Getting Started**
* [Quickstart](doc/quickstart.md)
* [Building & Installation](doc/building.md)
* [Pipeline Commands](doc/using.md)

**User Guide**
* [Dataset structure](doc/dataset.md)
* [Configuration reference](doc/configuration.md)
* [Ground control points](doc/ground_control_points.md)
* [Rig models](doc/rig.md)
* [Large datasets](doc/large_datasets.md)
* [Quality report](doc/quality_report.md)

**Reference**
* [Camera models & coordinate systems](doc/geometry.md)
* [Reconstruction algorithm](doc/reconstruction.md)
* [Sensor / calibration database](doc/sensor_database.md)
* [Reporting](doc/reporting.md)

**Mathematical Notes**
* [Dense matching](doc/dense_matching.md)
* [Reconstruction merging](doc/merging_notes.md)

## License
OpenSfM is BSD-style licensed, as found in the LICENSE file.