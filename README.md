OpenSfM
=======
[![Conda](https://github.com/OpenSfM/OpenSfM/actions/workflows/conda.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/conda.yml) [![Docker Ubuntu 20.04](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu20.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu20.yml) [![Docker Ubuntu 24.04](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu24.yml/badge.svg)](https://github.com/OpenSfM/OpenSfM/actions/workflows/docker_ubuntu24.yml)

## Overview
OpenSfM is a Structure from Motion library written in Python. The library serves as a processing pipeline for reconstructing camera poses and 3D scenes from multiple images. It consists of basic modules for Structure from Motion (feature detection/matching, minimal solvers) with a focus on building a robust and scalable reconstruction pipeline. It also integrates external sensor (e.g. GPS, accelerometer) measurements for geographical alignment and robustness. A JavaScript viewer is provided to preview the models and debug the pipeline.

<p align="center">
  <img src="https://opensfm.org/docs/_images/berlin_viewer.jpg" />
</p>

Checkout this [blog post with more demos](http://blog.mapillary.com/update/2014/12/15/sfm-preview.html)


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

**Windows:** run `setup.bat` — it handles everything automatically.

Then reconstruct a dataset:
```bash
conda activate opensfm
./bin/opensfm reconstruct path/to/dataset   # Linux/macOS
bin\opensfm.bat reconstruct path\to\dataset  # Windows
```
## Documentation

* [Building](doc/building.md)
* [Using OpenSfM](doc/using.md)
* [Dataset structure](doc/dataset.md)
* [Geometric models](doc/geometry.md)
* [Camera coordinate system](doc/camera_coordinate_system.md)
* [Reconstruction algorithm](doc/reconstruction.md)
* [Large datasets](doc/large_datasets.md)
* [Reporting](doc/reporting.md)
* [Quality report](doc/quality_report.md)
* [Rig models](doc/rig.md)
* [Sensor / calibration database](doc/sensor_database.md)
* [Dense matching notes](doc/dense_matching.md)
* [Reconstruction merging notes](doc/merging_notes.md)

## License
OpenSfM is BSD-style licensed, as found in the LICENSE file.  See also the Facebook Open Source [Terms of Use][] and [Privacy Policy][]

[Terms of Use]: https://opensource.facebook.com/legal/terms (Facebook Open Source - Terms of Use)
[Privacy Policy]: https://opensource.facebook.com/legal/privacy (Facebook Open Source - Privacy Policy)
