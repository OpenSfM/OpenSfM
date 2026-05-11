# Building OpenSfM

## Prerequisites

[Conda](https://docs.conda.io/en/latest/miniconda.html) (or Miniconda) is the only prerequisite on all platforms. The conda environment installs the full toolchain including the C++ compiler, CMake, and all library dependencies.

## Installation

### Linux

```bash
git clone --recursive https://github.com/OpenSfM/OpenSfM
cd OpenSfM
conda create --name opensfm --file conda-linux-64.lock --yes
conda activate opensfm
pip install -e .
```

### macOS (Apple Silicon)

```bash
git clone --recursive https://github.com/OpenSfM/OpenSfM
cd OpenSfM
conda create --name opensfm --file conda-osx-arm64.lock --yes
conda activate opensfm
pip install -e .
```

### Windows

A single script handles everything (Miniconda download, VS Build Tools, environment creation, and build):

```bat
git clone --recursive https://github.com/OpenSfM/OpenSfM
cd OpenSfM
setup.bat
```

`setup.bat` will:
1. Download and install Miniconda if conda is not found
2. Download and install VS Build Tools with the C++ workload if MSVC is not found
3. Create the `opensfm` conda environment from `conda-win-64.lock`
4. Build and install OpenSfM

> Admin rights are only required if VS Build Tools need to be installed.

## Running

Activate the environment, then run the desired pipeline step:

**Linux / macOS:**

```bash
conda activate opensfm
./bin/opensfm COMMAND path/to/dataset
```

**Windows:**

```bat
conda activate opensfm
bin\opensfm.bat COMMAND path\to\dataset
```

See [using.md](using.md) for the full list of available commands.

## Viewer

A web-based 3D viewer is included under `viewer/`. See [using.md](using.md) for details.

## Building Docker Images

Example Dockerfiles are provided for Ubuntu 20.04 and 24.04:

```bash
docker build -t opensfm.ubuntu24 -f Dockerfile.ubuntu24 .
```
