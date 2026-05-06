#!/bin/bash

# This script updates the conda lock files for all platforms based on the conda.yml
# It uses conda-lock to generate lock files for linux-64, osx-64, and win-64 platforms.

# It must be run from the conda environment where conda-lock will be installed
conda install -c conda-forge conda-lock
conda-lock --file conda.yml --lockfile conda-lock.yml --platform linux-64 --platform osx-arm64 --platform win-64 --kind explicit