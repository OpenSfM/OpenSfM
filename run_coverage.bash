#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-opensfm}"
PYTEST_MARK_EXPRESSION="${PYTEST_MARK_EXPRESSION:-not slow}"

log() {
    echo "[coverage] $*"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

activate_conda_env() {
    local conda_base
    local had_nounset=0

    require_command conda
    if [[ $- == *u* ]]; then
        had_nounset=1
        set +u
    fi
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    if [[ $had_nounset -eq 1 ]]; then
        set -u
    fi
}

log "Activating conda environment: $CONDA_ENV_NAME"
activate_conda_env

require_command pip
require_command python

PYTEST_ARGS=("$@")

log "Removing build/ to ensure coverage instrumentation is rebuilt"
rm -rf build

log "Installing OpenSfM with coverage-enabled CMake settings"
pip install \
    --config-settings=cmake.define.OPENSFM_ENABLE_COVERAGE=ON \
    --config-settings=cmake.define.OPENSFM_BUILD_TESTS=ON \
    -e '.[test]'

require_command ctest
require_command gcovr

log "Running C++ tests"
ctest --test-dir build --output-on-failure

if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/lib/libtcmalloc.so" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libtcmalloc.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

log "Running Python tests with coverage"
python -m pytest \
    opensfm/test \
    --cov=opensfm \
    --cov-config=.coveragerc \
    --cov-report=term-missing \
    --cov-report=xml:coverage-python.xml \
    -m "$PYTEST_MARK_EXPRESSION" \
    "${PYTEST_ARGS[@]}"

log "Generating C++ coverage report"
gcovr \
    --root . \
    build \
    --filter 'opensfm/src/lib/' \
    --exclude 'opensfm/src/lib/third_party/' \
    --exclude '.*/test/.*' \
    --exclude 'opensfm/src/lib/testing_main.cc$' \
    --xml-pretty \
    --output coverage-cpp.xml \
    --print-summary

log "Generating SVG badge and JSON summary"
python .github/scripts/generate_coverage_badge.py \
    coverage-python.xml \
    coverage-cpp.xml \
    --output badges/coverage.svg \
    --summary-json badges/coverage-summary.json

log "Coverage artifacts updated: coverage-python.xml, coverage-cpp.xml, badges/coverage.svg, badges/coverage-summary.json"