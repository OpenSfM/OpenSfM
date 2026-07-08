# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is adapted from the hand-crafted [.github/copilot-instructions.md](.github/copilot-instructions.md), which has proven to work well for coding agents on this repo. Keep the two in sync when changing project-wide guidance.

## 1. Architecture & "Big Picture"
OpenSfM is a Structure-from-Motion library with a hybrid architecture:
- **Core Logic (C++)**: Heavy computational tasks (geometry, bundle adjustment, features, dense, map) live in C++ under `opensfm/src/lib/`. Each library (e.g., `geometry`, `bundle`, `sfm`, `map`) is laid out as `src/` (implementation), `test/` (gtest `*_test.cc`), and `python/` (PyBind11 bindings). They are exposed to Python as compiled modules (e.g., `pygeometry`, `pybundle`, `pymap`) and imported from the root package (e.g., `from opensfm import pygeometry`).
- **Pipeline Layout (Python)**: Application logic, pipeline orchestration, and the CLI are in the Python `opensfm/` package.
- **Two-layer commands**: `opensfm/commands/<name>.py` are thin CLI wrappers (subclass `CommandBase`, define `name`/`help`, parse args) that delegate to `opensfm/actions/<name>.py`, where the real `run_dataset(data, ...)` logic lives. Add new pipeline steps in both places.
- **Data Abstraction**: The `DataSet` class (`opensfm/dataset.py`) is the central definition for filesystem interactions. All pipeline stages read/write to a hardcoded directory structure (`images/`, `config.yaml`, `reconstruction.json`) managed by this class.
- **State Management**: The `Reconstruction` class (`opensfm/types.py`) wraps the C++ `pymap.Map` object. This pattern (Python class holding a C++ handle) is pervasive.

## 2. Key Developer Workflows

> Always activate the conda environment at least once before building, running, or testing: `conda activate opensfm`.

### Building
The project uses `scikit-build-core` to compile C++ extensions (config in `pyproject.toml`).
- **Full Build**: `pip install -e .[test]` — builds C++ extensions plus tests and installs the package in editable mode. The build directory is `build/` (compiled `.so` files live there via editable redirect mode).
- **Rebuild C++**: Re-run `pip install -e .` after changing `.cc` files.
- **Dependencies**: Python deps are in `pyproject.toml`. The outer environment (C++ toolchain, libraries) is set up through `conda` and `conda.yml`. Any change to `conda.yml` requires recreating the env: `conda env create --file conda.yml --yes` then `conda activate opensfm`.

### Testing
- **Frameworks**:
    - Python: `pytest`, tests in `opensfm/test/test_*.py`.
    - C++: `gtest`, tests in `opensfm/src/lib/<XXX>/test/*_test.cc`.
- **Run Python tests**: `pytest opensfm/test/`. First run `export LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so` (tcmalloc is mandatory). Run a single test with `pytest opensfm/test/test_foo.py::test_bar`. `not slow` excludes long integration tests.
- **Run C++ tests**: built with `OPENSFM_BUILD_TESTS=ON`; run via `ctest --output-on-failure` from the CMake build directory.
- **Coverage**: `./run_coverage.bash` (handles conda activation; defaults to `not slow`).
- **Style**: favor many small tests with few asserts, ideally one function per test; only 1-2 larger integration tests per file. Prefer toy/synthetic examples that verify correctness, not just return semantics.
- **Synthetic Data**: Python tests rely heavily on synthetic scenes generated in `opensfm/test/conftest.py` (e.g., fixtures `scene_synthetic`, `scene_synthetic_cube`).

### Running the Pipeline
- **Entry Point**: `bin/opensfm` (bash, sets `LD_PRELOAD` for tcmalloc) → `bin/opensfm_main.py` → `commands.command_runner`.
- **Example**: `bin/opensfm reconstruct path/to/dataset` invokes `Command.run_impl` in `opensfm/commands/reconstruct.py`, which calls `actions/reconstruct.py::run_dataset`.

## 3. Coding Conventions & Patterns

### Python/C++ Interop
- **Wrappers**: Prefer Python wrappers over raw C++ objects (e.g., use `opensfm.types.Reconstruction` instead of `opensfm.pymap.Map` where possible).

### Code Philosophy
- **Readability**: Prioritize clear, simple, maintainable code. You're a seasoned engineer who knows real-world trade-offs — apply frameworks/design patterns only when strictly justified now, not for hypothetical futures. Use descriptive names; break complex functions into smaller, focused ones.
- **Documentation**: Docstrings for all public functions/classes. For complex algorithms, comment the rationale and key steps.
- **C++ Style**: Follow existing style; format with `clang-format` (`.clang-format` at repo root). Key points:
    * Stick to STL and Eigen for data structures. Avoid OpenCV datastructures in C++, except for image resizing.
    * Use SoA (Structure of Arrays) for performance-critical data like point clouds.
    * Minimize heap allocation/fragmentation: prefer stack-allocated buffers; reuse heap buffers; avoid tiny allocations inside long loops.
- **Design**: Reuse the core lib's data structures. Manipulating large reconstructions efficiently is paramount — minimize copies, apply Data-Oriented Design, and leverage GPU acceleration where possible.

### Performance & Data-Safety Rules
- **Assume large datasets (10k+ images)**: never reload data that an earlier stage of the same process already loaded. Pass loaded EXIFs/matches along, and run extra passes where the `feature_loader`/`vlad` caches are still warm (i.e. before `matching.clear_cache()`) — a standalone command is a cold process that reloads everything.
- **Earlier pipeline outputs are immutable**: features, matches and EXIF written by a previous step must never be deleted or overwritten by a later step — additive writes only (see `matching.save_matches_merging`). Domain experts refuse anything else as too fragile.
- **Reuse before writing**: pair selection, matching and graph code almost always exists. `pairs_selection.match_candidates_from_metadata` generates pairs between two candidate sets and honors `config_override` (all strategies at 0 = all pairs); see also `matching.match_images_with_pairs` and `tracking.load_matches`.

### Type Hinting
- Codebase uses `pyre-strict`. New code needs complete type annotations; files often carry a `# pyre-strict` header.

### Configuration
- **Definition**: Default parameters are in `opensfm/config.py` (dataclass `OpenSfMConfig`) — also the reference doc for tunable parameters.
- **Overrides**: Overridden by `config.yaml` in the dataset directory.
- **Access**: `data.config['param_name']` where `data` is a `DataSet`.

## 4. Essential Files
- `opensfm/dataset.py`: **READ THIS FIRST** for file I/O — defines where every file lives.
- `opensfm/types.py`: Key data structures (`Reconstruction`, `Shot`, `Camera`).
- `opensfm/config.py`: All tunable parameters.
- `opensfm/src/lib/map/map.cc`: C++ backing implementation for `Reconstruction`.
- `opensfm/commands/` + `opensfm/actions/`: CLI wrappers and their implementations.

## 5. Common Pitfalls
- **Direct File Access**: Avoid manual `open()`. Use `dataset.load_*` / `dataset.save_*` to stay consistent with the expected directory structure.
- **Geometry Types**: Be careful with rotation representations (angle-axis vs matrices). Check `pygeometry` helpers / `opensfm/src/lib/geometry/` when behavior is unclear.
- **Skipping Conda**: Activate the conda environment at least once before building/running.
- **tcmalloc**: Required at runtime — set `LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so` (the `bin/opensfm` wrapper does this for you).

## 6. General Notes
- **Multi-Platform**: Runs on Windows, Linux, macOS. Avoid platform-specific code; use cross-platform libraries (Qt for GUI, STL/Eigen for C++).
- **Code edits**: Favor `sed`/`awk` for complex code replacements over ad-hoc Python scripts that open/write files directly, so changes stay clear and auditable.

## 7. Policy for Coding Agents
- **No Git operations**: This is a shared repository. Do **not** perform Git operations (commit, push, pull, merge, rebase, etc.) — leave them to human developers so proper code review and collaboration are preserved. Only do so if the user explicitly asks in the current session.
