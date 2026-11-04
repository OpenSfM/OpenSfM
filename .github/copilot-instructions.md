# OpenSfM AI Coding Instructions

## 1. Architecture & "Big Picture"
OpenSfM is a Structure-from-Motion library with a hybrid architecture:
- **Core Logic (C++)**: Heavy computational tasks (geometry, bundle adjustment, features) are implemented in C++ under `opensfm/src/`. These are exposed to Python as compiled modules (e.g., `pygeometry`, `pymap`, `pfm`) using **PyBind11**.
- **Pipeline Layout (Python)**: The application logic, pipeline orchestration, and user interface are in Python (`opensfm/` package).
- **Data Abstraction**: The `DataSet` class (`opensfm/dataset.py`) is the central definition for filesystem interactions. All pipeline stages read/write to a hardcoded directory structure (`images/`, `config.yaml`, `reconstruction.json`) managed by this class.
- **State Management**: The `Reconstruction` class (`opensfm/types.py`) wraps the C++ `pymap.Map` object. This pattern (Python class holding a C++ handle) is pervasive.

## 2. Key Developer Workflows

### Building
The project uses `scikit-build-core` to compile C++ extensions.
- **Full Build**: `pip install -e .[test]` (Builds C++ extensions and tests, and installs the package in editable mode).
- **Rebuild C++**: Re-running `pip install -e .` is usually required after changing `.cc` files.
- **Dependencies**: Managed via `pyproject.toml` for managing Python dependencies. Outer envinvironment (C++ toolchain, libraries) is set up through `conda` and the `conda.yml` file. Any change to `conda.yml` requires recreating the conda environment with `conda env create --file conda.yml --yes`, then activating it with `conda activate opensfm`. Note that before running the build commands, the conda environment must be activated once. In sandboxed or non-interactive shells where `conda activate` fails, skip activation and call the environment's interpreter directly (`<conda_base>/envs/opensfm/bin/python -m pytest ...`) — everything works without activation. On macOS, set up conda for the shell with `conda init zsh`, not bare `conda init`.

### Testing
- **Framework**: 
    - Python : `pytest` is used, and tests are in files `test_YYY.py` under `opensfm/test`.
    - C++ : `gtest` is used. For a given library XXX, tests are put under `opensfm/src/XXX/test/` as `YYY_test.cc`.
- **Style**: we favor many tests with few asserts and ideally one function to be tested. Only a few (1-2) larger integration tests per file. We also strive to test using toy examples to check for function correctness and not just check return semantics.
- **Location**: `opensfm/test/` for Python tests. C++ tests are in `./cmake_build`.
- **Synthetic Data**: Python tests heavily rely on synthetic scenes generated in `opensfm/test/conftest.py`. Look for fixtures like `scene_synthetic` or `scene_synthetic_cube`.
- **Run Tests**: `pytest opensfm/test/` for Python tests (a single test: `pytest opensfm/test/test_foo.py::test_bar`; `-m "not slow"` excludes long integration tests). `pytest` comes from the `[test]` extra — if missing, `python -m pip install pytest`. C++ tests can be run via `ctest` with `cd cmake_build && ctest --output-on-failure && cd ..`.
- **tcmalloc (Linux only)**: run `export LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so` before running tests or the pipeline (`bin/opensfm` does it for you). On macOS the library is not shipped and nothing is needed.
- **Timing**: `opensfm/test/test_commands.py` runs the full pipeline on `data/berlin` and takes ~8 minutes — it is NOT marked `slow`, so exclude it explicitly from quick sweeps and use a >=10 min timeout when running it. Most other test files finish in seconds.
- **Sample datasets**: `data/berlin` (3 images) and `data/lund` (6 images) ship in the repo for end-to-end checks. To fake dataset I/O in tests, monkeypatch methods on a `SyntheticDataSet` instance (pattern: `test_matching.py::test_match_images`) rather than writing new mocks.

### Running the Pipeline
- **Entry Point**: `bin/opensfm` (shell script) -> `bin/opensfm_main.py`.
- **Commands**: Each CLI command (e.g., `detect_features`, `reconstruct`) is implemented as a module in `opensfm/commands/`.
- **Example**: `bin/opensfm reconstruct path/to/dataset` invokes the `run()` method in `opensfm/commands/reconstruct.py`.

## 3. Coding Conventions & Patterns

### Python/C++ Interop
- **Wrappers**: Do not use C++ objects directly if a Python wrapper exists (e.g., use `opensfm.types.Reconstruction` instead of `opensfm.pymap.Map` where possible). 
- **Extensions**: C++ extensions are imported from the root package (e.g., `from opensfm import pygeometry`).


### Code Philosophy
- **Readability**: Prioritize clear, simple and maintainable code. Avoid complicated constructs : you're an seasoned engineer that knows about real-world trade-offs, not a rookie trying to expose its design patterns knowledge. You'll use frameworks and design pattern when they're strictly justified at the present moment, not for an hypothetical future. Use descriptive variable and function names, and break down complex functions into smaller, focused ones.
- **Documentation**: Document all public functions and classes with docstrings. For complex algorithms, include comments explaining the rationale and key steps.
- **Testing**: Write tests for new features and bug fixes. Use the existing test suite as a reference for style and structure.
- **C++ Style**: Follow the existing C++ style in the codebase. Use `clang-format` for consistent formatting. Key points:
    * Stick to use STL and Eigen for data structures. Avoid using OpenCV datastructures in C++, except for image resizing.
    * Use SoA structuring for performance critical data like point clouds.
    * Strive to minimize heap allocation and fragmentation by using stack-allocated buffer. In case of heap-allocated, re-use buffers, and avoid tiny buffers in long for loops.
- **Design**: For the data, try to re-use core lib's data structures. In addition, it is of upmost importance to be able to manipulate large reconstructions efficiently. We strive to handle large amounts of data, so minimize copy operations, use Data-Oriented Design principles, and leverage GPU acceleration where possible (e.g., for map rendering).

### Performance & Data-Safety Rules
- **Assume large datasets (10k+ images)**: never reload data that an earlier stage of the same process already loaded. Pass loaded EXIFs/matches along, and run extra passes where the `feature_loader`/`vlad` caches are still warm (i.e. before `matching.clear_cache()`) — a standalone command is a cold process that reloads everything.
- **Earlier pipeline outputs are immutable**: features, matches and EXIF written by a previous step must never be deleted or overwritten by a later step — additive writes only; exclude already-processed items upstream instead of reconciling on write (see `matching.bridge_matching_components`). Domain experts refuse anything else as too fragile.
- **Reuse before writing**: pair selection, matching and graph code almost always exists. `pairs_selection.match_candidates_from_metadata` generates pairs between two candidate sets and honors `config_override` (all strategies at 0 = all pairs); see also `matching.match_images_with_pairs` and `tracking.load_matches`.

### Type Hinting
- **Strictness**: The codebase uses `pyre-strict`. Ensure all new code has complete type annotations. A comment `# pyre-strict` is often present at the top of files.

### Configuration
- **Definition**: Default parameters are in `opensfm/config.py` (dataclass `OpenSfMConfig`).
- **Overrides**: Parameters are overridden by `config.yaml` in the dataset directory.
- **Access**: Access config values via `data.config['param_name']` (where `data` is a `DataSet` instance).

## 4. Essential Files & Entry Points
- `opensfm/dataset.py`: **READ THIS FIRST** when dealing with file I/O. Defines where every file lives.
- `opensfm/types.py`: Defines key data structures (`Reconstruction`, `Shot`, `Camera`).
- `opensfm/config.py`: Documentation for all tunable parameters.
- `opensfm/src/map/map.cc`: The backing C++ implementation for `Reconstruction`.
- `opensfm/commands/`: Implementation of individual pipeline steps.
- `opensfm/pairs_selection.py`: `match_candidates_from_metadata(images_ref, images_cand, exifs, data, config_override)` — the single entry point for pair selection between two image sets.
- `opensfm/matching.py`: `match_images` (pair selection + matching), `match_images_with_pairs` (match an explicit pair list), `bridge_matching_components` (cross-component pass), `clear_cache` (drops the feature/VLAD caches).
- `opensfm/tracking.py`: `load_matches(data, images)` iterates the saved per-image match files.

## 5. Common Pitfalls
- **Direct File Access**: Avoid manual `open()` calls. Use `dataset.load_*` and `dataset.save_*` methods to ensure consistency with the expected directory structure.
- **Geometry Types**: Be careful with rotation representations (angle-axis vs matrices). `pygeometry` exposes helper functions; check `opensfm/src/geometry/` for implementation details if behavior is unclear.
- **Match storage**: `DataSet.save_matches` overwrites the whole `matches/{image}_matches.pkl.gz` file, and a pair's matches may be stored under either image (`find_matches` checks both orientations, flipping columns). Add new entries only — never rewrite existing ones.
- **config_override**: matching/pair-selection functions take a `config_override` dict merged over `data.config` — drive existing functions with overrides instead of re-implementing variants.

## 6. General Algorithmic Notes
- **Multi-Platforms**: The codebase is designed to run on Windows, Linux, and macOS. Avoid platform-specific code unless absolutely necessary, and use cross-platform libraries (e.g., Qt for GUI, STL/Eigen for C++).

If `instructions/memory.instructions.md` exists, refer to it for project-specific commands, style guides, and recent progress updates (it is absent until the first MEMUPDATE).

If and only if the prompt iS MEMUPDATE, summarize the current state of the project in instructions/memory.instructions.md, including recent progress, lessons learned, and known technical debt. This information should be concise and focused on actionable insights for developers.

## 7. For coding agents:
- **STRICTLY FORBIDDEN**:
    - Do not perform any Git operations (commits, pushes, pulls, merges, rebases, etc.). This is a shared repository and all Git operations must be performed by human developers to ensure proper code review and collaboration.
    - Do not add new CLI commands: the maintainer keeps the command surface fixed. Optional pipeline behavior goes behind a config value in `opensfm/config.py` consumed by an existing action (e.g. `force_match_components` in `match_features`) — do not add command/action pairs.
- **BEST PRACTICES**:
    - You will favor using bash tools such as sed or awk to perform complex code replacements, and avoid doing complex string manipulation in Python with ad-hoc scripts that open and write to files directly. This is to ensure that code changes are made in a clear and auditable way for humans, and to leverage the power of existing command-line tools for code transformation.