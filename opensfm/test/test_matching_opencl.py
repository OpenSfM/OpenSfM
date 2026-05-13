# pyre-strict
"""Tests for OpenCL-accelerated brute-force descriptor matching."""

import numpy as np
import pytest
from numpy.typing import NDArray
from typing import List, Set, Tuple

from opensfm import pyfeatures
from opensfm.matching import match_brute_force_symmetric


def _random_descriptors(n: int, dim: int, seed: int) -> NDArray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def _plant_matches(
    f1: NDArray, f2: NDArray, n: int, offset1: int, offset2: int, seed: int
) -> None:
    """Copy descriptors from f1[offset1:offset1+n] into f2[offset2:offset2+n] with tiny noise."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, f1.shape[1]).astype(np.float32) * 0.001
    f2[offset2: offset2 + n] = f1[offset1: offset1 + n] + noise


opencl_available: bool = pyfeatures.opencl_matching_available()
skip_no_opencl = pytest.mark.skipif(
    not opencl_available, reason="OpenCL not available"
)


@skip_no_opencl
def test_basic_matching() -> None:
    """Planted near-duplicate descriptors should be matched."""
    dim = 128
    f1 = _random_descriptors(200, dim, 42)
    f2 = _random_descriptors(300, dim, 99)
    _plant_matches(f1, f2, 10, 0, 50, 777)

    result = pyfeatures.match_brute_force_opencl(f1, f2, 0.8)
    assert result.shape[1] == 2
    assert result.shape[0] >= 10

    matches_set: Set[Tuple[int, int]] = {
        (int(r[0]), int(r[1])) for r in result}
    for i in range(10):
        assert (i, 50 + i) in matches_set


@skip_no_opencl
def test_symmetric_matching() -> None:
    """Symmetric matching should return only mutually-consistent pairs."""
    dim = 128
    f1 = _random_descriptors(150, dim, 11)
    f2 = _random_descriptors(200, dim, 22)
    _plant_matches(f1, f2, 5, 10, 30, 333)

    sym_result = pyfeatures.match_brute_force_opencl_symmetric(f1, f2, 0.8)
    asym_result = pyfeatures.match_brute_force_opencl(f1, f2, 0.8)

    assert sym_result.shape[0] <= asym_result.shape[0]
    assert sym_result.shape[0] >= 5

    sym_set: Set[Tuple[int, int]] = {
        (int(r[0]), int(r[1])) for r in sym_result}
    for i in range(5):
        assert (10 + i, 30 + i) in sym_set


@skip_no_opencl
def test_empty_input() -> None:
    """Empty descriptor arrays should return empty matches."""
    f1 = np.zeros((0, 128), dtype=np.float32)
    f2 = _random_descriptors(50, 128, 1)

    result = pyfeatures.match_brute_force_opencl(f1, f2, 0.8)
    assert result.shape[0] == 0
    assert result.shape[1] == 2


@skip_no_opencl
def test_consistency_with_cpu_brute_force() -> None:
    """OpenCL matches should be a superset of CPU symmetric matches (same algorithm)."""
    dim = 128
    f1 = _random_descriptors(80, dim, 44)
    f2 = _random_descriptors(100, dim, 55)
    _plant_matches(f1, f2, 8, 0, 10, 666)

    config = {"lowes_ratio": 0.75}

    gpu_matches = pyfeatures.match_brute_force_opencl_symmetric(f1, f2, 0.75)
    cpu_matches: List[Tuple[int, int]] = match_brute_force_symmetric(
        f1, f2, config
    )

    gpu_set: Set[Tuple[int, int]] = {
        (int(r[0]), int(r[1])) for r in gpu_matches}
    cpu_set: Set[Tuple[int, int]] = set(cpu_matches)

    # Both use brute-force + Lowe's ratio + symmetric intersection,
    # so they should produce identical results.
    assert gpu_set == cpu_set
