# pyre-strict
import numpy as np
from numpy.typing import ArrayLike, NDArray
from opensfm import io, pygeometry, pymap, reconstruction


def unit_vector(x: ArrayLike) -> NDArray:
    return np.array(x) / np.linalg.norm(x)


def test_triangulate_bearings_dlt() -> None:
    rt1 = np.append(np.identity(3), [[0], [0], [0]], axis=1)
    rt2 = np.append(np.identity(3), [[-1], [0], [0]], axis=1)
    b1 = unit_vector([0.0, 0, 1])
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    res, X = pygeometry.triangulate_bearings_dlt(
        [rt1, rt2], np.asarray(
            [b1, b2]), max_reprojection, min_ray_angle, min_depth
    )
    assert np.allclose(X, [0, 0, 1.0])
    assert res is True


def test_triangulate_bearings_dlt_coincident_camera_origins() -> None:
    rt1 = np.append(np.identity(3), [[0], [0], [0]], axis=1)
    rt2 = np.append(np.identity(3), [[0], [0], [0]], axis=1)  # same origin
    b1 = unit_vector([0.0, 0, 1])
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    res, X = pygeometry.triangulate_bearings_dlt(
        [rt1, rt2], np.asarray(
            [b1, b2]), max_reprojection, min_ray_angle, min_depth
    )
    assert res is False


def test_triangulate_bearings_midpoint() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([1.0, 0, 0])
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    valid_triangulation, X = pygeometry.triangulate_bearings_midpoint(
        np.asarray([o1, o2]),
        np.asarray([b1, b2]),
        2 * [max_reprojection],
        min_ray_angle,
        min_depth,
    )
    assert np.allclose(X, [0, 0, 1.0])
    assert valid_triangulation is True


def test_triangulate_bearings_midpoint_coincident_camera_origins() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([0.0, 0, 0])  # same origin
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    valid_triangulation, X = pygeometry.triangulate_bearings_midpoint(
        np.asarray([o1, o2]),
        np.asarray([b1, b2]),
        2 * [max_reprojection],
        min_ray_angle,
        min_depth,
    )
    assert valid_triangulation is False


def test_triangulate_two_bearings_midpoint() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([1.0, 0, 0])
    b2 = unit_vector([-1.0, 0, 1])
    ok, X = pygeometry.triangulate_two_bearings_midpoint(
        np.asarray([o1, o2]), np.asarray([b1, b2])
    )
    assert ok is True
    assert np.allclose(X, [0, 0, 1.0])


def test_triangulate_two_bearings_midpoint_failed() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([1.0, 0, 0])

    # almost parallel. 1e-5 will make it triangulate again.
    b2 = b1 + np.array([-1e-10, 0, 0])

    ok, X = pygeometry.triangulate_two_bearings_midpoint(
        np.asarray([o1, o2]), np.asarray([b1, b2])
    )
    assert ok is False
