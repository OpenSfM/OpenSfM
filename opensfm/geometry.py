# pyre-strict
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray

from opensfm import transformations


def rotation_from_angle_axis(angle_axis: NDArray) -> NDArray:
    return pygeometry.rotation_from_angle_axis(np.asarray(angle_axis, dtype=np.float64))


def rotation_from_ptr(pan: float, tilt: float, roll: float) -> NDArray:
    """World-to-camera rotation matrix from pan, tilt and roll."""
    R1 = rotation_from_angle_axis(np.array([0.0, 0.0, roll]))
    R2 = rotation_from_angle_axis(np.array([tilt + np.pi / 2, 0.0, 0.0]))
    R3 = rotation_from_angle_axis(np.array([0.0, 0.0, pan]))
    return R1.dot(R2).dot(R3)


def ptr_from_rotation(
    rotation_matrix: NDArray,
) -> Tuple[float, float, float]:
    """Pan tilt and roll from camera rotation matrix"""
    pan = pan_from_rotation(rotation_matrix)
    tilt = tilt_from_rotation(rotation_matrix)
    roll = roll_from_rotation(rotation_matrix)
    return pan, tilt, roll


def pan_from_rotation(rotation_matrix: NDArray) -> float:
    Rt_ez = np.dot(rotation_matrix.T, [0, 0, 1])
    return np.arctan2(Rt_ez[0], Rt_ez[1])


def tilt_from_rotation(rotation_matrix: NDArray) -> float:
    Rt_ez = np.dot(rotation_matrix.T, [0, 0, 1])
    l = np.linalg.norm(Rt_ez[:2])
    return np.arctan2(-Rt_ez[2], l)


def roll_from_rotation(rotation_matrix: NDArray) -> float:
    Rt_ex = np.dot(rotation_matrix.T, [1, 0, 0])
    Rt_ez = np.dot(rotation_matrix.T, [0, 0, 1])
    a = np.cross(Rt_ez, [0, 0, 1])
    a /= np.linalg.norm(a)
    b = np.cross(Rt_ex, a)
    return np.arcsin(np.dot(Rt_ez, b))


def rotation_from_ptr_v2(pan: float, tilt: float, roll: float) -> NDArray:
    """Camera rotation matrix from pan, tilt and roll.

    This is the implementation used in the Single Image Calibration code.
    """
    tilt += np.pi / 2
    return transformations.euler_matrix(pan, tilt, roll, "szxz")[:3, :3]


def ptr_from_rotation_v2(rotation_matrix: NDArray) -> Tuple[float, float, float]:
    """Pan tilt and roll from camera rotation matrix.

    This is the implementation used in the Single Image Calibration code.
    """
    T = np.identity(4)
    T[:3, :3] = rotation_matrix
    pan, tilt, roll = transformations.euler_from_matrix(T, "szxz")
    return pan, tilt - np.pi / 2, roll


def rotation_from_opk(omega: float, phi: float, kappa: float) -> NDArray:
    """World-to-camera rotation matrix from OPK angles (in radians).

    Omega (ω): rotation about X (East).
    Phi   (φ): rotation about Y (North).
    Kappa (κ): rotation about Z (Up).
    """
    from opensfm import pygeometry

    return np.asarray(pygeometry.rotation_from_opk(omega, phi, kappa))


def opk_from_rotation(
    rotation_matrix: NDArray,
) -> Tuple[float, float, float]:
    """Omega, phi, kappa from world-to-camera rotation matrix (in radians)."""

    opk = pygeometry.opk_from_rotation(
        np.asarray(rotation_matrix, dtype=np.float64))
    return float(opk[0]), float(opk[1]), float(opk[2])


def omega_from_rotation(rotation_matrix: NDArray) -> float:
    return np.arctan2(-rotation_matrix[1, 2], rotation_matrix[2, 2])


def phi_from_rotation(rotation_matrix: NDArray) -> float:
    return np.arcsin(rotation_matrix[0, 2])


def kappa_from_rotation(rotation_matrix: NDArray) -> float:
    return np.arctan2(-rotation_matrix[0, 1], rotation_matrix[0, 0])

def average_rotation(rotations: NDArray) -> NDArray:
    """
    Average rotations in axis-angle representation by converting them to 
    SciPy Rotation objects, averaging them and converting back to axis-angle.
    """
    r = Rotation.from_rotvec(rotations)
    average_rotation = r.mean()
    return average_rotation.as_rotvec()
