# pyre-strict
from typing import Any, Dict, List, overload, Sequence, Tuple, Union

import numpy as np
import pyproj
from numpy.typing import NDArray

Scalars = Union[float, NDArray]

WGS84_a = 6378137.0
WGS84_b = 6356752.314245


@overload
def ecef_from_lla(lat: float, lon: float,
                  alt: float) -> Tuple[float, float, float]: ...


@overload
def ecef_from_lla(
    lat: NDArray, lon: NDArray, alt: NDArray
) -> Tuple[NDArray, NDArray, NDArray]: ...


def ecef_from_lla(
    lat: Scalars, lon: Scalars, alt: Scalars
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Compute ECEF XYZ from latitude, longitude and altitude.

    All using the WGS84 model.
    Altitude is the distance to the WGS84 ellipsoid.
    Check results here http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm

    >>> lat, lon, alt = 10, 20, 30
    >>> x, y, z = ecef_from_lla(lat, lon, alt)
    >>> np.allclose(lla_from_ecef(x,y,z), [lat, lon, alt])
    True
    """
    a2 = WGS84_a**2
    b2 = WGS84_b**2
    lat = np.radians(lat)
    lon = np.radians(lon)
    L = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * L + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * L + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * L + alt) * np.sin(lat)
    return x, y, z


@overload
def lla_from_ecef(x: float, y: float,
                  z: float) -> Tuple[float, float, float]: ...


@overload
def lla_from_ecef(
    x: NDArray, y: NDArray, z: NDArray
) -> Tuple[NDArray, NDArray, NDArray]: ...


def lla_from_ecef(
    x: Scalars, y: Scalars, z: Scalars
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Compute latitude, longitude and altitude from ECEF XYZ.

    All using the WGS84 model.
    Altitude is the distance to the WGS84 ellipsoid.
    """
    a = WGS84_a
    b = WGS84_b
    ea = np.sqrt((a**2 - b**2) / a**2)
    eb = np.sqrt((a**2 - b**2) / b**2)
    # pyre-ignore[6]: pyre ignores that x,y are all either scalars or arrays
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + eb**2 * b * np.sin(theta) ** 3, p - ea**2 * a * np.cos(theta) ** 3
    )
    N = a / np.sqrt(1 - ea**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


def ecef_from_topocentric_transform(lat: float, lon: float, alt: float) -> NDArray:
    """
    Transformation from a topocentric frame at reference position to ECEF.

    The topocentric reference frame is a metric one with the origin
    at the given (lat, lon, alt) position, with the X axis heading east,
    the Y axis heading north and the Z axis vertical to the ellipsoid.
    >>> a = ecef_from_topocentric_transform(30, 20, 10)
    >>> b = ecef_from_topocentric_transform_finite_diff(30, 20, 10)
    >>> np.allclose(a, b)
    True
    """
    x, y, z = ecef_from_lla(lat, lon, alt)
    sa = np.sin(np.radians(lat))
    ca = np.cos(np.radians(lat))
    so = np.sin(np.radians(lon))
    co = np.cos(np.radians(lon))
    return np.array(
        [
            [-so, -sa * co, ca * co, x],
            [co, -sa * so, ca * so, y],
            [0, ca, sa, z],
            [0, 0, 0, 1],
        ]
    )


def ecef_from_topocentric_transform_finite_diff(
    lat: float, lon: float, alt: float
) -> NDArray:
    """
    Transformation from a topocentric frame at reference position to ECEF.

    The topocentric reference frame is a metric one with the origin
    at the given (lat, lon, alt) position, with the X axis heading east,
    the Y axis heading north and the Z axis vertical to the ellipsoid.
    """
    eps = 1e-2
    x, y, z = ecef_from_lla(lat, lon, alt)
    v1 = (
        (
            np.array(ecef_from_lla(lat, lon + eps, alt))
            - np.array(ecef_from_lla(lat, lon - eps, alt))
        )
        / 2
        / eps
    )
    v2 = (
        (
            np.array(ecef_from_lla(lat + eps, lon, alt))
            - np.array(ecef_from_lla(lat - eps, lon, alt))
        )
        / 2
        / eps
    )
    v3 = (
        (
            np.array(ecef_from_lla(lat, lon, alt + eps))
            - np.array(ecef_from_lla(lat, lon, alt - eps))
        )
        / 2
        / eps
    )
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)
    return np.array(
        [
            [v1[0], v2[0], v3[0], x],
            [v1[1], v2[1], v3[1], y],
            [v1[2], v2[2], v3[2], z],
            [0, 0, 0, 1],
        ]
    )


@overload
def topocentric_from_lla(
    lat: float,
    lon: float,
    alt: float,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[float, float, float]: ...


@overload
def topocentric_from_lla(
    lat: NDArray,
    lon: NDArray,
    alt: NDArray,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[NDArray, NDArray, NDArray]: ...


def topocentric_from_lla(
    lat: Scalars,
    lon: Scalars,
    alt: Scalars,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Transform from lat, lon, alt to topocentric XYZ.

    >>> lat, lon, alt = -10, 20, 100
    >>> np.allclose(topocentric_from_lla(lat, lon, alt, lat, lon, alt),
    ...     [0,0,0])
    True
    >>> x, y, z = topocentric_from_lla(lat, lon, alt, 0, 0, 0)
    >>> np.allclose(lla_from_topocentric(x, y, z, 0, 0, 0),
    ...     [lat, lon, alt])
    True
    """
    T = np.linalg.inv(ecef_from_topocentric_transform(reflat, reflon, refalt))
    # pyre-ignore[6]: pyre gets confused with Scalar vs float vs NDarray
    x, y, z = ecef_from_lla(lat, lon, alt)
    tx = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
    ty = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
    tz = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    return tx, ty, tz


@overload
def lla_from_topocentric(
    x: float,
    y: float,
    z: float,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[float, float, float]: ...


@overload
def lla_from_topocentric(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[NDArray, NDArray, NDArray]: ...


def lla_from_topocentric(
    x: Scalars,
    y: Scalars,
    z: Scalars,
    reflat: float,
    reflon: float,
    refalt: float,
) -> Tuple[Scalars, Scalars, Scalars]:
    """
    Transform from topocentric XYZ to lat, lon, alt.
    """
    T = ecef_from_topocentric_transform(reflat, reflon, refalt)
    ex = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
    ey = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
    ez = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    return lla_from_ecef(ex, ey, ez)


@overload
def gps_distance(latlon_1: Sequence[float],
                 latlon_2: Sequence[float]) -> float: ...


@overload
def gps_distance(
    latlon_1: Sequence[NDArray], latlon_2: Sequence[NDArray]
) -> NDArray: ...
@overload
def gps_distance(latlon_1: NDArray, latlon_2: NDArray) -> NDArray: ...


class TopocentricConverter:
    """Convert to and from a topocentric reference frame."""

    def __init__(self, reflat: float, reflon: float, refalt: float) -> None:
        """Init the converter given the reference origin."""
        self.lat = reflat
        self.lon = reflon
        self.alt = refalt

    @overload
    def to_topocentric(
        self, lat: float, lon: float, alt: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def to_topocentric(
        self, lat: NDArray, lon: NDArray, alt: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]: ...

    def to_topocentric(
        self, lat: Scalars, lon: Scalars, alt: Scalars
    ) -> Tuple[Scalars, Scalars, Scalars]:
        """Convert lat, lon, alt to topocentric x, y, z."""
        # pyre-ignore[6]: pyre gets confused with Scalar vs float vs NDarray
        return topocentric_from_lla(lat, lon, alt, self.lat, self.lon, self.alt)

    @overload
    def to_lla(self, x: float, y: float,
               z: float) -> Tuple[float, float, float]: ...

    @overload
    def to_lla(
        self, x: NDArray, y: NDArray, z: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]: ...

    def to_lla(
        self, x: Scalars, y: Scalars, z: Scalars
    ) -> Tuple[Scalars, Scalars, Scalars]:
        """Convert topocentric x, y, z to lat, lon, alt."""
        # pyre-ignore[6]: pyre gets confused with Scalar vs float vs NDarray
        return lla_from_topocentric(x, y, z, self.lat, self.lon, self.alt)

    def __eq__(self, o: "TopocentricConverter") -> bool:
        return np.allclose([self.lat, self.lon, self.alt], (o.lat, o.lon, o.alt))


def construct_proj_transformer(proj_str: str, inverse: bool = False) -> pyproj.Transformer:
    """
    Construct a pyproj Transformer object, converting between the given projection antod WGS84 (EPSG:4326).
    If inverse is True, the transformation is from WGS84 to the given projection.
    """
    crs_4326 = pyproj.CRS.from_epsg(4326)
    if inverse:
        return pyproj.Transformer.from_proj(crs_4326, pyproj.CRS(proj_str))
    else:
        return pyproj.Transformer.from_proj(pyproj.CRS(proj_str), crs_4326)


def transform_to_proj(
    point: Sequence[float], reference: TopocentricConverter, projection: pyproj.Transformer
) -> List[float]:
    """
    Transform a point defined wrt. the local topocentric frame to a projection
    defined by the given Transformer. We assume the Transformer goes from
    WGS84 to the desired projection.
    """
    assert projection.source_crs.to_epsg(
    ) == 4326, "Transformer source CRS must be WGS84 (EPSG:4326)"

    lat, lon, altitude = reference.to_lla(point[0], point[1], point[2])
    easting, northing = projection.transform(lat, lon)
    return [easting, northing, altitude]


def get_proj_transform_matrix(
    reference: TopocentricConverter, projection: pyproj.Transformer
) -> NDArray:
    """Get the linear transform from reconstruction coords to geocoords."""
    p = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    q = [transform_to_proj(point, reference, projection) for point in p]

    transformation = np.array(
        [
            [q[0][0] - q[3][0], q[1][0] - q[3][0], q[2][0] - q[3][0], q[3][0]],
            [q[0][1] - q[3][1], q[1][1] - q[3][1], q[2][1] - q[3][1], q[3][1]],
            [q[0][2] - q[3][2], q[1][2] - q[3][2], q[2][2] - q[3][2], q[3][2]],
            [0, 0, 0, 1],
        ]
    )
    return transformation


def transform_reconstruction_with_matrix(
    reconstruction: "types.Reconstruction", transformation: NDArray
) -> None:
    """Apply a rigid transformation to a reconstruction in-place."""
    A, b = transformation[:3, :3], transformation[:3, 3]
    A1 = np.linalg.inv(A)

    for shot in reconstruction.shots.values():
        R = shot.pose.get_rotation_matrix()
        shot.pose.set_rotation_matrix(np.dot(R, A1))
        shot.pose.set_origin(np.dot(A, shot.pose.get_origin()) + b)

    for point in reconstruction.points.values():
        point.coordinates = list(np.dot(A, point.coordinates) + b)


def transform_reconstruction_with_proj(
    reconstruction: "types.Reconstruction", transformation: pyproj.Transformer
) -> None:
    """Apply a proj Transformer to a reconstruction in-place."""
    eps = 1e-3
    for rig_instance in reconstruction.rig_instances.values():
        origin = rig_instance.pose.get_origin()

        # Jacobian for rotation update
        p0 = np.array(transform_to_proj(
            origin, reconstruction.reference, transformation))
        px = np.array(transform_to_proj(
            origin + [eps, 0, 0], reconstruction.reference, transformation))
        py = np.array(transform_to_proj(
            origin + [0, eps, 0], reconstruction.reference, transformation))
        pz = np.array(transform_to_proj(
            origin + [0, 0, eps], reconstruction.reference, transformation))
        J = np.column_stack(
            ((px - p0) / eps, (py - p0) / eps, (pz - p0) / eps))

        rig_instance.pose.set_origin(p0)
        rig_instance.pose.set_rotation_matrix(
            rig_instance.pose.get_rotation_matrix() @ np.linalg.inv(J))

    for point in reconstruction.points.values():
        point.coordinates = transform_to_proj(
            point.coordinates, reconstruction.reference, transformation
        )


def gps_distance(
    latlon_1: Union[Sequence[Scalars], NDArray],
    latlon_2: Union[Sequence[Scalars], NDArray],
) -> Scalars:
    """
    Distance between two (lat,lon) pairs.

    >>> p1 = (42.1, -11.1)
    >>> p2 = (42.2, -11.3)
    >>> 19000 < gps_distance(p1, p2) < 20000
    True
    """
    x1, y1, z1 = ecef_from_lla(latlon_1[0], latlon_1[1], 0.0)
    x2, y2, z2 = ecef_from_lla(latlon_2[0], latlon_2[1], 0.0)

    dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    return dis
