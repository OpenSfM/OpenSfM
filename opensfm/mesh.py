# pyre-strict
import itertools
import logging
from typing import List, Tuple

import numpy as np
import scipy.spatial
from numpy.typing import NDArray
from opensfm import pygeometry, pymap, types


logger: logging.Logger = logging.getLogger(__name__)


def triangle_mesh_from_points(
    shot: pymap.Shot,
    point_coords: NDArray,
) -> Tuple[List[List[float]], List[List[int]]]:
    """Build a Delaunay mesh for a shot from an explicit array of 3D points.

    This variant does **not** require a TracksManager and is intended for
    the preview pipeline where tracks are constructed ad-hoc.

    Args:
        shot: The shot for which to build the mesh.
        point_coords: (N, 3) array of world-space 3D point coordinates
            visible in the shot.

    Returns:
        (vertices, faces) where *vertices* is a list of [x,y,z] and
        *faces* is a list of [i,j,k] index triples.
    """
    cam = shot.camera

    if cam.projection_type in [
        "perspective",
        "brown",
        "radial",
        "simple_radial",
    ]:
        return _triangle_mesh_from_points_perspective(shot, point_coords)
    elif cam.projection_type in [
        "fisheye",
        "fisheye_opencv",
        "fisheye62",
        "fisheye624",
        "dual",
    ]:
        return _triangle_mesh_from_points_fisheye(shot, point_coords)
    elif pygeometry.Camera.is_panorama(cam.projection_type):
        return _triangle_mesh_from_points_spherical(shot, point_coords)
    else:
        raise NotImplementedError(
            f"triangle_mesh_from_points not implemented for "
            f"projection type {cam.projection_type}"
        )


def _triangle_mesh_from_points_perspective(
    shot: pymap.Shot,
    point_coords: NDArray,
) -> Tuple[List[List[float]], List[List[int]]]:
    cam = shot.camera

    dx = float(cam.width) / 2 / max(cam.width, cam.height)
    dy = float(cam.height) / 2 / max(cam.width, cam.height)
    corner_pixels = [[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]]
    pixels: List[List[float]] = list(corner_pixels)
    vertices: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(4)]

    for coord in point_coords:
        pixel = shot.project(coord)
        if np.isnan(pixel).any():
            continue
        if -dx <= pixel[0] <= dx and -dy <= pixel[1] <= dy:
            vertices.append(coord.tolist())
            pixels.append(pixel.tolist())

    if len(pixels) < 4:
        return [], []

    try:
        tri = scipy.spatial.Delaunay(pixels)
    except Exception:
        logger.warning(
            "Delaunay triangulation failed for shot %s", shot.id
        )
        return [], []

    sums = [0.0, 0.0, 0.0, 0.0]
    depths = [0.0, 0.0, 0.0, 0.0]
    for t in tri.simplices:
        for i in range(4):
            if i in t:
                for j in t:
                    if j >= 4:
                        depths[i] += shot.pose.transform(vertices[j])[2]
                        sums[i] += 1

    for i in range(4):
        d = depths[i] / sums[i] if sums[i] > 0 else 50.0
        vertices[i] = back_project_no_distortion(
            shot, corner_pixels[i], d
        ).tolist()

    faces = tri.simplices.tolist()
    return vertices, faces


def _triangle_mesh_from_points_fisheye(
    shot: pymap.Shot,
    point_coords: NDArray,
) -> Tuple[List[List[float]], List[List[int]]]:
    bearings = []
    vertices: List[List[float]] = []

    num_circle_points: int = 20
    for i in range(num_circle_points):
        a = 2 * np.pi * float(i) / num_circle_points
        point = 30 * np.array([np.cos(a), np.sin(a), 0])
        bearing = point / np.linalg.norm(point)
        point = shot.pose.transform_inverse(point)
        vertices.append(point.tolist())
        bearings.append(bearing)

    point = 30 * np.array([0, 0, 1])
    bearing = 0.3 * point / np.linalg.norm(point)
    point = shot.pose.transform_inverse(point)
    vertices.append(point.tolist())
    bearings.append(bearing)

    for coord in point_coords:
        direction = shot.pose.transform(coord)
        pixel = direction / np.linalg.norm(direction)
        if not np.isnan(pixel).any():
            vertices.append(coord.tolist())
            bearings.append(pixel.tolist())

    tri = scipy.spatial.ConvexHull(bearings)
    faces = tri.simplices.tolist()

    def good_face(face: List[int]) -> bool:
        return (
            face[0] >= num_circle_points
            or face[1] >= num_circle_points
            or face[2] >= num_circle_points
        )

    faces = list(filter(good_face, faces))
    return vertices, faces


def _triangle_mesh_from_points_spherical(
    shot: pymap.Shot,
    point_coords: NDArray,
) -> Tuple[List[List[float]], List[List[int]]]:
    bearings = []
    vertices: List[List[float]] = []

    for pt in itertools.product([-1, 1], repeat=3):
        bearing = 0.3 * np.array(pt) / np.linalg.norm(pt)
        bearings.append(bearing)
        world_pt = shot.pose.transform_inverse(bearing)
        vertices.append(world_pt.tolist())

    for coord in point_coords:
        direction = shot.pose.transform(coord)
        pixel = direction / np.linalg.norm(direction)
        if not np.isnan(pixel).any():
            vertices.append(coord.tolist())
            bearings.append(pixel.tolist())

    tri = scipy.spatial.ConvexHull(bearings)
    faces = tri.simplices.tolist()
    return vertices, faces


def triangle_mesh(
    shot_id: str, r: types.Reconstruction, tracks_manager: pymap.TracksManager
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Create triangle meshes in a list
    """
    if shot_id not in r.shots or shot_id not in tracks_manager.get_shot_ids():
        return [], []

    shot = r.shots[shot_id]

    if shot.camera.projection_type in [
        "perspective",
        "brown",
        "radial",
        "simple_radial",
    ]:
        return triangle_mesh_perspective(shot_id, r, tracks_manager)
    elif shot.camera.projection_type in [
        "fisheye",
        "fisheye_opencv",
        "fisheye62",
        "fisheye624",
        "dual",
    ]:
        return triangle_mesh_fisheye(shot_id, r, tracks_manager)
    elif pygeometry.Camera.is_panorama(shot.camera.projection_type):
        return triangle_mesh_spherical(shot_id, r, tracks_manager)
    else:
        raise NotImplementedError(
            f"triangle_mesh not implemented for projection type {shot.camera.projection_type}"
        )


def triangle_mesh_perspective(
    shot_id: str, r: types.Reconstruction, tracks_manager: pymap.TracksManager
) -> Tuple[List[List[float]], List[List[int]]]:
    shot = r.shots[shot_id]
    cam = shot.camera

    dx = float(cam.width) / 2 / max(cam.width, cam.height)
    dy = float(cam.height) / 2 / max(cam.width, cam.height)
    pixels = [[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]]
    vertices = [[0.0, 0.0, 0.0] for i in range(4)]
    for track_id in tracks_manager.get_shot_observations(shot_id):
        if track_id in r.points:
            point = r.points[track_id]
            pixel = shot.project(point.coordinates)
            nonans = not np.isnan(pixel).any()
            if nonans and -dx <= pixel[0] <= dx and -dy <= pixel[1] <= dy:
                vertices.append(point.coordinates)
                pixels.append(pixel.tolist())

    try:
        tri = scipy.spatial.Delaunay(pixels)
    except Exception as e:
        logger.error(
            "Delaunay triangulation failed for input: {}".format(repr(pixels)))
        raise e

    sums = [0.0, 0.0, 0.0, 0.0]
    depths = [0.0, 0.0, 0.0, 0.0]
    for t in tri.simplices:
        for i in range(4):
            if i in t:
                for j in t:
                    if j >= 4:
                        depths[i] += shot.pose.transform(vertices[j])[2]
                        sums[i] += 1
    for i in range(4):
        if sums[i] > 0:
            d = depths[i] / sums[i]
        else:
            d = 50.0
        vertices[i] = back_project_no_distortion(shot, pixels[i], d).tolist()

    faces = tri.simplices.tolist()
    return vertices, faces


def back_project_no_distortion(
    shot: pymap.Shot, pixel: List[float], depth: float
) -> NDArray:
    """
    Back-project a pixel of a perspective camera ignoring its radial distortion
    """
    K = shot.camera.get_K()
    K1 = np.linalg.inv(K)
    p = np.dot(K1, [pixel[0], pixel[1], 1])
    p *= depth / p[2]
    return shot.pose.transform_inverse(p)


def triangle_mesh_fisheye(
    shot_id: str, r: types.Reconstruction, tracks_manager: pymap.TracksManager
) -> Tuple[List[List[float]], List[List[int]]]:
    shot = r.shots[shot_id]

    bearings = []
    vertices = []

    # Add boundary vertices
    num_circle_points: int = 20
    for i in range(num_circle_points):
        a = 2 * np.pi * float(i) / num_circle_points
        point = 30 * np.array([np.cos(a), np.sin(a), 0])
        bearing = point / np.linalg.norm(point)
        point = shot.pose.transform_inverse(point)
        vertices.append(point.tolist())
        bearings.append(bearing)

    # Add a single vertex in front of the camera
    point = 30 * np.array([0, 0, 1])
    bearing = 0.3 * point / np.linalg.norm(point)
    point = shot.pose.transform_inverse(point)
    vertices.append(point.tolist())
    bearings.append(bearing)

    # Add reconstructed points
    for track_id in tracks_manager.get_shot_observations(shot_id):
        if track_id in r.points:
            point = r.points[track_id].coordinates
            direction = shot.pose.transform(point)
            pixel = direction / np.linalg.norm(direction)
            if not np.isnan(pixel).any():
                vertices.append(point)
                bearings.append(pixel.tolist())

    # Triangulate
    tri = scipy.spatial.ConvexHull(bearings)
    faces = tri.simplices.tolist()

    # Remove faces having only boundary vertices
    def good_face(face: List[int]) -> bool:
        return (
            face[0] >= num_circle_points
            or face[1] >= num_circle_points
            or face[2] >= num_circle_points
        )

    faces = list(filter(good_face, faces))

    return vertices, faces


def triangle_mesh_spherical(
    shot_id: str, r: types.Reconstruction, tracks_manager: pymap.TracksManager
) -> Tuple[List[List[float]], List[List[int]]]:
    shot = r.shots[shot_id]

    bearings = []
    vertices = []

    # Add vertices to ensure that the camera is inside the convex hull
    # of the points
    for point in itertools.product([-1, 1], repeat=3):  # vertices of a cube
        bearing = 0.3 * np.array(point) / np.linalg.norm(point)
        bearings.append(bearing)
        point = shot.pose.transform_inverse(bearing)
        vertices.append(point.tolist())

    for track_id in tracks_manager.get_shot_observations(shot_id):
        if track_id in r.points:
            point = r.points[track_id].coordinates
            direction = shot.pose.transform(point)
            pixel = direction / np.linalg.norm(direction)
            if not np.isnan(pixel).any():
                vertices.append(point)
                bearings.append(pixel.tolist())

    tri = scipy.spatial.ConvexHull(bearings)
    faces = tri.simplices.tolist()

    return vertices, faces
