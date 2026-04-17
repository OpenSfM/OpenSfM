# pyre-strict

import math
import numpy as np
import pytest

from opensfm import align
from opensfm import config
from opensfm import geo
from opensfm import geometry
from opensfm import multiview
from opensfm import pygeometry
from opensfm import pymap
from opensfm import transformations as tf
from opensfm import types


# ============================================================================
# Helper functions to create test data
# ============================================================================


def create_simple_reconstruction(
    num_shots: int = 3,
    line: bool = False,
    add_gps: bool = True,
    gps_noise: float = 1.0,
    add_opk: bool = True,
) -> types.Reconstruction:
    """Create a simple test reconstruction with optional GPS and OPK data."""
    reconstruction = types.Reconstruction()
    reference = geo.TopocentricConverter(47.0, 6.0, 0)
    reconstruction.reference = reference

    camera = pygeometry.Camera.create_perspective(0.5, 0.0, 0.0)
    camera.id = "cam1"
    camera.focal = 0.5
    reconstruction.add_camera(camera)

    for i in range(num_shots):
        pose = pygeometry.Pose()
        shot = pymap.Shot(f"shot{i}", camera, pose)

        # Place shots in a line or grid pattern
        if line:
            origin = np.array([float(i * 10.0), 0.0, 0.0])
        else:
            origin = np.array([float(i * 10.0), float((i % 2) * 10.0), 0.0])

        pose.set_origin(origin)
        pose.set_rotation_matrix(np.eye(3))
        shot.pose = pose

        # Add GPS metadata
        if add_gps:
            gps_value = origin + np.array([gps_noise, gps_noise, gps_noise])
            shot.metadata.gps_position.value = gps_value.tolist()

        # Add OPK metadata
        if add_opk:
            shot.metadata.opk_angles.value = [0.0, 0.0, 0.0]

        reconstruction.add_shot(shot)

    # Add some 3D points
    for i in range(num_shots * 2):
        reconstruction.create_point(f"point{i}", np.array(
            [float(i * 5), float(i * 3), float(i * 2)]))

    return reconstruction


def create_gcps_with_observations(
    reconstruction: types.Reconstruction,
    num_gcps: int = 3,
    line: bool = False,
) -> list[pymap.GroundControlPoint]:
    """Create GCPs with observations linked to shots."""
    gcps = []
    reference = reconstruction.reference

    for i in range(num_gcps):
        gcp = pymap.GroundControlPoint()
        gcp.id = f"gcp{i}"
        gcp.survey_point_id = i

        # Create GCP in ENU coordinates, convert to LLA
        if line:
            enu = np.array([float(i * 10.0), 0.0, 0.0])
        else:
            enu = np.array([float(i * 10.0), float((i % 2) * 10.0), 0.0])

        lat, lon, alt = reference.to_lla(*enu)
        gcp.lla = {"latitude": lat, "longitude": lon, "altitude": alt}
        gcp.has_altitude = True

        # Add observations from first 2 shots
        for shot_idx in range(min(2, len(reconstruction.shots))):
            shot = list(reconstruction.shots.values())[shot_idx]
            obs = pymap.GroundControlPointObservation()
            obs.shot_id = shot.id
            # Simple projection: just use coordinates as image position
            obs.projection = [100.0 + i * 10, 100.0 + i * 10]
            gcp.add_observation(obs)

        gcps.append(gcp)

    return gcps


def create_gcps_no_observations(
    num_gcps: int = 3, line: bool = False
) -> list[pymap.GroundControlPoint]:
    """Create GCPs without observations (for non-triangulation tests)."""
    reference = geo.TopocentricConverter(47.0, 6.0, 0)
    gcps = []

    for i in range(num_gcps):
        gcp = pymap.GroundControlPoint()
        gcp.id = f"gcp{i}"
        gcp.survey_point_id = i

        if line:
            enu = np.array([float(i * 10.0), 0.0, 0.0])
        else:
            enu = np.array([float(i * 10.0), float((i % 2) * 10.0), 0.0])

        lat, lon, alt = reference.to_lla(*enu)
        gcp.lla = {"latitude": lat, "longitude": lon, "altitude": alt}
        gcp.has_altitude = True
        gcps.append(gcp)

    return gcps


# ============================================================================
# Tests for apply_similarity_pose
# ============================================================================


class TestApplySimilarityPose:
    """Tests for apply_similarity_pose function.

    The similarity y = s * A @ x + b transforms origin as:
        new_origin = s * A @ old_origin + b
    and rotation as:
        new_R = old_R @ A.T
    """

    def test_apply_similarity_pose_identity(self) -> None:
        """Identity similarity leaves origin and rotation unchanged."""
        pose = pygeometry.Pose()
        pose.set_origin([1.0, 2.0, 3.0])
        pose.set_rotation_matrix(np.eye(3))

        align.apply_similarity_pose(
            pose, 1.0, np.eye(3), np.array([0.0, 0.0, 0.0]))

        np.testing.assert_array_almost_equal(
            pose.get_origin(), [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(
            pose.get_rotation_matrix(), np.eye(3))

    def test_apply_similarity_pose_translation(self) -> None:
        """Translation-only similarity shifts origin by b."""
        pose = pygeometry.Pose()
        pose.set_origin([1.0, 2.0, 3.0])
        pose.set_rotation_matrix(np.eye(3))

        b = np.array([10.0, 20.0, 30.0])
        align.apply_similarity_pose(pose, 1.0, np.eye(3), b)

        np.testing.assert_array_almost_equal(
            pose.get_origin(), [11.0, 22.0, 33.0])
        # Rotation unchanged: R @ I.T = R
        np.testing.assert_array_almost_equal(
            pose.get_rotation_matrix(), np.eye(3))

    def test_apply_similarity_pose_scale(self) -> None:
        """Scale-only similarity multiplies origin by s."""
        pose = pygeometry.Pose()
        pose.set_origin([1.0, 2.0, 3.0])
        pose.set_rotation_matrix(np.eye(3))

        align.apply_similarity_pose(
            pose, 2.0, np.eye(3), np.array([0.0, 0.0, 0.0]))

        np.testing.assert_array_almost_equal(
            pose.get_origin(), [2.0, 4.0, 6.0])

    def test_apply_similarity_pose_rotation(self) -> None:
        """90-degree Z rotation maps (1,0,0) to (0,1,0)."""
        pose = pygeometry.Pose()
        pose.set_origin([1.0, 0.0, 0.0])
        pose.set_rotation_matrix(np.eye(3))

        angle = np.pi / 2
        Rz = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])

        align.apply_similarity_pose(pose, 1.0, Rz, np.array([0.0, 0.0, 0.0]))

        np.testing.assert_array_almost_equal(
            pose.get_origin(), [0.0, 1.0, 0.0], decimal=10)
        # new_R = I @ Rz.T = Rz.T
        np.testing.assert_array_almost_equal(
            pose.get_rotation_matrix(), Rz.T, decimal=10)

    def test_apply_similarity_pose_combined(self) -> None:
        """Combined scale + rotation + translation: new_origin = s * A @ o + b."""
        pose = pygeometry.Pose()
        origin = np.array([3.0, 4.0, 0.0])
        R0 = np.eye(3)
        pose.set_origin(origin)
        pose.set_rotation_matrix(R0)

        s = 2.0
        angle = np.pi / 2
        A = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        b = np.array([1.0, 1.0, 1.0])

        align.apply_similarity_pose(pose, s, A, b)

        expected_origin = s * A @ origin + b  # 2*[-4,3,0]+[1,1,1] = [-7,7,1]
        np.testing.assert_array_almost_equal(
            pose.get_origin(), expected_origin, decimal=10)
        np.testing.assert_array_almost_equal(
            pose.get_rotation_matrix(), R0 @ A.T, decimal=10)


# ============================================================================
# Tests for apply_similarity
# ============================================================================


class TestApplySimilarity:
    """Tests for apply_similarity function."""

    def test_apply_similarity_scales_points(self) -> None:
        """Scale=2 doubles all point coordinates."""
        reconstruction = create_simple_reconstruction(
            num_shots=2, add_gps=False)

        orig_coords = {
            pid: p.coordinates.copy()
            for pid, p in reconstruction.points.items()
        }

        align.apply_similarity(reconstruction, 2.0,
                               np.eye(3), np.array([0.0, 0.0, 0.0]))

        for pid, point in reconstruction.points.items():
            np.testing.assert_array_almost_equal(
                point.coordinates, 2.0 * orig_coords[pid])

    def test_apply_similarity_translates_shots_and_points(self) -> None:
        """Translation-only shifts all origins and points by b."""
        reconstruction = create_simple_reconstruction(
            num_shots=2, add_gps=False)
        orig_origins = {
            sid: np.array(s.pose.get_origin()).copy()
            for sid, s in reconstruction.shots.items()
        }
        orig_coords = {
            pid: p.coordinates.copy()
            for pid, p in reconstruction.points.items()
        }

        b = np.array([10.0, 20.0, 30.0])
        align.apply_similarity(reconstruction, 1.0, np.eye(3), b)

        for sid, shot in reconstruction.shots.items():
            np.testing.assert_array_almost_equal(
                shot.pose.get_origin(), orig_origins[sid] + b)
        for pid, point in reconstruction.points.items():
            np.testing.assert_array_almost_equal(
                point.coordinates, orig_coords[pid] + b)

    def test_apply_similarity_combined(self) -> None:
        """Combined s=3, 90-deg Z rotation, translation [1,2,3]."""
        reconstruction = create_simple_reconstruction(
            num_shots=2, add_gps=False)

        s = 3.0
        angle = np.pi / 2
        A = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        b = np.array([1.0, 2.0, 3.0])

        orig_coords = {
            pid: p.coordinates.copy()
            for pid, p in reconstruction.points.items()
        }
        orig_origins = {
            sid: np.array(sh.pose.get_origin()).copy()
            for sid, sh in reconstruction.shots.items()
        }

        align.apply_similarity(reconstruction, s, A, b)

        for pid, point in reconstruction.points.items():
            expected = s * A @ orig_coords[pid] + b
            np.testing.assert_array_almost_equal(
                point.coordinates, expected, decimal=10)
        for sid, shot in reconstruction.shots.items():
            expected = s * A @ orig_origins[sid] + b
            np.testing.assert_array_almost_equal(
                shot.pose.get_origin(), expected, decimal=5)


# ============================================================================
# Tests for alignment_constraints
# ============================================================================


class TestAlignmentConstraints:
    """Tests for alignment_constraints function."""

    def test_alignment_constraints_gps_only(self) -> None:
        """GPS constraints: X = rig origins, Xp = GPS positions (origin + noise)."""
        noise = 5.0
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=True, gps_noise=noise)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        X, Xp = align.alignment_constraints(conf, reconstruction, [], True)

        assert len(X) == len(reconstruction.rig_instances)
        # X should be the rig instance origins, Xp should be the GPS positions
        for i, rig_instance in enumerate(reconstruction.rig_instances.values()):
            np.testing.assert_array_almost_equal(
                X[i], rig_instance.pose.get_origin())
            expected_gps = np.array(rig_instance.pose.get_origin()) + noise
            np.testing.assert_array_almost_equal(Xp[i], expected_gps)

    def test_alignment_constraints_gps_ignores_shots_without_gps(self) -> None:
        """Shots without GPS metadata are excluded from constraints."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        X, Xp = align.alignment_constraints(conf, reconstruction, [], True)

        assert len(X) == 0
        assert len(Xp) == 0

    def test_alignment_constraints_no_data(self) -> None:
        """No GPS and no GCP yields empty constraints."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False)
        conf = config.default_config()
        conf["bundle_use_gps"] = False
        conf["bundle_use_gcp"] = False

        X, Xp = align.alignment_constraints(conf, reconstruction, [], False)

        assert len(X) == 0
        assert len(Xp) == 0


# ============================================================================
# Tests for detect_alignment_constraints
# ============================================================================


class TestDetectAlignmentConstraints:
    """Tests for detect_alignment_constraints function."""

    def test_detect_alignment_constraints_well_conditioned_is_naive(self) -> None:
        """2D spread of GPS → naive method."""
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        method = align.detect_alignment_constraints(
            conf, reconstruction, [], True)
        assert method == "naive"

    def test_detect_alignment_constraints_collinear_is_orientation_prior(self) -> None:
        """Collinear GPS → orientation_prior method."""
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=True, add_gps=True)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        method = align.detect_alignment_constraints(
            conf, reconstruction, [], True)
        assert method == "orientation_prior"

    def test_detect_alignment_constraints_fewer_than_3_is_orientation_prior(self) -> None:
        """Fewer than 3 constraints → orientation_prior."""
        reconstruction = create_simple_reconstruction(
            num_shots=2, add_gps=True)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        method = align.detect_alignment_constraints(
            conf, reconstruction, [], True)
        assert method == "orientation_prior"

    def test_detect_alignment_constraints_single_shot(self) -> None:
        """Single shot → orientation_prior (< 3 constraints)."""
        reconstruction = create_simple_reconstruction(
            num_shots=1, add_gps=True)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        method = align.detect_alignment_constraints(
            conf, reconstruction, [], True)
        assert method == "orientation_prior"


# ============================================================================
# Tests for detect_orientation_prior
# ============================================================================


class TestDetectOrientationPrior:
    """Tests for detect_orientation_prior function."""

    def test_detect_orientation_prior_with_opk_returns_horizontal_plane(self) -> None:
        """With OPK=[0,0,0] (identity rotation), the plane normal should be vertical [0,0,±1]."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, line=True, add_opk=True
        )
        conf = config.default_config()

        plane = align.detect_orientation_prior(reconstruction, conf)

        assert plane is not None
        assert len(plane) == 4
        # Normal is plane[:3]. With identity OPK, the vertical is [0,0,1].
        normal = plane[:3] / np.linalg.norm(plane[:3])
        assert abs(abs(normal[2]) - 1.0) < 0.1  # nearly vertical

    def test_detect_orientation_prior_no_opk_falls_back_to_estimate(self) -> None:
        """Without OPK, falls back to estimate_ground_plane."""
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_opk=False, add_gps=True
        )
        conf = config.default_config()

        plane = align.detect_orientation_prior(reconstruction, conf)

        # With 4 shots on z=0 plane, should detect a horizontal plane
        assert plane is not None
        normal = plane[:3] / np.linalg.norm(plane[:3])
        assert abs(abs(normal[2]) - 1.0) < 0.1


# ============================================================================
# Tests for estimate_ground_plane
# ============================================================================


class TestEstimateGroundPlane:
    """Tests for estimate_ground_plane function."""

    def test_estimate_ground_plane_coplanar_shots(self) -> None:
        """Cameras at z=0 → plane normal is [0,0,±1]."""
        reconstruction = create_simple_reconstruction(num_shots=4, line=False)
        conf = config.default_config()

        plane = align.estimate_ground_plane(reconstruction, conf)

        assert plane is not None
        assert len(plane) == 4
        normal = plane[:3] / np.linalg.norm(plane[:3])
        # All shots are at z=0, plane should be horizontal
        assert abs(abs(normal[2]) - 1.0) < 0.1


# ============================================================================
# Tests for get_horizontal_and_vertical_directions
# ============================================================================


class TestGetHorizontalAndVerticalDirections:
    """Tests for get_horizontal_and_vertical_directions function.

    Given camera rotation R and EXIF orientation tag, returns (x_right, y_down, z_front).
    See http://sylvana.net/jpegcrop/exif_orientation.html
    """

    def test_orientation_1_identity(self) -> None:
        """Orientation 1 (normal): x=R[0], y=R[1], z=R[2]."""
        R = np.eye(3)
        x, y, z = align.get_horizontal_and_vertical_directions(R, 1)
        np.testing.assert_array_equal(x, [1, 0, 0])
        np.testing.assert_array_equal(y, [0, 1, 0])
        np.testing.assert_array_equal(z, [0, 0, 1])

    def test_orientation_2_mirrored(self) -> None:
        """Orientation 2: x=-R[0], y=R[1], z=-R[2]."""
        R = np.eye(3)
        x, y, z = align.get_horizontal_and_vertical_directions(R, 2)
        np.testing.assert_array_equal(x, [-1, 0, 0])
        np.testing.assert_array_equal(y, [0, 1, 0])
        np.testing.assert_array_equal(z, [0, 0, -1])

    def test_orientation_3_upside_down(self) -> None:
        """Orientation 3: x=-R[0], y=-R[1], z=R[2]."""
        R = np.eye(3)
        x, y, z = align.get_horizontal_and_vertical_directions(R, 3)
        np.testing.assert_array_equal(x, [-1, 0, 0])
        np.testing.assert_array_equal(y, [0, -1, 0])
        np.testing.assert_array_equal(z, [0, 0, 1])

    def test_orientation_6_rotated_90_cw(self) -> None:
        """Orientation 6: x=-R[1], y=R[0], z=R[2]."""
        R = np.eye(3)
        x, y, z = align.get_horizontal_and_vertical_directions(R, 6)
        np.testing.assert_array_equal(x, [0, -1, 0])
        np.testing.assert_array_equal(y, [1, 0, 0])
        np.testing.assert_array_equal(z, [0, 0, 1])

    def test_all_orientations_with_non_identity_rotation(self) -> None:
        """All 8 orientations produce consistent results with a non-trivial R."""
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        expected = {
            1: (R[0, :], R[1, :], R[2, :]),
            2: (-R[0, :], R[1, :], -R[2, :]),
            3: (-R[0, :], -R[1, :], R[2, :]),
            4: (R[0, :], -R[1, :], R[2, :]),
            5: (R[1, :], R[0, :], -R[2, :]),
            6: (-R[1, :], R[0, :], R[2, :]),
            7: (-R[1, :], -R[0, :], -R[2, :]),
            8: (R[1, :], -R[0, :], R[2, :]),
        }
        for orientation in range(1, 9):
            x, y, z = align.get_horizontal_and_vertical_directions(
                R, orientation)
            ex, ey, ez = expected[orientation]
            np.testing.assert_array_almost_equal(x, ex)
            np.testing.assert_array_almost_equal(y, ey)
            np.testing.assert_array_almost_equal(z, ez)


# ============================================================================
# Tests for compute_naive_similarity
# ============================================================================


class TestComputeNaiveSimilarity:
    """Tests for compute_naive_similarity function."""

    def test_compute_naive_similarity_recovers_translation(self) -> None:
        """With identity-placed cameras and GPS = origin + noise, should recover b ≈ [noise,noise,noise]."""
        noise = 3.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        result = align.compute_naive_similarity(
            conf, reconstruction, [], use_gps=True, use_scale=True
        )

        assert result is not None
        s, A, b = result
        # Scale should be ~1 (GPS is just a translation of the true positions)
        np.testing.assert_almost_equal(s, 1.0, decimal=3)
        # Rotation should be ~identity
        np.testing.assert_array_almost_equal(A, np.eye(3), decimal=3)
        # Translation should be ~[noise, noise, noise]
        np.testing.assert_array_almost_equal(
            b, [noise, noise, noise], decimal=3)

    def test_compute_naive_similarity_single_constraint(self) -> None:
        """Single constraint: s=1, A=I, b = Xp[0] - X[0]."""
        noise = 7.0
        reconstruction = create_simple_reconstruction(
            num_shots=1, line=False, add_gps=True, gps_noise=noise
        )
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        result = align.compute_naive_similarity(
            conf, reconstruction, [], use_gps=True, use_scale=True
        )

        assert result is not None
        s, A, b = result
        assert s == 1.0
        np.testing.assert_array_almost_equal(A, np.eye(3))
        np.testing.assert_array_almost_equal(b, [noise, noise, noise])

    def test_compute_naive_similarity_no_constraints_returns_none(self) -> None:
        """No GPS → returns None."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False, add_opk=False
        )
        conf = config.default_config()
        conf["bundle_use_gps"] = False
        conf["bundle_use_gcp"] = False

        result = align.compute_naive_similarity(
            conf, reconstruction, [], use_gps=True, use_scale=True
        )

        assert result is None

    def test_compute_naive_similarity_no_scale(self) -> None:
        """With use_scale=False, s should be ~1.0."""
        noise = 3.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        result = align.compute_naive_similarity(
            conf, reconstruction, [], use_gps=True, use_scale=False
        )

        assert result is not None
        s, A, b = result
        np.testing.assert_almost_equal(s, 1.0, decimal=3)


# ============================================================================
# Tests for compute_orientation_prior_similarity
# ============================================================================


class TestComputeOrientationPriorSimilarity:
    """Tests for compute_orientation_prior_similarity function."""

    def test_recovers_translation_on_collinear_shots(self) -> None:
        """Collinear shots with GPS noise → recovers translation along the line and vertically."""
        noise = 3.0
        reconstruction = create_simple_reconstruction(
            num_shots=3, line=True, add_gps=True, gps_noise=noise, add_opk=True
        )
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        plane = align.detect_orientation_prior(reconstruction, conf)
        assert plane is not None

        result = align.compute_orientation_prior_similarity(
            reconstruction, conf, [], use_gps=True, use_scale=True, plane=plane,
        )

        assert result is not None
        s, A, b = result
        # Scale should be ~1 since GPS is just translated
        np.testing.assert_almost_equal(s, 1.0, decimal=1)
        # After applying similarity, origins should land near GPS positions
        for rig in reconstruction.rig_instances.values():
            origin = np.array(rig.pose.get_origin())
            transformed = s * A @ origin + b
            shot = list(rig.shots.values())[0]
            gps = np.array(shot.metadata.gps_position.value)
            np.testing.assert_allclose(transformed, gps, atol=1.0)

    def test_returns_none_for_no_plane(self) -> None:
        """None plane → returns None."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=True)
        conf = config.default_config()

        result = align.compute_orientation_prior_similarity(
            reconstruction, conf, [], use_gps=True, use_scale=True, plane=None
        )

        assert result is None


# ============================================================================
# Tests for compute_reconstruction_similarity
# ============================================================================


class TestComputeReconstructionSimilarity:
    """Tests for compute_reconstruction_similarity function."""

    def test_naive_recovers_gps_translation(self) -> None:
        """Naive method with pure-translation GPS should recover b ≈ [noise,noise,noise]."""
        noise = 5.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "naive"

        result = align.compute_reconstruction_similarity(
            reconstruction, [], conf, use_gps=True, use_scale=True
        )

        assert result is not None
        s, A, b = result
        np.testing.assert_almost_equal(s, 1.0, decimal=3)
        np.testing.assert_array_almost_equal(A, np.eye(3), decimal=3)
        np.testing.assert_array_almost_equal(
            b, [noise, noise, noise], decimal=3)

    def test_orientation_prior_method_on_collinear(self) -> None:
        """Orientation prior on collinear shots should produce valid alignment."""
        noise = 3.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=True, add_gps=True, gps_noise=noise, add_opk=True
        )
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "orientation_prior"

        result = align.compute_reconstruction_similarity(
            reconstruction, [], conf, use_gps=True, use_scale=True
        )

        assert result is not None
        s, A, b = result
        assert s > 0
        # After alignment, origins should be near GPS positions
        for rig in reconstruction.rig_instances.values():
            origin = np.array(rig.pose.get_origin())
            transformed = s * A @ origin + b
            shot = list(rig.shots.values())[0]
            gps = np.array(shot.metadata.gps_position.value)
            np.testing.assert_allclose(transformed, gps, atol=1.0)

    def test_auto_selects_naive_for_well_conditioned(self) -> None:
        """Auto method picks naive for well-conditioned GPS and recovers correct b."""
        noise = 2.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "auto"

        result = align.compute_reconstruction_similarity(
            reconstruction, [], conf, use_gps=True, use_scale=True
        )

        assert result is not None
        s, A, b = result
        np.testing.assert_array_almost_equal(
            b, [noise, noise, noise], decimal=2)

    def test_degenerate_returns_none(self) -> None:
        """No GPS data → returns None."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False, add_opk=False)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "naive"

        result = align.compute_reconstruction_similarity(
            reconstruction, [], conf, use_gps=True, use_scale=True
        )
        assert result is None


# ============================================================================
# Tests for align_reconstruction
# ============================================================================


class TestAlignReconstruction:
    """Tests for align_reconstruction function."""

    def test_align_reconstruction_shifts_to_gps(self) -> None:
        """After alignment with GPS, shot origins should match GPS positions."""
        noise = 4.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)

        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "naive"

        result = align.align_reconstruction(
            reconstruction, [], conf, use_gps=True, bias_override=False
        )

        assert result is not None
        # After alignment, origins should be near GPS values
        for shot in reconstruction.shots.values():
            gps = np.array(shot.metadata.gps_position.value)
            np.testing.assert_allclose(
                shot.pose.get_origin(), gps, atol=0.1)

    def test_align_reconstruction_no_data_is_identity(self) -> None:
        """No GPS/GCP: returns s=1, A=I, b=0 and leaves reconstruction unchanged."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False)
        orig_origins = {
            sid: np.array(s.pose.get_origin()).copy()
            for sid, s in reconstruction.shots.items()
        }

        conf = config.default_config()
        conf["bundle_use_gps"] = False
        conf["bundle_use_gcp"] = False

        result = align.align_reconstruction(
            reconstruction, [], conf, use_gps=False, bias_override=False
        )

        assert result is not None
        s, A, b = result
        assert s == 1.0
        np.testing.assert_array_almost_equal(A, np.eye(3))
        np.testing.assert_array_almost_equal(b, np.zeros(3))
        # Origins should be unchanged
        for sid, shot in reconstruction.shots.items():
            np.testing.assert_array_almost_equal(
                shot.pose.get_origin(), orig_origins[sid])

    def test_align_reconstruction_preserves_rig_cameras(self) -> None:
        """Alignment preserves rig camera count."""
        reconstruction = create_simple_reconstruction(
            num_shots=2, add_gps=True)
        original_rig_cameras = len(reconstruction.rig_cameras)

        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "naive"

        align.align_reconstruction(
            reconstruction, [], conf, use_gps=True, bias_override=False
        )

        assert len(reconstruction.rig_cameras) == original_rig_cameras


# ============================================================================
# Tests for set_gps_bias
# ============================================================================


class TestSetGpsBias:
    """Tests for set_gps_bias function."""

    def test_set_gps_bias_no_gcp_returns_identity(self) -> None:
        """Without GCP observations, set_gps_bias returns identity (no alignment data)."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=True)
        conf = config.default_config()
        conf["bundle_use_gcp"] = True
        conf["bundle_use_gps"] = False

        result = align.set_gps_bias(reconstruction, conf, [], use_scale=True)

        # No GCP observations → no triangulation → identity similarity
        assert result is not None
        s, A, b = result
        np.testing.assert_almost_equal(s, 1.0)
        np.testing.assert_array_almost_equal(A, np.eye(3))
        np.testing.assert_array_almost_equal(b, np.zeros(3))

    def test_set_gps_bias_sets_camera_bias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When similarity computation succeeds, camera biases are set."""
        noise = 3.0
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=True, gps_noise=noise)
        conf = config.default_config()

        dummy_similarity = (2.0, np.eye(3), np.array([1.0, 2.0, 3.0]))
        monkeypatch.setattr(align, "compute_reconstruction_similarity",
                            lambda *args, **kwargs: dummy_similarity)

        result = align.set_gps_bias(reconstruction, conf, [], use_scale=True)

        assert result is not None
        s_ret, A_ret, b_ret = result
        assert s_ret == 2.0
        np.testing.assert_array_almost_equal(b_ret, [1.0, 2.0, 3.0])

        biases = reconstruction.get_biases()
        assert "cam1" in biases
        assert biases["cam1"] is not None


# ============================================================================
# Tests for triangulate_all_gcp
# ============================================================================


class TestTriangulateAllGcp:
    """Tests for triangulate_all_gcp function."""

    def test_no_observations_returns_empty(self) -> None:
        """GCPs without observations yield empty lists."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False)
        gcps = create_gcps_no_observations(num_gcps=2)

        triangulated, measured = align.triangulate_all_gcp(
            reconstruction, gcps, threshold=1.0
        )

        assert len(triangulated) == 0
        assert len(measured) == 0

    def test_triangulated_and_measured_same_length(self) -> None:
        """triangulated and measured always have the same length."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False)
        gcps = create_gcps_with_observations(reconstruction, num_gcps=3)

        triangulated, measured = align.triangulate_all_gcp(
            reconstruction, gcps, threshold=100.0
        )

        assert len(triangulated) == len(measured)

    def test_measured_coordinates_match_gcp_lla(self) -> None:
        """measured entries correspond to GCP LLA converted to ENU."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False)
        gcps = create_gcps_with_observations(reconstruction, num_gcps=3)

        triangulated, measured = align.triangulate_all_gcp(
            reconstruction, gcps, threshold=100.0
        )

        # For each successfully triangulated GCP, the measured point
        # should match the ENU conversion of the GCP LLA
        reference = reconstruction.reference
        for m, gcp in zip(measured, [g for g in gcps]):
            enu = np.array(reference.to_topocentric(*gcp.lla_vec))
            np.testing.assert_array_almost_equal(m, enu, decimal=3)


# ============================================================================
# Integration Tests
# ============================================================================


class TestAlignmentIntegration:
    """Integration tests combining multiple functions."""

    def test_gps_only_pipeline_aligns_to_gps(self) -> None:
        """Full GPS-only pipeline: detect → compute → apply → origins match GPS."""
        noise = 3.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise
        )
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "auto"

        method = align.detect_alignment_constraints(
            conf, reconstruction, [], True)
        assert method == "naive"

        result = align.compute_reconstruction_similarity(
            reconstruction, [], conf, use_gps=True, use_scale=True
        )
        assert result is not None
        align.apply_similarity(reconstruction, *result)

        for shot in reconstruction.shots.values():
            gps = np.array(shot.metadata.gps_position.value)
            np.testing.assert_allclose(
                shot.pose.get_origin(), gps, atol=0.1)

    def test_gcp_only_no_observations_returns_identity(self) -> None:
        """GCP without observations → no triangulation → identity similarity."""
        reconstruction = create_simple_reconstruction(
            num_shots=3, add_gps=False)
        gcps = create_gcps_no_observations(num_gcps=2)
        conf = config.default_config()
        conf["bundle_use_gps"] = False
        conf["bundle_use_gcp"] = True
        conf["align_method"] = "auto"

        result = align.compute_reconstruction_similarity(
            reconstruction, gcps, conf, use_gps=False, use_scale=True
        )
        # No triangulable GCPs → empty constraints → identity
        assert result is not None
        s, A, b = result
        np.testing.assert_almost_equal(s, 1.0)
        np.testing.assert_array_almost_equal(A, np.eye(3))
        np.testing.assert_array_almost_equal(b, np.zeros(3))

    def test_line_is_orientation_prior_2d_is_naive(self) -> None:
        """Line → orientation_prior; 2D spread → naive."""
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False

        rec_2d = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True)
        assert align.detect_alignment_constraints(
            conf, rec_2d, [], True) == "naive"

        rec_line = create_simple_reconstruction(
            num_shots=4, line=True, add_gps=True)
        assert align.detect_alignment_constraints(
            conf, rec_line, [], True) == "orientation_prior"

    def test_scale_disabled_keeps_scale_one(self) -> None:
        """use_scale=False forces s ≈ 1.0."""
        noise = 3.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = False
        conf["align_method"] = "naive"

        result = align.compute_reconstruction_similarity(
            reconstruction, [], conf, use_gps=True, use_scale=False
        )
        assert result is not None
        s, A, b = result
        np.testing.assert_almost_equal(s, 1.0, decimal=3)

    def test_gcp_only_well_conditioned(self) -> None:
        """GCP-only alignment: origins stay at original positions (GCPs have no noise)."""
        reconstruction = create_simple_reconstruction(
            num_shots=4, add_gps=False)
        gcps = create_gcps_with_observations(
            reconstruction, num_gcps=4, line=False)
        conf = config.default_config()
        conf["bundle_use_gps"] = False
        conf["bundle_use_gcp"] = True
        conf["align_method"] = "auto"

        orig_origins = [np.array(shot.pose.get_origin())
                        for shot in reconstruction.shots.values()]

        result = align.compute_reconstruction_similarity(
            reconstruction, gcps, conf, use_gps=False, use_scale=True
        )
        assert result is not None
        align.apply_similarity(reconstruction, *result)

        for orig, shot in zip(orig_origins, reconstruction.shots.values()):
            np.testing.assert_array_almost_equal(
                shot.pose.get_origin(), orig, decimal=5)

    def test_gps_and_gcp_well_conditioned(self) -> None:
        """GPS+GCP: origins close to original (GCPs anchor, GPS adds slight bias)."""
        noise = 2.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)
        gcps = create_gcps_with_observations(
            reconstruction, num_gcps=4, line=False)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = True
        conf["align_method"] = "auto"

        orig_origins = [np.array(shot.pose.get_origin())
                        for shot in reconstruction.shots.values()]

        result = align.compute_reconstruction_similarity(
            reconstruction, gcps, conf, use_gps=True, use_scale=True
        )
        assert result is not None
        align.apply_similarity(reconstruction, *result)

        for orig, shot in zip(orig_origins, reconstruction.shots.values()):
            np.testing.assert_array_almost_equal(
                shot.pose.get_origin(), orig, decimal=1)

    def test_bad_gps_good_gcp_alignment(self) -> None:
        """Collinear GPS + well-conditioned GCP: GCP anchors the alignment."""
        noise = 2.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=True, add_gps=True, add_opk=False, gps_noise=noise)
        gcps = create_gcps_with_observations(
            reconstruction, num_gcps=4, line=False)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = True
        conf["align_method"] = "auto"

        orig_origins = [np.array(shot.pose.get_origin())
                        for shot in reconstruction.shots.values()]

        result = align.compute_reconstruction_similarity(
            reconstruction, gcps, conf, use_gps=True, use_scale=True
        )
        assert result is not None
        align.apply_similarity(reconstruction, *result)

        for orig, shot in zip(orig_origins, reconstruction.shots.values()):
            np.testing.assert_array_almost_equal(
                shot.pose.get_origin(), orig, decimal=1)

    def test_good_gps_bad_gcp_alignment(self) -> None:
        """Well-conditioned GPS + collinear GCP: origins still close to original."""
        noise = 2.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=False, add_gps=True, gps_noise=noise)
        gcps = create_gcps_with_observations(
            reconstruction, num_gcps=4, line=True)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = True
        conf["align_method"] = "auto"

        orig_origins = [np.array(shot.pose.get_origin())
                        for shot in reconstruction.shots.values()]

        result = align.compute_reconstruction_similarity(
            reconstruction, gcps, conf, use_gps=True, use_scale=True
        )
        assert result is not None
        align.apply_similarity(reconstruction, *result)

        for orig, shot in zip(orig_origins, reconstruction.shots.values()):
            np.testing.assert_array_almost_equal(
                shot.pose.get_origin(), orig, decimal=1)

    def test_bad_gps_bad_gcp_alignment_fallback(self) -> None:
        """Both collinear + OPK fallback: origins shift towards GPS positions."""
        noise = 2.0
        reconstruction = create_simple_reconstruction(
            num_shots=4, line=True, add_gps=True, add_opk=True, gps_noise=noise)
        gcps = create_gcps_with_observations(
            reconstruction, num_gcps=4, line=True)
        conf = config.default_config()
        conf["bundle_use_gps"] = True
        conf["bundle_use_gcp"] = True
        conf["align_method"] = "auto"

        orig_origins = [np.array(shot.pose.get_origin())
                        for shot in reconstruction.shots.values()]

        result = align.compute_reconstruction_similarity(
            reconstruction, gcps, conf, use_gps=True, use_scale=True
        )
        assert result is not None
        align.apply_similarity(reconstruction, *result)

        for orig, shot in zip(orig_origins, reconstruction.shots.values()):
            expected_shifted = orig + np.array([noise, noise, noise])
            np.testing.assert_allclose(
                shot.pose.get_origin(), expected_shifted, atol=3.0)
