# pyre-strict
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple, Any
import base64
import os

import numpy as np
import pyproj
import cv2
from PIL import Image as PILImage  # type: ignore[import-not-found]
from opensfm import features, multiview, pymap, types, io, geometry
from opensfm import geo
from opensfm.dataset import DataSet
from opensfm import pygeometry
from opensfm.actions import export_geocoords


import matplotlib.cm as cm
from scipy.spatial import Delaunay, KDTree

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError as _rerun_import_error:
    raise ImportError(
        "The 'rerun-sdk' package is required to use export_rerun. "
        "Install it with: pip install rerun-sdk"
    ) from _rerun_import_error

logger: logging.Logger = logging.getLogger(__name__)

# --- Constants ---

# Colors
GPS_COLOR = [100, 180, 255]
GCP_COMPUTED_COLOR = [120, 255, 140]
GCP_REFERENCE_COLOR = [255, 230, 100]
GCP_ERROR_COLOR = [255, 100, 100]
FEATURE_COLOR = [100, 255, 220]
LABEL_COLOR = [220, 220, 220]
CAMERA_PATH_COLOR = [255, 255, 255]

# Sizes & Dimensions
SIZE_TIE_POINT = 0.2
SIZE_GPS_ARROW = 1.0
SIZE_GPS_RESIDUAL_LINE = 0.05
SIZE_GCP_TARGET = 2.0
SIZE_GCP_THICKNESS = 0.05
SIZE_GCP_2D_THICKNESS = 0.5
SIZE_GCP_COMPUTED_ARROW = 0.1
SIZE_GCP_RESIDUAL_LINE = 0.02
SIZE_2D_POINT_GCP = 5.0
SIZE_2D_POINT_FEATURE = 2.0
SIZE_LABEL_SHIFT_SHOT = 0.5
SIZE_MATCHGRAPH_LINE = 0.1
SIZE_CAMERA_PATH_LINE = 0.5
SIZE_GCP_2D_BOX_HALF = 5.0
SIZE_GCP_2D_OFFSET = 5.0

# Configuration
MAX_IMAGE_WIDTH = 1500
IMAGE_COMPRESSION_QUALITY = 50
DEFAULT_IMAGE_PLANE_DISTANCE = 2.0
KNN_FOR_SCALE = 5
MATCHGRAPH_MIN_INLIERS_FACTOR = 5
COVERAGE_GRID_STEPS = 10
COVERAGE_MARGIN = 0.01
COVERAGE_EXTENT_MARGIN = 0.1
COVERAGE_MIN_COUNT = 0.0
COVERAGE_MAX_COUNT = 20.0

# Colormaps
MATCHGRAPH_CMAP = cm.get_cmap("plasma")
COVERAGE_CMAP = cm.get_cmap("viridis")


# --- Abstractions ---

class Drawer(ABC):
    """Abstract base class for visualization backends."""

    @abstractmethod
    def init(self, title: str, output_path: str) -> None:
        pass

    @abstractmethod
    def log_shot(
        self,
        shot_id: str,
        pose: pygeometry.Pose,
        camera: pygeometry.Camera,
        image: Optional[np.ndarray],
        image_plane_distance: float,
    ) -> None:
        pass

    @abstractmethod
    def log_exif_orientation(
        self,
        shot_id: str,
        R: np.ndarray,
        t: np.ndarray,
        camera: pygeometry.Camera,
        image_plane_distance: float = 1.0,
    ) -> None:
        pass

    @abstractmethod
    def log_gps(
        self,
        shot_id: str,
        pose: pygeometry.Pose,
        gps_topo: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def log_gcp_2d_observations(
        self,
        shot_id: str,
        refs: List[Tuple[float, float]],
        comps: List[Tuple[float, float]],
        labels_refs: List[str],
        labels_comps: List[str],
        gcp_3d_errors: Dict[str, float],
    ) -> None:
        pass

    @abstractmethod
    def log_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        pass

    @abstractmethod
    def log_camera_path(self, points: List[np.ndarray]) -> None:
        pass

    @abstractmethod
    def log_gcp_3d(
        self,
        gcp_id: str,
        reference_pos: np.ndarray,
        computed_pos: Optional[np.ndarray],
        error: float = -1.0,
    ) -> None:
        pass

    @abstractmethod
    def log_matchgraph(self, lines: List[List[np.ndarray]], colors: List[List[int]]) -> None:
        pass

    @abstractmethod
    def log_coverage_map(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def log_reference_lla(self, lat: float, lon: float, alt: float) -> None:
        pass

    @abstractmethod
    def log_stats_camera_models(self, stats: Dict[str, Any], data: DataSet) -> None:
        pass

    @abstractmethod
    def log_stats_summary(self, stats: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_stats_gps_bias(self, stats: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def setup_blueprint(self, camera_ids: Sequence[str]) -> None:
        pass


class RerunDrawer(Drawer):
    """Concrete implementation for Rerun visualization."""

    def init(self, title: str, output_path: str) -> None:
        rr.init(title, spawn=False)
        rr.save(output_path)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    def setup_blueprint(self, camera_ids: Sequence[str]) -> None:
        # Left: summary + processing time + GPS/GCP errors + per-camera biases (stacked vertically)
        bias_views = [
            rrb.TextDocumentView(
                origin=f"STATS/BIAS/{_sanitize_entity_path_component(cam)}",
                name=f"CAMERA {cam} Bias",
            )
            for cam in camera_ids
        ]

        # Right: spatial viewport, then per-camera (params + residual + heatmap) rows
        camera_rows = [
            rrb.Horizontal(
                rrb.TextDocumentView(
                    origin=f"STATS/CAMERAS/{_sanitize_entity_path_component(cam)}/PARAMS",
                    name=f"{cam} Params",
                ),
                rrb.Spatial2DView(
                    origin=f"STATS/CAMERAS/{_sanitize_entity_path_component(cam)}/RESIDUAL",
                    name=f"{cam} Residuals",
                ),
                rrb.Spatial2DView(
                    origin=f"STATS/CAMERAS/{_sanitize_entity_path_component(cam)}/HEATMAP",
                    name=f"{cam} Heatmap",
                ),
            )
            for cam in camera_ids
        ]

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TextDocumentView(
                        origin="STATS/SUMMARY", name="Processing Summary"),
                    rrb.TextDocumentView(
                        origin="STATS/PROCESSING_TIME", name="Processing Time Details"),
                    rrb.TextDocumentView(
                        origin="STATS/ERRORS/GPS", name="GPS Errors"),
                    rrb.TextDocumentView(
                        origin="STATS/ERRORS/GCP", name="GCP Errors"),
                    rrb.TextDocumentView(
                        origin="STATS/ERRORS/ORIENTATION", name="Orientation Errors"),
                    *bias_views,
                ),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            origin="WORLD",
                            name="3D Scene",
                            background=[13, 17, 23],
                            line_grid=rrb.LineGrid3D(visible=False),
                            spatial_information=rrb.SpatialInformation(
                                show_axes=False,
                                show_bounding_box=False,
                            ),
                        ),
                        rrb.Spatial2DView(origin="SHOTS", name="Image View"),
                    ),
                    rrb.Vertical(*camera_rows),
                    row_shares=[0.8, 0.2],
                ),
                rrb.BlueprintPanel(state="expanded"),
                rrb.SelectionPanel(state="collapsed"),
                rrb.TimePanel(state="collapsed"),
                column_shares=[0.15, 0.85],
            ),
        )
        rr.send_blueprint(blueprint)

    def log_shot(
        self,
        shot_id: str,
        pose: pygeometry.Pose,
        camera: pygeometry.Camera,
        image: Optional[np.ndarray],
        image_plane_distance: float,
    ) -> None:
        # Pose
        t = pose.get_origin()
        R_world_from_camera = pose.get_rotation_matrix().T

        rr.log(
            f"WORLD/SHOTS/{shot_id}",
            rr.Transform3D(
                translation=t,
                mat3x3=R_world_from_camera,
                from_parent=False,
            ),
            static=True,
        )

        labels_shift = np.array([0, 0, SIZE_LABEL_SHIFT_SHOT])
        rr.log(
            f"WORLD/SHOTS/{shot_id}",
            rr.Points3D(
                positions=labels_shift,
                labels=[shot_id],
                radii=0.0,
                colors=[LABEL_COLOR],
            ),
            static=True,
        )

        # Pinhole & Image
        width = int(camera.width)
        height = int(camera.height)
        width, height = _get_scaled_dimensions(width, height)
        fx, fy, cx, cy = _get_camera_calibration(camera, width, height)

        rr.log(
            f"WORLD/SHOTS/{shot_id}/CAMERA",
            rr.Pinhole(
                resolution=[width, height],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                image_plane_distance=image_plane_distance,
            ),
            static=True,
        )

        if image is not None:
            img_compressed = rr.Image(image).compress(
                jpeg_quality=IMAGE_COMPRESSION_QUALITY)
            rr.log(f"WORLD/SHOTS/{shot_id}/CAMERA/IMAGE",
                   img_compressed, static=True)
            rr.log(f"SHOTS/IMAGE", img_compressed, static=True)

    def log_exif_orientation(
        self,
        shot_id: str,
        R: np.ndarray,
        t: np.ndarray,
        camera: pygeometry.Camera,
        image_plane_distance: float = 1.0,
    ) -> None:
        path = f"WORLD/EXIF_SHOTS/{shot_id}"

        # Log transform (camera-to-world)
        rr.log(
            path,
            rr.Transform3D(
                translation=t,
                mat3x3=R.T,
                from_parent=False,
            ),
            static=True,
        )

        # Log pinhole
        width = int(camera.width)
        height = int(camera.height)
        width, height = _get_scaled_dimensions(width, height)
        fx, fy, cx, cy = _get_camera_calibration(camera, width, height)

        rr.log(
            f"{path}/EXIF_SHOTS/CAMERA",
            rr.Pinhole(
                resolution=[width, height],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                image_plane_distance=image_plane_distance,
            ),
            static=True,
        )

    def log_gps(
        self,
        shot_id: str,
        pose: pygeometry.Pose,
        gps_topo: np.ndarray,
    ) -> None:
        base_path = f"WORLD/GPS/{shot_id}/"
        origin = gps_topo.copy()
        origin[2] += SIZE_GPS_ARROW
        vector = np.array([0, 0, -SIZE_GPS_ARROW])

        rr.log(
            f"{base_path}/POSITION",
            rr.Arrows3D(
                origins=[origin],
                vectors=[vector],
                colors=[GPS_COLOR],
                radii=SIZE_GPS_ARROW / 8.0,
            ),
            static=True,
        )
        rr.log(
            f"{base_path}/RESIDUAL",
            rr.LineStrips3D(
                [[pose.get_origin(), gps_topo]],
                colors=[GPS_COLOR],
                radii=SIZE_GPS_RESIDUAL_LINE,
            ),
            static=True,
        )

    def log_gcp_2d_observations(
        self,
        shot_id: str,
        refs: List[Tuple[float, float]],
        comps: List[Tuple[float, float]],
        labels_refs: List[str],
        labels_comps: List[str],
        gcp_3d_errors: Dict[str, float],
    ) -> None:
        # Reference: Checkerboard
        if refs:
            box_centers = []
            box_half_sizes = []
            box_colors = []
            outline_centers = []
            outline_half_sizes = []
            outline_colors = []

            q_half = SIZE_GCP_2D_BOX_HALF
            offset = SIZE_GCP_2D_OFFSET

            for (x, y) in refs:
                # Top-Left quadrant
                box_centers.append([x - offset, y - offset])
                box_half_sizes.append([q_half, q_half])
                box_colors.append(GCP_REFERENCE_COLOR)
                # Bottom-Right quadrant
                box_centers.append([x + offset, y + offset])
                box_half_sizes.append([q_half, q_half])
                box_colors.append(GCP_REFERENCE_COLOR)
                # Outline
                outline_centers.append([x, y])
                outline_half_sizes.append([q_half * 2, q_half * 2])
                outline_colors.append(GCP_REFERENCE_COLOR)

            rr.log(
                f"SHOTS/IMAGE/GCP/REFERENCE",
                rr.Boxes2D(
                    centers=box_centers,
                    half_sizes=box_half_sizes,
                    colors=box_colors,
                    radii=SIZE_GCP_2D_THICKNESS,
                ),
                static=True,
            )
            rr.log(
                f"SHOTS/IMAGE/GCP/REFERENCE/OUTLINE",
                rr.Boxes2D(
                    centers=outline_centers,
                    half_sizes=outline_half_sizes,
                    colors=outline_colors,
                    radii=SIZE_GCP_2D_THICKNESS,
                ),
                static=True,
            )
        else:
            rr.log(f"SHOTS/IMAGE/GCP/REFERENCE", rr.Clear(recursive=True))
            rr.log(f"SHOTS/IMAGE/GCP/REFERENCE/OUTLINE",
                   rr.Clear(recursive=True))
            rr.log(f"SHOTS/IMAGE/GCP/REFERENCE/LABELS",
                   rr.Clear(recursive=True))

        # Computed: Arrow
        if comps:
            # Match computed to reference to draw arrows
            ref_map = {label: pos for label, pos in zip(labels_refs, refs)}
            origins = []
            vectors = []
            valid_labels = []

            for label, comp_pos in zip(labels_comps, comps):
                if label in ref_map:
                    ref_pos = ref_map[label]
                    vec = [comp_pos[0] - ref_pos[0], comp_pos[1] - ref_pos[1]]
                    origins.append(ref_pos)
                    vectors.append(vec)

                    # Log the error as label
                    error_3d = gcp_3d_errors.get(label, -1.0)
                    if error_3d >= 0:
                        valid_labels.append(f"{label}: {error_3d:.3f}m")
                    else:
                        valid_labels.append(label)

            if origins:
                rr.log(
                    f"SHOTS/IMAGE/GCP/COMPUTED",
                    rr.Arrows2D(
                        origins=origins,
                        vectors=vectors,
                        colors=[GCP_COMPUTED_COLOR],
                        labels=valid_labels,
                        radii=SIZE_GCP_2D_THICKNESS,
                    ),
                    static=True,
                )
            else:
                rr.log(f"SHOTS/IMAGE/GCP/COMPUTED", rr.Clear(recursive=True))
        else:
            rr.log(f"SHOTS/IMAGE/GCP/COMPUTED", rr.Clear(recursive=True))

    def log_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        rr.log(
            "WORLD/POINTS",
            rr.Points3D(
                positions=positions,
                colors=colors,
                radii=SIZE_TIE_POINT,
            ),
            static=True,
        )

    def log_camera_path(self, points: List[np.ndarray]) -> None:
        rr.log(
            "WORLD/PATH",
            rr.LineStrips3D(
                [points],
                colors=[CAMERA_PATH_COLOR],
                radii=SIZE_CAMERA_PATH_LINE,
            ),
            static=True,
        )

    def log_gcp_3d(
        self,
        gcp_id: str,
        reference_pos: np.ndarray,
        computed_pos: Optional[np.ndarray],
        error: float = -1.0,
    ) -> None:
        base_path = f"WORLD/GCP/{gcp_id}"
        quad_r = SIZE_GCP_TARGET / 4.0
        thickness = SIZE_GCP_THICKNESS
        pos_np = np.array(reference_pos)
        color = GCP_REFERENCE_COLOR if computed_pos is not None else GCP_ERROR_COLOR

        # Checkerboard
        centers = [
            pos_np + [-quad_r, quad_r, 0],
            pos_np + [quad_r, -quad_r, 0]
        ]
        half_sizes = [[quad_r, quad_r, thickness]] * 2

        rr.log(
            f"{base_path}/TARGET",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                colors=[color] * 2,
                fill_mode="solid",
            ),
            static=True,
        )
        rr.log(
            f"{base_path}/OUTLINE",
            rr.Boxes3D(
                centers=[pos_np],
                half_sizes=[[SIZE_GCP_TARGET / 2.0,
                             SIZE_GCP_TARGET / 2.0, thickness]],
                colors=[color],
                radii=thickness,
                fill_mode="major_wireframe",
            ),
            static=True,
        )

        label = gcp_id
        if error >= 0:
            label = f"{gcp_id}: {error:.3f}m"

        rr.log(
            f"{base_path}",
            rr.Points3D(
                positions=[pos_np + [0, 0, SIZE_GCP_TARGET]],
                labels=[label],
                radii=0.0,
                colors=[LABEL_COLOR],
            ),
            static=True,
        )

        if computed_pos is not None:
            rr.log(
                f"{base_path}/COMPUTED",
                rr.Arrows3D(
                    origins=[reference_pos],
                    vectors=[computed_pos - reference_pos],
                    colors=[GCP_COMPUTED_COLOR],
                    radii=SIZE_GCP_COMPUTED_ARROW,
                ),
                static=True,
            )

    def log_matchgraph(self, lines: List[List[np.ndarray]], colors: List[List[int]]) -> None:
        rr.log(
            "WORLD/STATS/MATCHGRAPH",
            rr.LineStrips3D(
                lines,
                colors=colors,
                radii=SIZE_MATCHGRAPH_LINE,
            ),
            static=True,
        )

    def log_coverage_map(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        rr.log(
            "WORLD/STATS/COVERAGE",
            rr.Mesh3D(
                vertex_positions=vertices,
                vertex_colors=colors,
                triangle_indices=indices,
            ),
            static=True,
        )

    def log_reference_lla(self, lat: float, lon: float, alt: float) -> None:
        rr.log(
            "world/reference",
            rr.TextDocument(
                f"Reference LLA:\n"
                f"Latitude: {lat:.6f}°\n"
                f"Longitude: {lon:.6f}°\n"
                f"Altitude: {alt:.2f}m",
            ),
        )

    def log_stats_camera_models(self, stats: Dict[str, Any], data: DataSet) -> None:
        if "camera_errors" not in stats:
            return

        for camera, params in stats["camera_errors"].items():
            sanitized_id = _sanitize_entity_path_component(camera)
            sanitized_id_spaces = _sanitize_entity_path_component(
                camera, strip_spaces=False)

            initial = params.get("initial_values", {})
            optimized = params.get("optimized_values", {})

            keys = sorted(list(set(initial.keys()) | set(optimized.keys())))
            columns = {"State": ["Initial", "Optimized"]}
            for k in keys:
                columns[k] = [initial.get(k), optimized.get(k)]

            _log_dataframe(
                f"STATS/CAMERAS/{sanitized_id}/PARAMS",
                columns,
                static=True,
            )

            # Residuals + heatmap as real images for Spatial2D views
            res_path = os.path.join(
                data.data_path, "stats", f"residuals_{sanitized_id_spaces}.png")
            heat_path = os.path.join(
                data.data_path, "stats", f"heatmap_{sanitized_id_spaces}.png")

            res_img = _try_load_image_as_numpy(data, res_path)
            if res_img is not None:
                rr.log(f"STATS/CAMERAS/{sanitized_id}/RESIDUAL",
                       rr.Image(res_img), static=True)

            heat_img = _try_load_image_as_numpy(data, heat_path)
            if heat_img is not None:
                rr.log(f"STATS/CAMERAS/{sanitized_id}/HEATMAP",
                       rr.Image(heat_img), static=True)

    def log_stats_summary(self, stats: Dict[str, Any]) -> None:
        # Keep Dataset + Reconstruction as markdown, but remove the time-details table (moved to DataFrame)
        md = "# Processing Summary\n\n"

        if "processing_statistics" in stats:
            ps = stats["processing_statistics"]
            md += "## Dataset\n"
            md += f"- **Date**: {ps.get('date', 'N/A')}\n"
            if "area" in ps:
                md += f"- **Area Covered**: {ps['area'] / 1e6:.6f} km²\n"
            if "steps_times" in ps and "Total Time" in ps["steps_times"]:
                md += f"- **Total Time**: {ps['steps_times']['Total Time']:.2f} s\n"
            md += "\n"

            if "steps_times" in ps:
                steps = list(ps["steps_times"].keys())
                times = [float(ps["steps_times"][s]) for s in steps]
                _log_dataframe(
                    "STATS/PROCESSING_TIME",
                    {"Step": steps, "Time (seconds)": times},
                    static=True,
                )

        if "reconstruction_statistics" in stats:
            rs = stats["reconstruction_statistics"]
            rec_shots = rs.get("reconstructed_shots_count", 0)
            init_shots = rs.get("initial_shots_count", 0)
            rec_points = rs.get("reconstructed_points_count", 0)
            init_points = rs.get("initial_points_count", 0)

            ratio_shots = rec_shots / init_shots * 100 if init_shots > 0 else 0
            ratio_points = rec_points / init_points * 100 if init_points > 0 else 0

            md += "## Reconstruction\n"
            md += f"- **Images**: {rec_shots}/{init_shots} ({ratio_shots:.1f}%)\n"
            md += f"- **Points**: {rec_points}/{init_points} ({ratio_points:.1f}%)\n"
            md += f"- **Components**: {rs.get('components', 0)}\n"

            if "features_statistics" in stats:
                fs = stats["features_statistics"]
                if "detected_features" in fs:
                    md += f"- **Detected Features (median)**: {fs['detected_features'].get('median', 'N/A')}\n"
                if "reconstructed_features" in fs:
                    md += f"- **Reconstructed Features (median)**: {fs['reconstructed_features'].get('median', 'N/A')}\n"

        rr.log(
            "STATS/SUMMARY",
            rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )

    def log_stats_gps_bias(self, stats: Dict[str, Any]) -> None:
        def _log_errors_df(
            dst_path: str,
            err_stats: Dict[str, Any],
            comps: Optional[List[str]] = None,
            unit: str = "meters",
        ) -> None:
            if comps is None:
                comps = ["x", "y", "z"]

            rows_component: List[str] = []
            rows_mean: List[Optional[float]] = []
            rows_sigma: List[Optional[float]] = []
            rows_rms: List[Optional[float]] = []

            for c in comps:
                rows_component.append(c.upper() if len(c)
                                      == 1 else c.capitalize())
                rows_mean.append(float(err_stats.get("mean", {}).get(c, 0.0)))
                rows_sigma.append(float(err_stats.get("std", {}).get(c, 0.0)))
                rows_rms.append(float(err_stats.get("error", {}).get(c, 0.0)))

            rows_component.append("TOTAL")
            rows_mean.append(None)
            rows_sigma.append(None)
            rows_rms.append(float(err_stats.get("average_error", 0.0)))

            _log_dataframe(
                dst_path,
                {
                    "Component": rows_component,
                    f"Mean ({unit})": rows_mean,
                    f"Sigma ({unit})": rows_sigma,
                    f"RMS ({unit})": rows_rms,
                },
                static=True,
            )

        gps_key = "gps_errors"
        if gps_key in stats and "average_error" in stats[gps_key]:
            _log_errors_df("STATS/ERRORS/GPS", stats[gps_key])

        gcp_key = "gcp_errors"
        if gcp_key in stats and "average_error" in stats[gcp_key]:
            _log_errors_df("STATS/ERRORS/GCP", stats[gcp_key])

        opk_key = "opk_errors"
        if opk_key in stats and "average_error" in stats[opk_key]:
            _log_errors_df(
                "STATS/ERRORS/ORIENTATION",
                stats[opk_key],
                comps=["omega", "phi", "kappa"],
                unit="degrees",
            )

        if "camera_errors" in stats:
            for camera, params in stats["camera_errors"].items():
                if "bias" not in params:
                    continue
                bias = params["bias"]
                s = float(bias.get("scale", 1.0))
                t = bias.get("translation", [0.0, 0.0, 0.0])
                r = bias.get("rotation", [0.0, 0.0, 0.0])

                _log_dataframe(
                    f"STATS/BIAS/{_sanitize_entity_path_component(camera)}",
                    {
                        "Component": ["Scale", "Translation X", "Translation Y", "Translation Z", "Rotation X", "Rotation Y", "Rotation Z"],
                        "Value (meters or degrees)": [s, float(t[0]), float(t[1]), float(t[2]), float(r[0]), float(r[1]), float(r[2])],
                    },
                    static=True,
                )


# --- Main Logic ---

def run_dataset(
    data: DataSet,
    output: Optional[str] = None,
    reconstruction_index: int = 0,
    proj: bool = False,
) -> None:
    """Export reconstruction to Rerun format for 3D visualization."""

    # 1. Load Data
    reconstructions = data.load_reconstruction()
    if not reconstructions:
        logger.error("No reconstructions found in dataset")
        return

    if reconstruction_index >= len(reconstructions):
        logger.error(
            f"Reconstruction index {reconstruction_index} out of range")
        return

    reconstruction = reconstructions[reconstruction_index]

    centering = None
    projection = None
    if proj:
        gcp_list_file = data._gcp_list_file()
        if data.io_handler.isfile(gcp_list_file):
            with data.io_handler.open_rt(gcp_list_file) as f:
                proj_str = io.read_gcp_projection_string(f)

            if proj_str:
                logger.info(
                    f"Transforming reconstruction to geocoords: {proj_str}")
                reference = data.load_reference()

                # Transform to the GCP CS
                projection = geo.construct_proj_transformer(
                    proj_str, inverse=True)
                geo.transform_reconstruction_with_proj(
                    reconstruction, projection)

                # Center the reconstruction to avoid large coordinates
                all_points = [
                    p.coordinates for p in reconstruction.points.values()]
                if all_points:
                    center = np.mean(all_points, axis=0)
                else:
                    all_origins = [shot.pose.get_origin()
                                   for shot in reconstruction.shots.values()]
                    if all_origins:
                        center = np.mean(all_origins, axis=0)
                    else:
                        center = np.zeros(3)

                logger.info(f"Centering reconstruction at {center}")
                centering = -center

                for point in reconstruction.points.values():
                    point.coordinates += centering
                for shot in reconstruction.shots.values():
                    shot.pose.set_origin(shot.pose.get_origin() + centering)
        else:
            logger.warning("No gcp_list.txt found, skipping projection.")

    logger.info(
        f"Exporting reconstruction {reconstruction_index} with "
        f"{len(reconstruction.shots)} shots, "
        f"{len(reconstruction.points)} points"
    )

    # 2. Initialize Drawer
    output_path = output or data.data_path + "/rerun.rrd"
    drawer = RerunDrawer()
    drawer.init("OpenSfM Reconstruction", output_path)

    camera_ids = sorted(
        {shot.camera.id for shot in reconstruction.shots.values()})

    drawer.setup_blueprint(camera_ids)
    logger.info(f"Saving Rerun data to {output_path}")

    # 3. Precompute Data
    image_plane_distance = _compute_image_plane_distance(reconstruction)

    # Precompute GCP observations per shot
    gcp = data.load_ground_control_points() or []
    gcp_triangulations = _compute_gcp_triangulations(data, reconstruction, gcp)
    gcp_3d_errors = _compute_gcp_3d_errors(
        reconstruction, gcp, gcp_triangulations, centering)
    shot_gcp_obs = _precompute_gcp_observations(
        data, reconstruction, gcp, gcp_triangulations)

    # 4. Time-Dependent Logging (Shots)
    reference = reconstruction.reference

    for shot_id, shot in reconstruction.shots.items():
        sequence_id = _get_shot_sequence_id(shot_id)

        # Load image
        image = None
        try:
            image = data.load_image(shot_id)
            width = int(shot.camera.width)
            height = int(shot.camera.height)
            width, height = _get_scaled_dimensions(width, height)
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height),
                                   interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.warning(f"Could not load image for {shot_id}: {e}")

        # Log Shot Pose & Image
        drawer.log_shot(shot_id, shot.pose, shot.camera,
                        image, image_plane_distance)

        # Log GPS & EXIF Orientation
        gps_pos = None
        if reference and shot.metadata and shot.metadata.gps_position.has_value:
            gps_topo = shot.metadata.gps_position.value
            gps_pos = gps_topo
            if projection is not None:
                gps_pos = np.array(geo.transform_to_proj(
                    gps_topo, reference, projection))
                if centering is not None:
                    gps_pos += centering

            drawer.log_gps(shot_id, shot.pose, gps_pos)

        has_opk = shot.metadata and shot.metadata.opk_angles.has_value
        if gps_pos is not None or has_opk:
            t_exif = shot.pose.get_origin()
            if has_opk:
                opk = shot.metadata.opk_angles.value
                R_exif = geometry.rotation_from_opk(
                    np.radians(opk[0]), np.radians(opk[1]), np.radians(opk[2])
                )
            else:
                R_exif = shot.pose.get_rotation_matrix()

            drawer.log_exif_orientation(
                shot_id, R_exif, t_exif, shot.camera, image_plane_distance)

        # Log GCP 2D Observations
        if shot_id in shot_gcp_obs:
            obs = shot_gcp_obs[shot_id]
            drawer.log_gcp_2d_observations(
                shot_id,
                obs["refs"],
                obs["comps"],
                obs["labels_refs"],
                obs["labels_comps"],
                gcp_3d_errors,
            )
        else:
            drawer.log_gcp_2d_observations(
                shot_id, [], [], [], [], gcp_3d_errors)

    # 5. Static Logging (Structure & Stats)

    # Reference
    if reconstruction.reference:
        ref = reconstruction.reference
        drawer.log_reference_lla(ref.lat, ref.lon, ref.alt)

    # Load and log stats.json if available
    try:
        stats = data.load_stats()
        if stats:
            drawer.log_stats_summary(stats)
            drawer.log_stats_camera_models(stats, data)
            drawer.log_stats_gps_bias(stats)
    except Exception as e:
        logger.warning(f"Could not load or log stats.json: {e}")

    # 3D Points
    _export_points(reconstruction, drawer)

    # Camera Path
    _export_camera_path(reconstruction, drawer)

    # GCP 3D Geometry
    if gcp:
        _export_gcp_3d(data, reconstruction, gcp,
                       gcp_triangulations, gcp_3d_errors, drawer, centering)

    # Match Graph
    if data.tracks_exists():
        tracks_manager = data.load_tracks_manager()
        _export_matchgraph(data, reconstruction, tracks_manager, drawer)

    # Coverage Map
    _export_coverage_map(data, reconstruction, drawer)

    logger.info(f"Rerun export completed: {output_path}")
    logger.info(f"Open with: rerun {output_path}")


# --- Helper Functions ---

def _get_shot_sequence_id(shot_id: str) -> int:
    try:
        numbers = re.findall(r'\d+', shot_id)
        if numbers:
            return int(numbers[-1])
    except:
        pass
    return hash(shot_id) & 0x7FFFFFFF


def _get_scaled_dimensions(width: int, height: int) -> tuple[int, int]:
    if width > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / width
        return int(width * scale), int(height * scale)
    return width, height


def _compute_image_plane_distance(reconstruction: types.Reconstruction) -> float:
    positions = np.array([shot.pose.get_origin()
                         for shot in reconstruction.shots.values()])
    if len(positions) < 2:
        return DEFAULT_IMAGE_PLANE_DISTANCE
    try:
        tree = KDTree(positions)
        dists, _ = tree.query(positions, k=KNN_FOR_SCALE)
        median_nn_dist = np.median(dists[:, 1])
        return float(median_nn_dist)
    except ImportError:
        return DEFAULT_IMAGE_PLANE_DISTANCE


def _get_camera_calibration(camera: pygeometry.Camera, width: int, height: int):
    fx = fy = camera.focal * max(width, height)
    if hasattr(camera, "focal_x") and hasattr(camera, "focal_y"):
        fx = camera.focal_x * max(width, height)
        fy = camera.focal_y * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    if hasattr(camera, "c_x") and hasattr(camera, "c_y"):
        cx = camera.c_x * max(width, height) + width / 2.0
        cy = camera.c_y * max(width, height) + height / 2.0
    return fx, fy, cx, cy


def _compute_gcp_triangulations(
    data: DataSet,
    reconstruction: types.Reconstruction,
    gcp: List[pymap.GroundControlPoint]
) -> Dict[str, Optional[np.ndarray]]:
    triangulations = {}
    for point in gcp:
        if point.lla:
            triangulations[point.id] = multiview.triangulate_gcp(
                point, reconstruction.shots, data.config["gcp_reprojection_error_threshold"]
            )
    return triangulations


def _compute_gcp_3d_errors(
    reconstruction: types.Reconstruction,
    gcp: List[pymap.GroundControlPoint],
    triangulations: Dict[str, Optional[np.ndarray]],
    centering: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    errors = {}
    reference = reconstruction.reference
    for point in gcp:
        if point.id in triangulations and triangulations[point.id] is not None:
            comp_pos = triangulations[point.id]

            if centering is not None:
                ref_pos = np.array(point.coordinates) + centering
                errors[point.id] = np.linalg.norm(comp_pos - ref_pos)
            elif point.lla:
                ref_pos = reference.to_topocentric(*point.lla_vec)
                errors[point.id] = np.linalg.norm(comp_pos - ref_pos)
            else:
                errors[point.id] = -1.0
        else:
            errors[point.id] = -1.0
    return errors


def _precompute_gcp_observations(
    data: DataSet,
    reconstruction: types.Reconstruction,
    gcp: List[pymap.GroundControlPoint],
    triangulations: Dict[str, Optional[np.ndarray]]
) -> Dict[str, Dict[str, Any]]:
    """Precompute 2D GCP observations per shot."""
    shot_obs = defaultdict(
        lambda: {"refs": [], "comps": [], "labels_refs": [], "labels_comps": []})
    reference = reconstruction.reference

    for point in gcp:
        if not point.lla:
            continue

        triangulated = triangulations.get(point.id)

        # Project to shots
        for obs in point.observations:
            shot_id = obs.shot_id
            if shot_id not in reconstruction.shots:
                continue

            shot = reconstruction.shots[shot_id]
            width = int(shot.camera.width)
            height = int(shot.camera.height)
            width, height = _get_scaled_dimensions(width, height)
            normalizer = max(width, height)

            # Reference pixel
            px = obs.projection[0] * normalizer + width / 2.0
            py = obs.projection[1] * normalizer + height / 2.0

            shot_obs[shot_id]["refs"].append((px, py))
            shot_obs[shot_id]["labels_refs"].append(point.id)

            # Computed pixel
            if triangulated is not None:
                projected = shot.project(triangulated)
                px_comp = projected[0] * normalizer + width / 2.0
                py_comp = projected[1] * normalizer + height / 2.0
                shot_obs[shot_id]["comps"].append((px_comp, py_comp))
                shot_obs[shot_id]["labels_comps"].append(point.id)

    return shot_obs


def _export_points(reconstruction: types.Reconstruction, drawer: Drawer) -> None:
    if not reconstruction.points:
        return
    positions = []
    colors = []
    for point in reconstruction.points.values():
        positions.append(point.coordinates)
        c = point.color
        if all(val <= 1.0 for val in c):
            c = [int(val * 255) for val in c]
        colors.append(c)

    drawer.log_points(np.array(positions), np.array(colors, dtype=np.uint8))
    logger.info(f"Exported {len(positions)} automatic tie points")


def _export_camera_path(reconstruction: types.Reconstruction, drawer: Drawer) -> None:
    shots_with_time = []
    for shot in reconstruction.shots.values():
        if shot.metadata and shot.metadata.capture_time.has_value:
            shots_with_time.append(shot)

    if len(shots_with_time) < 2:
        return

    shots_with_time.sort(key=lambda x: x.metadata.capture_time.value)
    points = [shot.pose.get_origin() for shot in shots_with_time]
    drawer.log_camera_path(points)
    logger.info(f"Exported camera path with {len(points)} points")


def _export_gcp_3d(
    data: DataSet,
    reconstruction: types.Reconstruction,
    gcp: List[pymap.GroundControlPoint],
    triangulations: Dict[str, Optional[np.ndarray]],
    gcp_3d_errors: Dict[str, float],
    drawer: Drawer,
    centering: Optional[np.ndarray] = None,
) -> None:
    reference = reconstruction.reference
    count = 0
    for point in gcp:
        pos = None
        if centering is not None:
            pos = np.array(point.coordinates) + centering
        elif point.lla:
            pos = reference.to_topocentric(*point.lla_vec)

        if pos is not None:
            triangulated = triangulations.get(point.id)
            error = gcp_3d_errors.get(point.id, -1.0)

            drawer.log_gcp_3d(point.id, pos, triangulated, error)
            count += 1
    logger.info(f"Exported {count} Ground Control Points")


def _export_matchgraph(
    data: DataSet,
    reconstruction: types.Reconstruction,
    tracks_manager: pymap.TracksManager,
    drawer: Drawer
) -> None:
    logger.info("Exporting match graph...")
    all_shots = list(reconstruction.shots.keys())
    all_points = list(reconstruction.points.keys())
    connectivity = tracks_manager.get_all_pairs_connectivity(
        all_shots, all_points)

    if not connectivity:
        return

    all_values = list(connectivity.values())
    if not all_values:
        return

    lowest = np.percentile(all_values, 5)
    highest = np.percentile(all_values, 95)

    lines = []
    colors = []
    min_inliers = data.config.get("resection_min_inliers", 15)

    for (node1, node2), edge in connectivity.items():
        if edge < MATCHGRAPH_MIN_INLIERS_FACTOR * min_inliers:
            continue
        if node1 not in reconstruction.shots or node2 not in reconstruction.shots:
            continue

        p1 = reconstruction.shots[node1].pose.get_origin()
        p2 = reconstruction.shots[node2].pose.get_origin()

        c_val = max(0.0, min(1.0, 1.0 - (float(edge) - lowest) /
                    (highest - lowest + 1e-6)))
        rgba = MATCHGRAPH_CMAP(1.0 - c_val)

        lines.append([p1, p2])
        colors.append([int(c * 255) for c in rgba[:3]])

    if lines:
        drawer.log_matchgraph(lines, colors)
        logger.info(f"Exported match graph with {len(lines)} edges")


def _export_coverage_map(
    data: DataSet,
    reconstruction: types.Reconstruction,
    drawer: Drawer
) -> None:
    logger.info("Exporting coverage map...")
    if not reconstruction.points:
        return

    points = np.array([p.coordinates for p in reconstruction.points.values()])
    if len(points) == 0:
        return

    min_pt = np.percentile(points, 1, axis=0)
    max_pt = np.percentile(points, 99, axis=0)
    median_z = np.median(points[:, 2])

    extent = max_pt - min_pt
    min_pt -= extent * COVERAGE_EXTENT_MARGIN
    max_pt += extent * COVERAGE_EXTENT_MARGIN

    vertices = []
    for shot in reconstruction.shots.values():
        w = shot.camera.width
        h = shot.camera.height

        m = COVERAGE_MARGIN
        steps = np.linspace(0, 1, COVERAGE_GRID_STEPS)
        steps = m + steps * (1 - 2 * m)

        xs = w * steps
        ys = h * steps
        pixels = features.normalized_image_coordinates(
            np.array([[x, y] for y in ys for x in xs]), w, h)

        bearings = shot.camera.pixel_bearing_many(pixels)
        R_wc = shot.pose.get_rotation_matrix().T
        bearings_world = bearings @ R_wc.T
        origin = shot.pose.get_origin()

        for direction in bearings_world:
            if abs(direction[2]) < 1e-6:
                continue
            t = (median_z - origin[2]) / direction[2]
            if t <= 0:
                continue
            p_ground = origin + t * direction

            if (p_ground[0] >= min_pt[0] and p_ground[0] <= max_pt[0] and
                    p_ground[1] >= min_pt[1] and p_ground[1] <= max_pt[1]):
                vertices.append(p_ground)

    if not vertices or len(vertices) < 3:
        return

    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, :2])
    indices = tri.simplices

    counts = np.zeros(len(vertices), dtype=int)
    for shot in reconstruction.shots.values():
        p_cam = shot.pose.transform_many(vertices)
        valid_z = p_cam[:, 2] > 0
        indices_valid_z = np.where(valid_z)[0]

        if len(indices_valid_z) == 0:
            continue

        p_cam_valid = p_cam[indices_valid_z]
        projections = shot.camera.project_many(p_cam_valid)

        w = shot.camera.width
        h = shot.camera.height
        normalizer = max(w, h)

        px = projections[:, 0] * normalizer + w / 2.0
        py = projections[:, 1] * normalizer + h / 2.0

        in_view = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        counts[indices_valid_z[in_view]] += 1

    norm_counts = np.clip(counts, COVERAGE_MIN_COUNT, COVERAGE_MAX_COUNT)
    norm_counts = (norm_counts - COVERAGE_MIN_COUNT) / \
        (COVERAGE_MAX_COUNT - COVERAGE_MIN_COUNT)

    colors = COVERAGE_CMAP(norm_counts)
    vertex_colors = (colors[:, :3] * 255).astype(np.uint8)

    drawer.log_coverage_map(vertices, vertex_colors, indices)
    logger.info(f"Exported coverage map with {len(vertices)} vertices")


def _sanitize_entity_path_component(value: str, strip_spaces: bool = True) -> str:
    # Rerun entity paths use "/" as hierarchy separator.
    sanitized = value.replace("/", "_")
    if strip_spaces:
        sanitized = sanitized.replace(" ", "_")
    return sanitized


def _try_load_image_as_numpy(data: DataSet, path: str) -> Optional[np.ndarray]:
    if not data.io_handler.isfile(path):
        return None
    try:
        with data.io_handler.open_rb(path) as f:
            raw = f.read()
        img = PILImage.open(BytesIO(raw))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None


def _log_dataframe(path: str, columns: Dict[str, Sequence[Any]], *, static: bool = True) -> None:
    """
    Log a dataframe in the way expected by `DataframeView` (Selection View).

    Columns must be equal-length sequences (pad with None if needed).
    """
    if not columns:
        return

    cols: Dict[str, List[Any]] = {k: list(v) for k, v in columns.items()}
    max_len = max((len(v) for v in cols.values()), default=0)
    for k, v in cols.items():
        if len(v) < max_len:
            v.extend([None] * (max_len - len(v)))

    # Convert dict to a Markdown Table string
    header = "| " + " | ".join(cols.keys()) + " |"
    separator = "| " + \
        " | ".join([":---"] + ["---:"] * (len(cols) - 1)) + " |"
    body = []
    for i in range(max_len):
        cells = []
        for v in cols.values():
            val = v[i]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val) if val is not None else "")
        body.append("| " + " | ".join(cells) + " |")
    md_table = "\n".join([header, separator] + body)

    rr.log(path, rr.TextDocument(
        md_table, media_type="text/markdown"), static=True)
