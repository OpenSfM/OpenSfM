# pyre-strict
"""Preview dataset: a thin wrapper over the mother dataset for fast preview mode.

Symlinks shared read-only data (images, exif, config, camera_models, etc.)
from the parent dataset and provides its own output area under a ``preview/``
subfolder for per-image reconstructions and meshes.
"""

import json
import logging
import os
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from opensfm import config, features, geo, io, pygeometry, pymap, rig, types
from opensfm.dataset import DataSet


logger: logging.Logger = logging.getLogger(__name__)


class PreviewDataset:
    """Lightweight dataset that reads from a parent DataSet and writes preview outputs.

    Read-only data (images, exif, features, matches, camera models, reference,
    rig, config) are accessed through the parent dataset via symlinks.

    Preview-specific outputs (per-image reconstruction JSON, per-image meshes)
    are written into a ``preview/`` directory inside the parent data path.
    """

    def __init__(self, parent: DataSet) -> None:
        self._parent = parent
        self.io_handler: io.IoFilesystemBase = parent.io_handler
        self.config: Dict[str, Any] = parent.config
        self._preview_path = os.path.join(parent.data_path, "preview")
        self.io_handler.mkdir_p(self._preview_path)

    @property
    def data_path(self) -> str:
        return self._parent.data_path

    # ------------------------------------------------------------------
    # Image list & image I/O — delegated to parent
    # ------------------------------------------------------------------

    def images(self) -> List[str]:
        return self._parent.images()

    def open_image_file(self, image: str) -> BinaryIO:
        return self._parent.open_image_file(image)

    def load_image(
        self,
        image: str,
        unchanged: bool = False,
        anydepth: bool = False,
        grayscale: bool = False,
    ) -> NDArray:
        return self._parent.load_image(
            image, unchanged=unchanged, anydepth=anydepth, grayscale=grayscale
        )

    def image_size(self, image: str) -> Tuple[int, int]:
        return self._parent.image_size(image)

    # ------------------------------------------------------------------
    # Exif — delegated
    # ------------------------------------------------------------------

    def load_exif(self, image: str) -> Dict[str, Any]:
        return self._parent.load_exif(image)

    def exif_exists(self, image: str) -> bool:
        return self._parent.exif_exists(image)

    # ------------------------------------------------------------------
    # Features — delegated (read-only)
    # ------------------------------------------------------------------

    def features_exist(self, image: str) -> bool:
        return self._parent.features_exist(image)

    def load_features(self, image: str) -> Optional[features.FeaturesData]:
        return self._parent.load_features(image)

    def feature_type(self) -> str:
        return self._parent.feature_type()

    # ------------------------------------------------------------------
    # Matches — delegated (read-only)
    # ------------------------------------------------------------------

    def matches_exists(self, image: str) -> bool:
        return self._parent.matches_exists(image)

    def load_matches(self, image: str) -> Dict[str, NDArray]:
        return self._parent.load_matches(image)

    def find_matches(self, im1: str, im2: str) -> NDArray:
        return self._parent.find_matches(im1, im2)

    # ------------------------------------------------------------------
    # Camera models — delegated
    # ------------------------------------------------------------------

    def load_camera_models(self) -> Dict[str, pygeometry.Camera]:
        return self._parent.load_camera_models()

    def camera_models_overrides_exists(self) -> bool:
        return self._parent.camera_models_overrides_exists()

    def load_camera_models_overrides(self) -> Dict[str, pygeometry.Camera]:
        return self._parent.load_camera_models_overrides()

    # ------------------------------------------------------------------
    # Reference — delegated
    # ------------------------------------------------------------------

    def init_reference(self, images: Optional[List[str]] = None) -> None:
        self._parent.init_reference(images)

    def reference_exists(self) -> bool:
        return self._parent.reference_exists()

    def load_reference(self) -> geo.TopocentricConverter:
        return self._parent.load_reference()

    # ------------------------------------------------------------------
    # Rig — delegated
    # ------------------------------------------------------------------

    def load_rig_cameras(self) -> Dict[str, pymap.RigCamera]:
        return self._parent.load_rig_cameras()

    def load_rig_assignments(self) -> Dict[str, List[Tuple[str, str]]]:
        return self._parent.load_rig_assignments()

    # ------------------------------------------------------------------
    # GCP — delegated
    # ------------------------------------------------------------------

    def load_ground_control_points(self) -> List[pymap.GroundControlPoint]:
        return self._parent.load_ground_control_points()

    # ------------------------------------------------------------------
    # Masks / segmentation — delegated
    # ------------------------------------------------------------------

    def load_mask(self, image: str) -> Optional[NDArray]:
        return self._parent.load_mask(image)

    def load_segmentation(self, image: str) -> Optional[NDArray]:
        return self._parent.load_segmentation(image)

    def segmentation_labels(self) -> List[Dict[str, Any]]:
        return self._parent.segmentation_labels()

    def segmentation_ignore_values(self, image: str) -> List[int]:
        return self._parent.segmentation_ignore_values(image)

    def load_instances(self, image: str) -> Optional[NDArray]:
        return self._parent.load_instances(image)

    # ------------------------------------------------------------------
    # Words — delegated
    # ------------------------------------------------------------------

    def words_exist(self, image: str) -> bool:
        return self._parent.words_exist(image)

    def load_words(self, image: str) -> NDArray:
        return self._parent.load_words(image)

    # ------------------------------------------------------------------
    # Availability queries — features / matches on disk
    # ------------------------------------------------------------------

    def available_features(self) -> List[str]:
        """Return images for which features have been extracted."""
        return [im for im in self.images() if self.features_exist(im)]

    def available_matches(self) -> List[str]:
        """Return images for which matches files exist."""
        return [im for im in self.images() if self.matches_exists(im)]

    # ------------------------------------------------------------------
    # Preview-specific output paths
    # ------------------------------------------------------------------

    def _preview_reconstruction_path(self) -> str:
        return os.path.join(self._preview_path, "reconstructions")

    def _preview_mesh_path(self) -> str:
        return os.path.join(self._preview_path, "meshes")

    def _preview_reconstruction_file(self, image: str) -> str:
        return os.path.join(
            self._preview_reconstruction_path(),
            image + ".reconstruction.json",
        )

    def _preview_mesh_file(self, image: str) -> str:
        return os.path.join(self._preview_mesh_path(), image + ".mesh.json")

    # ------------------------------------------------------------------
    # Preview output: per-image reconstruction
    # ------------------------------------------------------------------

    def save_preview_reconstruction(
        self,
        image: str,
        reconstruction: types.Reconstruction,
    ) -> None:
        """Save a per-image reconstruction snapshot."""
        self.io_handler.mkdir_p(self._preview_reconstruction_path())
        path = self._preview_reconstruction_file(image)
        with self.io_handler.open_wt(path) as fout:
            io.json_dump(
                io.reconstructions_to_json([reconstruction]), fout, minify=True
            )

    def load_preview_reconstruction(
        self, image: str
    ) -> List[types.Reconstruction]:
        path = self._preview_reconstruction_file(image)
        with self.io_handler.open_rt(path) as fin:
            return io.reconstructions_from_json(io.json_load(fin))

    # ------------------------------------------------------------------
    # Preview output: per-image mesh
    # ------------------------------------------------------------------

    def save_preview_mesh(
        self,
        image: str,
        vertices: List[List[float]],
        faces: List[List[int]],
    ) -> None:
        """Save a per-image Delaunay mesh as JSON."""
        self.io_handler.mkdir_p(self._preview_mesh_path())
        path = self._preview_mesh_file(image)
        with self.io_handler.open_wt(path) as fout:
            io.json_dump({"vertices": vertices, "faces": faces},
                         fout, minify=True)

    def load_preview_mesh(
        self, image: str
    ) -> Tuple[List[List[float]], List[List[int]]]:
        path = self._preview_mesh_file(image)
        with self.io_handler.open_rt(path) as fin:
            data = io.json_load(fin)
            return data["vertices"], data["faces"]

    # ------------------------------------------------------------------
    # Full reconstruction save (cumulative preview state)
    # ------------------------------------------------------------------

    def save_reconstruction(
        self,
        reconstruction: List[types.Reconstruction],
        filename: Optional[str] = None,
    ) -> None:
        """Save the full reconstruction to the preview folder."""
        fname = filename or "reconstruction.json"
        path = os.path.join(self._preview_path, fname)
        with self.io_handler.open_wt(path) as fout:
            io.json_dump(io.reconstructions_to_json(reconstruction), fout)

    def load_reconstruction(
        self, filename: Optional[str] = None
    ) -> List[types.Reconstruction]:
        fname = filename or "reconstruction.json"
        path = os.path.join(self._preview_path, fname)
        with self.io_handler.open_rt(path) as fin:
            return io.reconstructions_from_json(io.json_load(fin))

    # ------------------------------------------------------------------
    # Profile log — delegated
    # ------------------------------------------------------------------

    def append_to_profile_log(self, content: str) -> None:
        self._parent.append_to_profile_log(content)

    # ------------------------------------------------------------------
    # Reports — stored in preview subfolder
    # ------------------------------------------------------------------

    def _report_path(self) -> str:
        return os.path.join(self._preview_path, "reports")

    def save_report(self, report_str: str, path: str) -> None:
        filepath = os.path.join(self._report_path(), path)
        self.io_handler.mkdir_p(os.path.dirname(filepath))
        with self.io_handler.open_wt(filepath) as fout:
            fout.write(report_str)

    def load_report(self, path: str) -> str:
        with self.io_handler.open_rt(
            os.path.join(self._report_path(), path)
        ) as fin:
            return fin.read()
