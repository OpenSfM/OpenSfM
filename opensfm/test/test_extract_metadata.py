# pyre-strict
import argparse
from types import SimpleNamespace
from typing import Any, Dict, List

from opensfm.actions import extract_metadata as extract_metadata_action
from opensfm.commands import extract_metadata as extract_metadata_command


class FakeData:
    def __init__(
        self,
        images: List[str],
        exifs: Dict[str, Dict[str, Any]],
        exif_overrides: Dict[str, Dict[str, Any]],
    ) -> None:
        self._images = images
        self._exifs = dict(exifs)
        self._exif_overrides = dict(exif_overrides)
        self.saved_camera_models: Dict[str, Any] = {}
        self.load_exif_calls = 0
        self.init_reference_calls = 0
        self.config = {
            "processes": 1,
        }

    def images(self) -> List[str]:
        return self._images

    def exif_exists(self, image: str) -> bool:
        return image in self._exifs

    def load_exif(self, image: str) -> Dict[str, Any]:
        self.load_exif_calls += 1
        return self._exifs[image]

    def save_exif(self, image: str, data: Dict[str, Any]) -> None:
        self._exifs[image] = data

    def exif_overrides_exists(self) -> bool:
        return bool(self._exif_overrides)

    def load_exif_overrides(self) -> Dict[str, Dict[str, Any]]:
        return self._exif_overrides

    def camera_models_overrides_exists(self) -> bool:
        return False

    def load_camera_models_overrides(self) -> Dict[str, Any]:
        raise AssertionError(
            "camera overrides should not be loaded in this test")

    def save_camera_models(self, camera_models: Dict[str, Any]) -> None:
        self.saved_camera_models = camera_models

    def init_reference(self) -> None:
        self.init_reference_calls += 1


def test_run_dataset_force_reextracts_and_initializes_reference(
    monkeypatch: Any,
) -> None:
    data = FakeData(
        ["image.jpg"],
        {"image.jpg": {"camera": "cached_camera", "model": "cached"}},
        {"image.jpg": {"gps": {"dop": 3.0}}},
    )

    calls = {"extract": 0}

    def fake_extract_exif(image: str, dataset: Any) -> Dict[str, Any]:
        del dataset
        calls["extract"] += 1
        return {
            "camera": "fresh_camera",
            "model": "fresh",
            "gps": {"latitude": 1.0, "longitude": 2.0},
        }

    def fake_camera_from_exif_metadata(exif_data: Dict[str, Any], dataset: Any) -> Any:
        del dataset
        return SimpleNamespace(id=exif_data["camera"])

    monkeypatch.setattr(extract_metadata_action,
                        "_extract_exif", fake_extract_exif)
    monkeypatch.setattr(
        extract_metadata_action.exif,
        "camera_from_exif_metadata",
        fake_camera_from_exif_metadata,
    )

    extract_metadata_action.run_dataset(data, force=True)

    assert calls["extract"] == 1
    assert data.load_exif_calls == 0
    assert data._exifs["image.jpg"]["camera"] == "fresh_camera"
    assert data._exifs["image.jpg"]["gps"]["dop"] == 3.0
    assert list(data.saved_camera_models.keys()) == ["fresh_camera"]
    assert data.init_reference_calls == 1


def test_run_dataset_without_force_uses_cached_exif_and_initializes_reference(
    monkeypatch: Any,
) -> None:
    data = FakeData(
        ["image.jpg"],
        {
            "image.jpg": {
                "camera": "cached_camera",
                "model": "cached",
                "gps": {"latitude": 1.0, "longitude": 2.0},
            }
        },
        {},
    )

    def fail_extract_exif(image: str, dataset: Any) -> Dict[str, Any]:
        del image, dataset
        raise AssertionError(
            "cached EXIF should be reused when force is false")

    def fake_camera_from_exif_metadata(exif_data: Dict[str, Any], dataset: Any) -> Any:
        del dataset
        return SimpleNamespace(id=exif_data["camera"])

    monkeypatch.setattr(extract_metadata_action,
                        "_extract_exif", fail_extract_exif)
    monkeypatch.setattr(
        extract_metadata_action.exif,
        "camera_from_exif_metadata",
        fake_camera_from_exif_metadata,
    )

    extract_metadata_action.run_dataset(data)

    assert data.load_exif_calls == 1
    assert list(data.saved_camera_models.keys()) == ["cached_camera"]
    assert data.init_reference_calls == 1


def test_command_passes_force_flag(monkeypatch: Any) -> None:
    parser = argparse.ArgumentParser()
    command = extract_metadata_command.Command()
    command.add_arguments(parser)
    args = parser.parse_args(["--force", "dataset"])

    called: Dict[str, Any] = {}

    def fake_run_dataset(dataset: Any, force: bool = False) -> None:
        called["dataset"] = dataset
        called["force"] = force

    monkeypatch.setattr(extract_metadata_command.extract_metadata,
                        "run_dataset", fake_run_dataset)

    dataset = object()
    command.run_impl(dataset, args)

    assert called == {"dataset": dataset, "force": True}
