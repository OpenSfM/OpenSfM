# pyre-strict
from opensfm import io, preview
from opensfm.dataset import DataSet


def run_dataset(data: DataSet) -> None:
    """Run the full preview pipeline (metadata + features + matching + SfM)."""
    report = preview.run_preview_dataset(data)
    data.save_report(io.json_dumps(report), "preview.json")
