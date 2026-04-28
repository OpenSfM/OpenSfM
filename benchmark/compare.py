# pyre-strict
"""HTML comparison report generation for benchmark runs."""

import json
import logging
import os
import time
from html import escape
from typing import Any, Dict, List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)


def _open_read(path: str, retries: int = 3, delay: float = 1.0) -> str:
    """Read a file with retries, working around NFS attribute-cache staleness.

    On NFSv3 mounts with the default attribute cache, a file just written may
    briefly appear inaccessible (Permission denied / stale handle).  We force
    a stat() to refresh the cached attributes, then retry on transient errors.
    """
    for attempt in range(retries):
        try:
            os.stat(path)            # force NFS attribute refresh
            with open(path, "r") as f:
                return f.read()
        except (PermissionError, OSError) as exc:
            if attempt < retries - 1:
                logger.warning(
                    "Retrying read of %s (attempt %d/%d): %s",
                    path, attempt + 1, retries, exc,
                )
                time.sleep(delay)
            else:
                raise
    return ""  # unreachable, keeps pyre happy


def load_run_stats(run_dir: str) -> Dict[str, Any]:
    """Load stats.json for each dataset in a run directory."""
    stats: Dict[str, Any] = {}
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(meta_path):
        return stats

    meta = json.loads(_open_read(meta_path))

    for dataset_name in meta.get("config", {}).get("datasets", []):
        stats_path = os.path.join(run_dir, dataset_name, "stats", "stats.json")
        if os.path.isfile(stats_path):
            stats[dataset_name] = json.loads(_open_read(stats_path))
        else:
            stats[dataset_name] = None
    return stats


def load_run_meta(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load run_meta.json from a run directory."""
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(meta_path):
        return None
    return json.loads(_open_read(meta_path))


def find_reference_run(
    output_dir: str,
    current_run_dir: str,
    explicit_ref: Optional[str] = None,
) -> Optional[str]:
    """Find a reference run directory for comparison.

    If explicit_ref is an existing directory path, use it directly.
    If explicit_ref is a commit hash prefix, search output_dir for a matching run.
    If explicit_ref is None, pick the most recent run (excluding current).
    """
    if explicit_ref:
        # Direct path
        if os.path.isdir(explicit_ref):
            meta_path = os.path.join(explicit_ref, "run_meta.json")
            if os.path.isfile(meta_path):
                return os.path.abspath(explicit_ref)

        # Search by commit hash prefix
        if os.path.isdir(output_dir):
            candidates = []
            for name in os.listdir(output_dir):
                run_path = os.path.join(output_dir, name)
                if not os.path.isdir(run_path):
                    continue
                meta_path = os.path.join(run_path, "run_meta.json")
                if not os.path.isfile(meta_path):
                    continue
                # Folder name starts with commit hash prefix
                if name.startswith(explicit_ref):
                    candidates.append((name, run_path))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]
        return None

    # Default: find latest run excluding current
    if not os.path.isdir(output_dir):
        return None

    current_name = os.path.basename(os.path.abspath(current_run_dir))
    candidates = []
    for name in os.listdir(output_dir):
        if name == current_name:
            continue
        run_path = os.path.join(output_dir, name)
        if not os.path.isdir(run_path):
            continue
        meta_path = os.path.join(run_path, "run_meta.json")
        if not os.path.isfile(meta_path):
            continue
        candidates.append((name, run_path))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Each metric: (label, json_path, lower_is_better)
# json_path is a dot-separated path into stats.json

MetricDef = Tuple[str, str, bool]

RECONSTRUCTION_METRICS: List[MetricDef] = [
    ("Components", "reconstruction_statistics.components", True),
    ("Reconstructed Shots", "reconstruction_statistics.reconstructed_shots_count", False),
    ("Reconstructed Points", "reconstruction_statistics.reconstructed_points_count", False),
    ("Observations", "reconstruction_statistics.observations_count", False),
]

REPROJECTION_METRICS: List[MetricDef] = [
    ("Reprojection Error (normalized)",
     "reconstruction_statistics.reprojection_error_normalized", True),
    ("Reprojection Error (pixels)",
     "reconstruction_statistics.reprojection_error_pixels", True),
    ("Reprojection Error (angular)",
     "reconstruction_statistics.reprojection_error_angular", True),
]

TRACK_METRICS: List[MetricDef] = [
    ("Avg Track Length", "reconstruction_statistics.average_track_length", False),
    ("Avg Track Length (>2)",
     "reconstruction_statistics.average_track_length_over_two", False),
]

FEATURE_METRICS: List[MetricDef] = [
    ("Detected Features (mean)", "features_statistics.detected_features.mean", False),
    ("Detected Features (median)",
     "features_statistics.detected_features.median", False),
    ("Reconstructed Features (mean)",
     "features_statistics.reconstructed_features.mean", False),
    ("Reconstructed Features (median)",
     "features_statistics.reconstructed_features.median", False),
]

GPS_METRICS: List[MetricDef] = [
    ("GPS Error (avg)", "gps_errors.average_error", True),
]

GCP_METRICS: List[MetricDef] = [
    ("GCP Error (avg)", "gcp_errors.average_error", True),
]

TIMING_METRICS: List[MetricDef] = [
    ("Feature Extraction (s)",
     "processing_statistics.steps_times.Feature Extraction", True),
    ("Features Matching (s)", "processing_statistics.steps_times.Features Matching", True),
    ("Tracks Merging (s)", "processing_statistics.steps_times.Tracks Merging", True),
    ("Reconstruction (s)", "processing_statistics.steps_times.Reconstruction", True),
    ("Total Time (s)", "processing_statistics.steps_times.Total Time", True),
]


def _get_nested(data: Optional[Dict[str, Any]], path: str) -> Any:
    """Get a value from a nested dict using dot-separated path."""
    if data is None:
        return None
    parts = path.split(".")
    current: Any = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _fmt(value: Any) -> str:
    """Format a metric value for display."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def _diff_class(
    current: Any, reference: Any, lower_is_better: bool
) -> str:
    """Return a CSS class indicating whether the current value is better or worse."""
    if current is None or reference is None:
        return ""
    try:
        c = float(current)
        r = float(reference)
    except (TypeError, ValueError):
        return ""
    if c == r:
        return ""
    if lower_is_better:
        return "better" if c < r else "worse"
    else:
        return "better" if c > r else "worse"


CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 20px; background: #fafafa; color: #333; }
h1 { font-size: 1.4em; }
h2 { font-size: 1.1em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 1.5em; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5em; font-size: 0.9em; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
th { background: #f0f0f0; text-align: center; }
td:first-child { text-align: left; font-weight: bold; }
td.label-cell { text-align: left; font-weight: normal; padding-left: 24px; color: #555; }
tr.ref-row td { background: #f8f8f8; }
tr.cur-row td { background: #fff; }
.better { color: #1a7f37; font-weight: bold; }
.worse  { color: #cf222e; font-weight: bold; }
.failed { color: #cf222e; font-weight: bold; font-style: italic; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
.dataset-header td { background: #e8e8e8; font-weight: bold; }
.dataset-header-failed td { background: #fdd; font-weight: bold; }
"""


def _render_metric_table(
    title: str,
    metrics: List[MetricDef],
    datasets: List[str],
    current_stats: Dict[str, Any],
    reference_stats: Optional[Dict[str, Any]],
    run_dir: str,
    current_meta: Optional[Dict[str, Any]] = None,
    reference_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Render one comparison table section.

    Datasets that failed (no stats) are shown with a FAILED banner.
    """
    has_ref = reference_stats is not None

    lines = [f"<h2>{escape(title)}</h2>", "<table>",
             "<tr><th>Dataset</th><th>Run</th>"]
    for label, _, _ in metrics:
        lines.append(f"<th>{escape(label)}</th>")
    lines.append("</tr>")

    for ds in datasets:
        cs = current_stats.get(ds)
        rs = reference_stats.get(ds) if reference_stats else None
        report_link = f"{ds}/stats/report.pdf"

        # Determine per-dataset success from run_meta
        cur_ds_meta = (current_meta or {}).get("datasets", {}).get(ds, {})
        cur_failed = cs is None or cur_ds_meta.get("success") is False
        ref_ds_meta = (reference_meta or {}).get("datasets", {}).get(ds, {})
        ref_failed = has_ref and (
            rs is None or ref_ds_meta.get("success") is False)

        # Failed step info
        cur_failed_step = cur_ds_meta.get("failed_step", "")
        ref_failed_step = ref_ds_meta.get("failed_step", "")

        # Dataset header
        colspan = len(metrics) + 2
        header_cls = "dataset-header-failed" if cur_failed else "dataset-header"
        ds_label = escape(ds)
        if cur_failed:
            ds_label += ' <span class="failed">FAILED'
            if cur_failed_step:
                ds_label += f" at {escape(cur_failed_step)}"
            ds_label += "</span>"
        lines.append(
            f'<tr class="{header_cls}"><td colspan="{colspan}">'
            f'<a href="{escape(report_link)}">{ds_label}</a></td></tr>'
        )

        if has_ref:
            # Reference row
            lines.append(
                '<tr class="ref-row"><td></td><td class="label-cell">reference</td>')
            if ref_failed:
                fail_text = "FAILED"
                if ref_failed_step:
                    fail_text += f" at {escape(ref_failed_step)}"
                lines.append(
                    f'<td colspan="{len(metrics)}" class="failed">{fail_text}</td>')
            else:
                for _, path, _ in metrics:
                    val = _get_nested(rs, path)
                    lines.append(f"<td>{_fmt(val)}</td>")
            lines.append("</tr>")

        # Current row
        lines.append(
            '<tr class="cur-row"><td></td><td class="label-cell">current</td>')
        if cur_failed and cs is None:
            fail_text = "FAILED"
            if cur_failed_step:
                fail_text += f" at {escape(cur_failed_step)}"
            lines.append(
                f'<td colspan="{len(metrics)}" class="failed">{fail_text}</td>')
        else:
            for _, path, lower_is_better in metrics:
                cur_val = _get_nested(cs, path)
                ref_val = _get_nested(rs, path) if rs else None
                cls = _diff_class(cur_val, ref_val, lower_is_better)
                cls_attr = f' class="{cls}"' if cls else ""
                lines.append(f"<td{cls_attr}>{_fmt(cur_val)}</td>")
        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def generate_comparison_html(
    current_stats: Dict[str, Any],
    reference_stats: Optional[Dict[str, Any]],
    current_meta: Dict[str, Any],
    reference_meta: Optional[Dict[str, Any]],
    run_dir: str,
) -> str:
    """Generate comparison.html and return the output path."""
    raw_datasets = current_meta.get("config", {}).get("datasets", [])
    # datasets may be a dict (name -> config_name) or a list
    datasets: List[str] = list(raw_datasets.keys()) if isinstance(
        raw_datasets, dict) else list(raw_datasets)

    cur_commit = current_meta.get("commit", "unknown")[:8]
    cur_date = current_meta.get("date", "unknown")
    ref_commit = reference_meta.get("commit", "unknown")[
        :8] if reference_meta else None
    ref_date = reference_meta.get(
        "date", "unknown") if reference_meta else None

    header_parts = [
        f"Commit: <strong>{escape(cur_commit)}</strong> &mdash; {escape(cur_date)}"]
    if ref_commit:
        header_parts.append(
            f"Reference: <strong>{escape(ref_commit)}</strong> &mdash; {escape(ref_date or '')}"
        )

    sections = [
        ("Reconstruction Summary", RECONSTRUCTION_METRICS),
        ("Reprojection Errors", REPROJECTION_METRICS),
        ("Track Statistics", TRACK_METRICS),
        ("Feature Statistics", FEATURE_METRICS),
        ("GPS Errors", GPS_METRICS),
        ("GCP Errors", GCP_METRICS),
        ("Processing Times", TIMING_METRICS),
    ]

    tables_html = ""
    for title, metrics in sections:
        tables_html += _render_metric_table(
            title, metrics, datasets, current_stats, reference_stats, run_dir,
            current_meta=current_meta, reference_meta=reference_meta,
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OpenSfM Benchmark — {escape(cur_commit)}</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>OpenSfM Benchmark Report</h1>
<div class="meta">
{"<br>".join(header_parts)}
</div>
{tables_html}
</body>
</html>
"""

    output_path = os.path.join(run_dir, "comparison.html")
    with open(output_path, "w") as f:
        f.write(html)

    logger.info("Comparison report written to %s", output_path)
    return output_path
