# pyre-strict
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import PIL
import numpy as np
from fpdf import FPDF
from opensfm import io
from opensfm.dataset import DataSet
from opensfm.report_locale import ReportLocale

logger: logging.Logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Design System — unified look constants
# Adapted from OpenSfM Desktop dark theme for a white-background PDF report.
# ─────────────────────────────────────────────────────────────────────────────

# Colors (RGB tuples)
COLOR_BACKGROUND: Tuple[int, int, int] = (255, 255, 255)        # White page
COLOR_PANEL: Tuple[int, int, int] = (37, 37, 38)                # #252526 — titles
COLOR_EDITOR: Tuple[int, int, int] = (30, 30, 30)               # #1E1E1E — headings
COLOR_SPLIT: Tuple[int, int, int] = (62, 62, 62)                # #3E3E3E — borders/rules
COLOR_TEXT: Tuple[int, int, int] = (44, 44, 46)                  # Near-black body text
COLOR_TEXT_SECONDARY: Tuple[int, int, int] = (108, 117, 125)    # Muted secondary
COLOR_ACCENT: Tuple[int, int, int] = (0, 184, 87)               # Vibrant green (tuned for white bg)
COLOR_ACCENT_FAINT: Tuple[int, int, int] = (232, 250, 240)      # Very subtle green tint
COLOR_TABLE_HEADER: Tuple[int, int, int] = (42, 45, 52)         # Slate-dark header
COLOR_TABLE_HEADER_TEXT: Tuple[int, int, int] = (255, 255, 255)  # White on dark
COLOR_TABLE_ROW_EVEN: Tuple[int, int, int] = (250, 251, 253)    # Almost-white zebra
COLOR_TABLE_ROW_ODD: Tuple[int, int, int] = (255, 255, 255)     # Pure white
COLOR_TABLE_LABEL: Tuple[int, int, int] = (242, 244, 247)       # Neutral light grey for row labels
COLOR_TABLE_BORDER: Tuple[int, int, int] = (228, 231, 236)      # Subtle border
COLOR_FOOTER: Tuple[int, int, int] = (51, 51, 51)               # #333333

# Quality grading colors
COLOR_GRADE_GOOD: Tuple[int, int, int] = (0, 184, 87)           # Green (matches accent)
COLOR_GRADE_AVG: Tuple[int, int, int] = (245, 158, 11)          # Warm amber
COLOR_GRADE_BAD: Tuple[int, int, int] = (220, 53, 69)           # Bootstrap-red

# Typography sizes (pt)
FONT_TITLE: int = 22
FONT_H1: int = 14
FONT_H2: int = 12
FONT_H3: int = 10
FONT_BODY: int = 9
FONT_SMALL: int = 8

# Spacing (mm)
MARGIN: int = 12
CELL_HEIGHT: float = 7.0
SECTION_GAP: float = 8.0
TABLE_GAP: float = 4.0
CONTENT_WIDTH: int = 186   # A4 (210) - 2*MARGIN

# Logo
LOGO_PATH: str = os.path.join(os.path.dirname(__file__), "..", "doc", "images", "logo.png")


def _quality_indicator(value: float, thresholds: Tuple[float, float, float]) -> str:
    """Return a colored circle indicator based on thresholds (bad, avg, good).

    thresholds = (bad_below, avg_below, good_above_or_equal)
    Values >= thresholds[2] are good, >= thresholds[1] average, else bad.
    """
    bad_t, avg_t, good_t = thresholds
    if value >= good_t:
        return "\u25cf"  # ● good (will be colored green)
    elif value >= avg_t:
        return "\u25cf"  # ● average (will be colored amber)
    else:
        return "\u25cf"  # ● bad (will be colored red)


def _quality_color(value: float, thresholds: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Return the RGB color for a quality grade."""
    _bad_t, avg_t, good_t = thresholds
    if value >= good_t:
        return COLOR_GRADE_GOOD
    elif value >= avg_t:
        return COLOR_GRADE_AVG
    else:
        return COLOR_GRADE_BAD


def _quality_color_lower_is_better(
    value: float, thresholds: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """Return the RGB color when lower values are better.

    thresholds = (good_below, avg_below, bad_at_or_above)
    """
    good_t, avg_t, bad_t = thresholds
    if value <= good_t:
        return COLOR_GRADE_GOOD
    elif value <= avg_t:
        return COLOR_GRADE_AVG
    else:
        return COLOR_GRADE_BAD


class Report:
    def __init__(self, data: DataSet) -> None:
        self.output_path: str = os.path.join(data.data_path, "stats")
        self.dataset_name: str = os.path.basename(data.data_path)
        self.io_handler: io.IoFilesystemBase = data.io_handler
        self.data_path: str = data.data_path

        self.locale: ReportLocale = ReportLocale(
            language=data.config.get("report_language", "en"),
            unit_system=data.config.get("report_unit_system", "metric"),
        )

        self.pdf = FPDF("P", "mm", "A4")
        self.pdf.set_auto_page_break(auto=True, margin=MARGIN)
        self.pdf.add_page()

        self.stats: Dict[str, Any] = data.load_stats()

    def save_report(self, filename: str) -> None:
        bytestring = self.pdf.output()
        if isinstance(bytestring, str):
            bytestring = bytestring.encode("utf8")

        with self.io_handler.open_wb(os.path.join(self.output_path, filename)) as fwb:
            fwb.write(bytestring)

    # ─────────────────────────────────────────────────────────────────────────
    # Primitive drawing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_accent_rule(self) -> None:
        """Draw a thin green accent line under the current position."""
        y = self.pdf.get_y()
        self.pdf.set_draw_color(*COLOR_ACCENT)
        self.pdf.set_line_width(0.8)
        self.pdf.line(MARGIN, y, MARGIN + CONTENT_WIDTH, y)
        self.pdf.set_xy(MARGIN, y + 2)

    def _make_table(
        self,
        columns_names: Optional[List[str]],
        rows: List[List[str]],
        row_header: bool = False,
    ) -> None:
        self.pdf.set_font("Helvetica", "", FONT_BODY)
        self.pdf.set_line_width(0.15)

        n_cols = len(rows[0]) if rows else 0
        if n_cols == 0:
            return
        columns_sizes = [int(CONTENT_WIDTH / n_cols)] * n_cols
        # Give remainder to last column
        columns_sizes[-1] = CONTENT_WIDTH - sum(columns_sizes[:-1])

        if columns_names:
            self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
            self.pdf.set_fill_color(*COLOR_TABLE_HEADER)
            self.pdf.set_text_color(*COLOR_TABLE_HEADER_TEXT)
            self.pdf.set_font("Helvetica", "B", FONT_BODY)
            for col, size in zip(columns_names, columns_sizes):
                self.pdf.rect(
                    self.pdf.get_x(), self.pdf.get_y(),
                    size, CELL_HEIGHT, style="FD",
                )
                self.pdf.cell(size, CELL_HEIGHT, "  " + col, align="L")
            self.pdf.set_xy(MARGIN, self.pdf.get_y() + CELL_HEIGHT)

        self.pdf.set_font("Helvetica", "", FONT_BODY)
        for row_idx, row in enumerate(rows):
            is_even = row_idx % 2 == 0
            row_bg = COLOR_TABLE_ROW_EVEN if is_even else COLOR_TABLE_ROW_ODD
            for i, (col, size) in enumerate(zip(row, columns_sizes)):
                if i == 0 and row_header:
                    self.pdf.set_fill_color(*COLOR_TABLE_LABEL)
                    self.pdf.set_text_color(*COLOR_PANEL)
                    self.pdf.set_font("Helvetica", "B", FONT_BODY)
                else:
                    self.pdf.set_fill_color(*row_bg)
                    self.pdf.set_text_color(*COLOR_TEXT)
                    self.pdf.set_font("Helvetica", "", FONT_BODY)
                self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
                self.pdf.rect(
                    self.pdf.get_x(), self.pdf.get_y(),
                    size, CELL_HEIGHT, style="FD",
                )
                self.pdf.cell(size, CELL_HEIGHT, "  " + col, align="L")
            self.pdf.set_xy(MARGIN, self.pdf.get_y() + CELL_HEIGHT)

    def _make_section(self, title: str) -> None:
        self.pdf.set_font("Helvetica", "B", FONT_H1)
        self.pdf.set_text_color(*COLOR_PANEL)
        self.pdf.cell(0, SECTION_GAP, title, align="L")
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + SECTION_GAP + 1)
        self._draw_accent_rule()
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def _make_subsection(self, title: str) -> None:
        self.pdf.set_font("Helvetica", "B", FONT_H2)
        self.pdf.set_text_color(*COLOR_SPLIT)
        self.pdf.cell(0, SECTION_GAP, title, align="L")
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + SECTION_GAP + 2)

    def _make_centered_image(self, image_path: str, desired_height: float) -> None:
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            local_image_path = os.path.join(
                tmp_local_dir, os.path.basename(image_path))
            with self.io_handler.open_wb(local_image_path) as fwb:
                with self.io_handler.open_rb(image_path) as f:
                    fwb.write(f.read())

            width, height = PIL.Image.open(local_image_path).size
            resized_width = width * desired_height / height
            if resized_width > CONTENT_WIDTH:
                resized_width = CONTENT_WIDTH
                desired_height = height * resized_width / width

            self.pdf.image(
                local_image_path,
                self.pdf.get_x() + CONTENT_WIDTH / 2 - resized_width / 2,
                self.pdf.get_y(),
                h=desired_height,
            )
            self.pdf.set_xy(
                MARGIN, self.pdf.get_y() + desired_height + TABLE_GAP
            )

    def _quality_cell(self, value: float, text: str, thresholds: Tuple[float, float, float]) -> str:
        """Return text prefixed with a quality indicator dot."""
        indicator = _quality_indicator(value, thresholds)
        return f"{indicator} {text}"

    def _make_graded_row(
        self, label: str, value: float, text: str,
        thresholds: Tuple[float, float, float], columns_sizes: List[int]
    ) -> None:
        """Render a table row with a colored quality dot."""
        color = _quality_color(value, thresholds)

        # Label cell (neutral light grey, matching unified table style)
        self.pdf.set_fill_color(*COLOR_TABLE_LABEL)
        self.pdf.set_text_color(*COLOR_PANEL)
        self.pdf.set_font("Helvetica", "B", FONT_BODY)
        self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
        self.pdf.set_line_width(0.15)
        self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), columns_sizes[0], CELL_HEIGHT, style="FD")
        self.pdf.cell(columns_sizes[0], CELL_HEIGHT, "  " + label, align="L")

        # Value cell with colored indicator
        self.pdf.set_fill_color(*COLOR_TABLE_ROW_EVEN)
        self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
        self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), columns_sizes[1], CELL_HEIGHT, style="FD")

        # Draw the dot in grade color
        dot_x = self.pdf.get_x() + 4
        dot_y = self.pdf.get_y() + CELL_HEIGHT / 2
        self.pdf.set_fill_color(*color)
        self.pdf.ellipse(dot_x - 1.4, dot_y - 1.4, 2.8, 2.8, style="F")

        # Draw the text
        self.pdf.set_text_color(*COLOR_TEXT)
        self.pdf.set_font("Helvetica", "", FONT_BODY)
        self.pdf.cell(columns_sizes[1], CELL_HEIGHT, "       " + text, align="L")

        self.pdf.set_xy(MARGIN, self.pdf.get_y() + CELL_HEIGHT)

    def _make_graded_row_lower(
        self, label: str, value: float, text: str,
        thresholds: Tuple[float, float, float], columns_sizes: List[int]
    ) -> None:
        """Render a table row with a colored quality dot (lower is better)."""
        color = _quality_color_lower_is_better(value, thresholds)

        # Label cell
        self.pdf.set_fill_color(*COLOR_TABLE_LABEL)
        self.pdf.set_text_color(*COLOR_PANEL)
        self.pdf.set_font("Helvetica", "B", FONT_BODY)
        self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
        self.pdf.set_line_width(0.15)
        self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), columns_sizes[0], CELL_HEIGHT, style="FD")
        self.pdf.cell(columns_sizes[0], CELL_HEIGHT, "  " + label, align="L")

        # Value cell with colored indicator
        self.pdf.set_fill_color(*COLOR_TABLE_ROW_EVEN)
        self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
        self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), columns_sizes[1], CELL_HEIGHT, style="FD")

        # Draw the dot in grade color
        dot_x = self.pdf.get_x() + 4
        dot_y = self.pdf.get_y() + CELL_HEIGHT / 2
        self.pdf.set_fill_color(*color)
        self.pdf.ellipse(dot_x - 1.4, dot_y - 1.4, 2.8, 2.8, style="F")

        # Draw the text
        self.pdf.set_text_color(*COLOR_TEXT)
        self.pdf.set_font("Helvetica", "", FONT_BODY)
        self.pdf.cell(columns_sizes[1], CELL_HEIGHT, "       " + text, align="L")

        self.pdf.set_xy(MARGIN, self.pdf.get_y() + CELL_HEIGHT)

    def _draw_graded_cell(
        self, text: str, size: int, value: float,
        thresholds: Tuple[float, float, float], row_bg: Tuple[int, int, int]
    ) -> None:
        """Draw a single table cell with a colored quality dot on the left."""
        color = _quality_color(value, thresholds)
        self.pdf.set_fill_color(*row_bg)
        self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
        self.pdf.set_line_width(0.15)
        self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), size, CELL_HEIGHT, style="FD")
        dot_x = self.pdf.get_x() + 4
        dot_y = self.pdf.get_y() + CELL_HEIGHT / 2
        self.pdf.set_fill_color(*color)
        self.pdf.ellipse(dot_x - 1.4, dot_y - 1.4, 2.8, 2.8, style="F")
        self.pdf.set_text_color(*COLOR_TEXT)
        self.pdf.set_font("Helvetica", "", FONT_BODY)
        self.pdf.cell(size, CELL_HEIGHT, "       " + text, align="L")

    def _draw_graded_cell_lower(
        self, text: str, size: int, value: float,
        thresholds: Tuple[float, float, float], row_bg: Tuple[int, int, int]
    ) -> None:
        """Draw a cell with quality dot where lower values are better."""
        color = _quality_color_lower_is_better(value, thresholds)
        self.pdf.set_fill_color(*row_bg)
        self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
        self.pdf.set_line_width(0.15)
        self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), size, CELL_HEIGHT, style="FD")
        dot_x = self.pdf.get_x() + 4
        dot_y = self.pdf.get_y() + CELL_HEIGHT / 2
        self.pdf.set_fill_color(*color)
        self.pdf.ellipse(dot_x - 1.4, dot_y - 1.4, 2.8, 2.8, style="F")
        self.pdf.set_text_color(*COLOR_TEXT)
        self.pdf.set_font("Helvetica", "", FONT_BODY)
        self.pdf.cell(size, CELL_HEIGHT, "       " + text, align="L")

    def make_title(self) -> None:
        # Logo
        logo_path = os.path.normpath(LOGO_PATH)
        if os.path.exists(logo_path):
            logo_height = 14
            width, height = PIL.Image.open(logo_path).size
            logo_width = width * logo_height / height
            self.pdf.image(logo_path, MARGIN, MARGIN, h=logo_height)
            title_x = MARGIN + logo_width + 4
        else:
            title_x = MARGIN

        # Title
        self.pdf.set_xy(title_x, MARGIN)
        self.pdf.set_font("Helvetica", "B", FONT_TITLE)
        self.pdf.set_text_color(*COLOR_ACCENT)
        self.pdf.cell(0, 10, "OpenSfM", align="L")

        self.pdf.set_xy(title_x, MARGIN + 9)
        self.pdf.set_font("Helvetica", "", FONT_H1)
        self.pdf.set_text_color(*COLOR_PANEL)
        self.pdf.cell(0, 8, self.locale.t("quality_report"), align="L")

        # Version number (right aligned)
        try:
            out, _ = subprocess.Popen(
                ["git", "describe", "--tags"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()
            version = out.strip().decode()
        except BaseException as e:
            logger.warning(
                f"Exception thrown while extracting 'git' version, {e}")
            version = ""

        version = "unknown" if version == "" else version

        self.pdf.set_xy(MARGIN, MARGIN)
        self.pdf.set_font("Helvetica", "", FONT_SMALL)
        self.pdf.set_text_color(*COLOR_TEXT_SECONDARY)
        self.pdf.cell(
            CONTENT_WIDTH, 6, f"v{version}", align="R"
        )
        self.pdf.set_xy(MARGIN, MARGIN + 22)

    def make_dataset_summary(self) -> None:
        self._make_section(self.locale.t("dataset_summary"))

        rows = [
            [self.locale.t("dataset"), self.dataset_name],
            [self.locale.t("date"), self.stats["processing_statistics"]["date"]],
            [
                self.locale.t("area_covered"),
                self.locale.format_area(self.stats["processing_statistics"]["area"]),
            ],
            [
                self.locale.t("processing_time"),
                self.locale.format_time(self.stats["processing_statistics"]["steps_times"]["Total Time"]),
            ],
        ]
        self._make_table(None, rows, True)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + SECTION_GAP)

    def _has_meaningful_gcp(self) -> bool:
        return (
            self.stats["reconstruction_statistics"]["has_gcp"]
            and "average_error" in self.stats["gcp_errors"]
        )

    def make_processing_summary(self) -> None:
        self._make_section(self.locale.t("processing_summary"))

        rec_shots, init_shots = (
            self.stats["reconstruction_statistics"]["reconstructed_shots_count"],
            self.stats["reconstruction_statistics"]["initial_shots_count"],
        )
        rec_points, init_points = (
            self.stats["reconstruction_statistics"]["reconstructed_points_count"],
            self.stats["reconstruction_statistics"]["initial_points_count"],
        )

        geo_string = []
        if self.stats["reconstruction_statistics"]["has_gps"]:
            geo_string.append(self.locale.t("gps"))
        if self._has_meaningful_gcp():
            geo_string.append(self.locale.t("gcp"))

        ratio_shots = rec_shots / init_shots * 100 if init_shots > 0 else -1
        ratio_points = rec_points / init_points * 100 if init_points > 0 else -1
        avg_track = self.stats["reconstruction_statistics"]["average_track_length"]

        # Quality thresholds: (bad_below, avg_below, good_at_or_above)
        shots_thresholds = (75.0, 90.0, 100.0)
        points_thresholds = (50.0, 65.0, 75.0)
        track_thresholds = (2.5, 2.75, 3.0)
        calibration_thresholds = (70.0, 85.0, 95.0)

        col_sizes = [int(CONTENT_WIDTH * 0.42), CONTENT_WIDTH - int(CONTENT_WIDTH * 0.42)]

        # Graded rows
        self._make_graded_row(
            self.locale.t("reconstructed_images"),
            ratio_shots,
            f"{rec_shots} / {init_shots} {self.locale.t('shots')} ({ratio_shots:.1f}%)",
            shots_thresholds, col_sizes,
        )
        self._make_graded_row(
            self.locale.t("reconstructed_points"),
            ratio_points,
            f"{rec_points} / {init_points} ({ratio_points:.1f}%)",
            points_thresholds, col_sizes,
        )
        self._make_graded_row(
            self.locale.t("average_track_length"),
            avg_track,
            f"{avg_track:.2f} {self.locale.t('unit_images')}",
            track_thresholds, col_sizes,
        )

        # Camera calibration deviation (average focal relative difference, inverted to similarity %)
        camera_errors = self.stats.get("camera_errors", {})
        focal_diffs = []
        for params in camera_errors.values():
            rel_diff = params.get("relative_difference", {})
            if "focal" in rel_diff:
                focal_diffs.append(rel_diff["focal"])
        if focal_diffs:
            avg_focal_diff = float(np.mean(focal_diffs))
            calibration_similarity = max(0.0, 100.0 - avg_focal_diff)
            self._make_graded_row(
                self.locale.t("camera_calibration_deviation"),
                calibration_similarity,
                f"{avg_focal_diff:.2f}% {self.locale.t('avg_focal_change')} ({calibration_similarity:.1f}% {self.locale.t('similarity')})",
                calibration_thresholds, col_sizes,
            )

        # Overlap graded rows (thresholds: bad < 50%, avg 50-70%, good >= 70%)
        overlap = self.stats.get("overlap", {})
        overlap_thresholds = (50.0, 60.0, 70.0)
        front_mean = overlap.get("front_overlap_mean", 0.0)
        side_mean = overlap.get("side_overlap_mean", 0.0)
        if front_mean > 0 or side_mean > 0:
            self._make_graded_row(
                self.locale.t("front_overlap"),
                front_mean,
                f"{front_mean:.1f}%",
                overlap_thresholds, col_sizes,
            )
            self._make_graded_row(
                self.locale.t("side_overlap"),
                side_mean,
                f"{side_mean:.1f}%",
                overlap_thresholds, col_sizes,
            )

        # Non-graded rows
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

        gsd = self.stats["reconstruction_statistics"].get("gsd", -1.0)
        rows = [
            [
                self.locale.t("ground_sampling_distance"),
                self.locale.format_gsd(gsd) if gsd > 0 else "N/A",
            ],
            [
                self.locale.t("reconstructed_components"),
                f"{self.stats['reconstruction_statistics']['components']}",
            ],
            [self.locale.t("geographic_referencing"), " + ".join(geo_string) if geo_string else self.locale.t("none")],
        ]

        gcp_crs = self.stats.get("gcp_errors", {}).get("coordinate_system")
        if gcp_crs:
            rows.append([self.locale.t("gcp_coordinate_system"), gcp_crs])

        self._make_table(None, rows, True)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)


        # GPS error graded row (based on multiples of input std dev)
        if self.stats["reconstruction_statistics"]["has_gps"] and "average_error" in self.stats.get("gps_errors", {}):
            gps_avg_error = self.stats["gps_errors"]["average_error"]
            avg_gps_std = self.stats["gps_errors"].get("average_gps_std")
            if avg_gps_std:
                # Use the average of XYZ std as reference
                ref_std = (avg_gps_std["x"] + avg_gps_std["y"] + avg_gps_std["z"]) / 3.0
                if ref_std > 1e-9:
                    # Thresholds: good <= 2*std, avg <= 3*std, bad > 4*std (lower is better)
                    gps_thresholds = (2.0 * ref_std, 3.0 * ref_std, 4.0 * ref_std)
                    self._make_graded_row_lower(
                        self.locale.t("gps_error"),
                        gps_avg_error,
                        self.locale.format_distance_label(gps_avg_error),
                        gps_thresholds, col_sizes,
                    )

        # GCP error graded row (based on GSD, same Z thresholds: good <= 3*GSD, avg <= 4*GSD, bad > 5*GSD)
        if self._has_meaningful_gcp() and gsd > 0:
            gcp_avg_error = self.stats["gcp_errors"]["average_error"]
            gcp_thresholds = (3.0 * gsd, 4.0 * gsd, 5.0 * gsd)
            self._make_graded_row_lower(
                self.locale.t("gcp_error"),
                gcp_avg_error,
                self.locale.format_distance_label(gcp_avg_error, precision=3),
                gcp_thresholds, col_sizes,
            )

        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

        # Top-view image
        topview_height = 120
        topview_grids = [
            f for f in self.io_handler.ls(self.output_path) if f.startswith("topview")
        ]
        if topview_grids:
            self._make_centered_image(
                os.path.join(self.output_path, topview_grids[0]), topview_height
            )

        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def make_processing_time_details(self) -> None:
        self._make_section(self.locale.t("processing_time_details"))

        columns_names = list(
            self.stats["processing_statistics"]["steps_times"].keys())
        formatted_floats = []
        for v in self.stats["processing_statistics"]["steps_times"].values():
            formatted_floats.append(self.locale.format_time(v))
        rows = [formatted_floats]
        self._make_table(columns_names, rows)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + SECTION_GAP)

    def make_gps_details(self) -> None:
        self._make_section(self.locale.t("gps_gcp_errors_details"))

        dist_unit = self.locale.distance_unit_label()

        for error_type in ["gps", "gcp"]:
            rows = []
            columns_names = [error_type.upper(), self.locale.t("mean"), self.locale.t("sigma"), self.locale.t("rms_error")]
            if "average_error" not in self.stats[error_type + "_errors"]:
                continue
            for comp in ["x", "y", "z"]:
                row = [self.locale.t(f"{comp}_error_distance").format(unit=dist_unit)]
                row.append(
                    self.locale.format_distance(self.stats[error_type + '_errors']['mean'][comp], precision=3))
                row.append(
                    self.locale.format_distance(self.stats[error_type + '_errors']['std'][comp], precision=3))
                row.append(
                    self.locale.format_distance(self.stats[error_type + '_errors']['error'][comp], precision=3))
                rows.append(row)

            rows.append(
                [
                    self.locale.t("total"),
                    "",
                    "",
                    self.locale.format_distance(self.stats[error_type + '_errors']['average_error'], precision=3),
                ]
            )

            # For GPS, append a priori std dev row
            if error_type == "gps":
                avg_gps_std = self.stats["gps_errors"].get("average_gps_std")
                if avg_gps_std:
                    rows.append(
                        [
                            self.locale.t("input_std_dev_distance").format(unit=dist_unit),
                            self.locale.format_distance(avg_gps_std['x'], precision=3),
                            self.locale.format_distance(avg_gps_std['y'], precision=3),
                            self.locale.format_distance(avg_gps_std['z'], precision=3),
                        ]
                    )

            self._make_table(columns_names, rows)
            self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

        rows = []
        columns_names = [
            self.locale.t("gps_bias"),
            self.locale.t("scale"),
            self.locale.t("translation"),
            self.locale.t("rotation"),
        ]
        for camera, params in self.stats["camera_errors"].items():
            bias = params["bias"]
            s, t, R = bias["scale"], bias["translation"], bias["rotation"]
            rows.append(
                [
                    camera,
                    f"{s:.2f}",
                    f"{t[0]:.2f}   {t[1]:.2f}   {t[2]:.2f}",
                    f"{R[0]:.2f}   {R[1]:.2f}   {R[2]:.2f}",
                ]
            )
        if rows:
            self._make_table(columns_names, rows)

        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def make_gcp_details(self) -> None:
        gcp_errors = self.stats.get("gcp_errors", {})
        details = gcp_errors.get("details", [])
        if not details:
            return

        self._make_section(self.locale.t("gcp_details"))

        # GSD-based quality thresholds for error cells
        gsd = self.stats["reconstruction_statistics"].get("gsd", -1.0)
        # X/Y: good <= 1*GSD, avg <= 2*GSD, bad > 3*GSD
        # Z:   good <= 3*GSD, avg <= 4*GSD, bad > 5*GSD
        xy_thresholds = (1.0 * gsd, 2.0 * gsd, 3.0 * gsd) if gsd > 0 else None
        z_thresholds = (3.0 * gsd, 4.0 * gsd, 5.0 * gsd) if gsd > 0 else None

        # Inlier ratio quality thresholds (as percentages: bad < 90, avg 90-95, good >= 95)
        inlier_thresholds = (90.0, 95.0, 100.0)

        dist_short = self.locale.distance_unit_short()
        columns_names = [
            self.locale.t("gcp_id"),
            self.locale.t("x_error_short").format(unit=dist_short),
            self.locale.t("y_error_short").format(unit=dist_short),
            self.locale.t("z_error_short").format(unit=dist_short),
            self.locale.t("inliers_total"),
        ]
        n_cols = len(columns_names)
        col_sizes = [int(CONTENT_WIDTH / n_cols)] * n_cols
        col_sizes[-1] = CONTENT_WIDTH - sum(col_sizes[:-1])

        # Header row
        self.pdf.set_font("Helvetica", "B", FONT_BODY)
        self.pdf.set_fill_color(*COLOR_TABLE_HEADER)
        self.pdf.set_text_color(*COLOR_TABLE_HEADER_TEXT)
        self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
        self.pdf.set_line_width(0.15)
        for col, size in zip(columns_names, col_sizes):
            self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), size, CELL_HEIGHT, style="FD")
            self.pdf.cell(size, CELL_HEIGHT, "  " + col, align="L")
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + CELL_HEIGHT)

        # Data rows
        for row_idx, entry in enumerate(details):
            gcp_id = entry["id"]
            error = entry["error"]
            n_inliers = entry["n_inliers"]
            n_total = entry["n_total"]
            row_bg = COLOR_TABLE_ROW_EVEN if row_idx % 2 == 0 else COLOR_TABLE_ROW_ODD

            # ID cell (label style)
            self.pdf.set_fill_color(*COLOR_TABLE_LABEL)
            self.pdf.set_text_color(*COLOR_PANEL)
            self.pdf.set_font("Helvetica", "B", FONT_BODY)
            self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
            self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), col_sizes[0], CELL_HEIGHT, style="FD")
            self.pdf.cell(col_sizes[0], CELL_HEIGHT, "  " + gcp_id, align="L")

            # X, Y, Z error cells with GSD-based quality dots
            for col_idx, axis in enumerate(["x", "y", "z"], start=1):
                cell_text = self.locale.format_distance(error[axis], precision=3) if error is not None else "N/A"
                thresholds = z_thresholds if axis == "z" else xy_thresholds
                if error is not None and thresholds is not None:
                    abs_err = abs(error[axis])
                    self._draw_graded_cell_lower(
                        cell_text, col_sizes[col_idx], abs_err, thresholds, row_bg
                    )
                else:
                    self.pdf.set_fill_color(*row_bg)
                    self.pdf.set_text_color(*COLOR_TEXT)
                    self.pdf.set_font("Helvetica", "", FONT_BODY)
                    self.pdf.set_draw_color(*COLOR_TABLE_BORDER)
                    self.pdf.rect(self.pdf.get_x(), self.pdf.get_y(), col_sizes[col_idx], CELL_HEIGHT, style="FD")
                    self.pdf.cell(col_sizes[col_idx], CELL_HEIGHT, "  " + cell_text, align="L")

            # Inliers/Total cell with quality dot
            inlier_pct = (n_inliers / n_total * 100.0) if n_total > 0 else 0.0
            inlier_text = f"{n_inliers} / {n_total}"
            self._draw_graded_cell(inlier_text, col_sizes[4], inlier_pct, inlier_thresholds, row_bg)

            self.pdf.set_xy(MARGIN, self.pdf.get_y() + CELL_HEIGHT)

        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def make_orientation_details(self) -> None:
        if "opk_errors" not in self.stats:
            return
        if "average_error" not in self.stats["opk_errors"]:
            return

        self._make_section(self.locale.t("orientation_details"))
        columns_names = [self.locale.t("component"), self.locale.t("mean"), self.locale.t("sigma"), self.locale.t("rms_error")]
        error_name = "opk_errors"

        rows = []
        for comp in ["omega", "phi", "kappa"]:
            row = [self.locale.t(f"{comp}_error")]
            row.append(
                f"{self.stats[error_name]['mean'][comp]:.3f}")
            row.append(
                f"{self.stats[error_name]['std'][comp]:.3f}")
            row.append(
                f"{self.stats[error_name]['error'][comp]:.3f}")
            rows.append(row)

        rows.append(
            [
                self.locale.t("total"),
                "",
                "",
                f"{self.stats[error_name]['average_error']:.3f}",
            ]
        )
        self._make_table(columns_names, rows)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def make_features_details(self) -> None:
        self._make_section(self.locale.t("features_details"))

        heatmap_height = 60
        heatmaps = [
            f for f in self.io_handler.ls(self.output_path) if f.startswith("heatmap") and f.endswith(".png")
        ]
        if heatmaps:
            self._make_centered_image(
                os.path.join(self.output_path, heatmaps[0]), heatmap_height
            )
            if len(heatmaps) > 1:
                logger.warning("Please implement multi-model display")

        columns_names = ["", self.locale.t("min"), self.locale.t("max"), self.locale.t("mean"), self.locale.t("median")]
        rows = []
        for comp in ["detected_features", "reconstructed_features"]:
            label_key = "detected" if "detected" in comp else "reconstructed"
            row = [self.locale.t(label_key)]
            for t in ["min", "max", "mean", "median"]:
                row.append(
                    f"{self.stats['features_statistics'][comp][t]:.0f}"
                )
            rows.append(row)
        self._make_table(columns_names, rows)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + SECTION_GAP)

    def make_reconstruction_details(self) -> None:
        self._make_section(self.locale.t("reconstruction_details"))

        rows = [
            [
                self.locale.t("reprojection_error"),
                (
                    f"{self.stats['reconstruction_statistics']['reprojection_error_normalized']:.2f} / "
                    f"{self.stats['reconstruction_statistics']['reprojection_error_pixels']:.2f} / "
                    f"{self.stats['reconstruction_statistics']['reprojection_error_angular']:.5f}"
                ),
            ],
            [
                self.locale.t("average_track_length"),
                f"{self.stats['reconstruction_statistics']['average_track_length']:.2f} {self.locale.t('unit_images')}",
            ],
            [
                self.locale.t("average_track_length_gt2"),
                f"{self.stats['reconstruction_statistics']['average_track_length_over_two']:.2f} {self.locale.t('unit_images')}",
            ],
        ]
        self._make_table(None, rows, True)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

        residual_histogram_height = 60
        residual_histogram = [
            f
            for f in self.io_handler.ls(self.output_path)
            if f.startswith("residual_histogram")
        ]
        if residual_histogram:
            self._make_centered_image(
                os.path.join(self.output_path, residual_histogram[0]),
                residual_histogram_height,
            )
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def make_camera_models_details(self) -> None:
        self._make_section(self.locale.t("camera_models_details"))

        for camera, params in self.stats["camera_errors"].items():
            string_id = "residuals_" + str(camera.replace("/", "_"))
            residual_grids = [
                f
                for f in self.io_handler.ls(self.output_path)
                if f.startswith(string_id) and f.endswith(".png")
            ]
            if not residual_grids:
                continue

            initial = params["initial_values"]
            optimized = params["optimized_values"]
            rel_diff = params.get("relative_difference", {})
            names = [""] + list(initial.keys())

            rows = []
            rows.append([self.locale.t("initial")] + [f"{x:.4f}" for x in initial.values()])
            rows.append([self.locale.t("optimized")] +
                        [f"{x:.4f}" for x in optimized.values()])
            rows.append([self.locale.t("rel_diff_pct")] +
                        [f"{rel_diff.get(k, 0.0):.2f}" for k in initial.keys()])

            self._make_subsection(camera)
            self._make_table(names, rows)
            self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

            residual_grid_height = 100
            self._make_centered_image(
                os.path.join(self.output_path,
                             residual_grids[0]), residual_grid_height
            )

    def make_rig_cameras_details(self) -> None:
        if len(self.stats["rig_errors"]) == 0:
            return

        self._make_section(self.locale.t("rig_cameras_details"))

        columns_names = [
            self.locale.t("translation_x"),
            self.locale.t("translation_y"),
            self.locale.t("translation_z"),
            self.locale.t("rotation_x"),
            self.locale.t("rotation_y"),
            self.locale.t("rotation_z"),
        ]
        for rig_camera_id, params in self.stats["rig_errors"].items():
            initial = params["initial_values"]
            optimized = params["optimized_values"]

            rows = []
            r_init, t_init = initial["rotation"], initial["translation"]
            r_opt, t_opt = optimized["rotation"], optimized["translation"]
            rows.append(
                [
                    self.locale.format_distance(t_init[0], precision=4),
                    self.locale.format_distance(t_init[1], precision=4),
                    self.locale.format_distance(t_init[2], precision=4),
                    f"{r_init[0]:.4f}",
                    f"{r_init[1]:.4f}",
                    f"{r_init[2]:.4f}",
                ]
            )
            rows.append(
                [
                    self.locale.format_distance(t_opt[0], precision=4),
                    self.locale.format_distance(t_opt[1], precision=4),
                    self.locale.format_distance(t_opt[2], precision=4),
                    f"{r_opt[0]:.4f}",
                    f"{r_opt[1]:.4f}",
                    f"{r_opt[2]:.4f}",
                ]
            )

            self._make_subsection(rig_camera_id)
            self._make_table(columns_names, rows)
            self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def make_tracks_details(self) -> None:
        self._make_section(self.locale.t("tracks_details"))
        matchgraph_height = 80
        matchgraph = [
            f
            for f in self.io_handler.ls(self.output_path)
            if f.startswith("matchgraph") and f.endswith(".png")
        ]
        if matchgraph:
            self._make_centered_image(
                os.path.join(self.output_path, matchgraph[0]), matchgraph_height
            )

        histogram = self.stats["reconstruction_statistics"]["histogram_track_length"]
        start_length, end_length = 2, 10
        row_length = [self.locale.t("length")]
        for length, _ in sorted(histogram.items(), key=lambda x: int(x[0])):
            if int(length) < start_length or int(length) > end_length:
                continue
            row_length.append(length)
        row_count = [self.locale.t("count")]
        for length, count in sorted(histogram.items(), key=lambda x: int(x[0])):
            if int(length) < start_length or int(length) > end_length:
                continue
            row_count.append(f"{count}")

        self._make_table(None, [row_length, row_count], True)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + SECTION_GAP)

    def make_overlap_summary(self) -> None:
        overlap = self.stats.get("overlap", {})
        if not overlap:
            return

        self._make_section(self.locale.t("overlap_summary"))

        # Display stats table
        front_mean = overlap.get("front_overlap_mean", 0.0)
        side_mean = overlap.get("side_overlap_mean", 0.0)

        rows = [
            [self.locale.t("front_overlap_mean"), f"{front_mean:.1f}%"],
            [self.locale.t("side_overlap_mean"), f"{side_mean:.1f}%"],
        ]
        self._make_table(None, rows, True)
        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

        # Display overlap map image
        overlap_maps = [
            f for f in self.io_handler.ls(self.output_path)
            if f.startswith("overlap_map") and f.endswith(".png")
        ]
        if overlap_maps:
            self._make_centered_image(
                os.path.join(self.output_path, overlap_maps[0]), 120
            )

        self.pdf.set_xy(MARGIN, self.pdf.get_y() + TABLE_GAP)

    def add_page_break(self) -> None:
        self.pdf.add_page("P")

    def generate_report(self) -> None:
        self.make_title()
        self.make_dataset_summary()
        self.make_processing_summary()
        self.add_page_break()

        self.make_features_details()
        self.make_reconstruction_details()
        self.add_page_break()

        self.make_tracks_details()
        self.make_camera_models_details()
        self.make_rig_cameras_details()
        self.add_page_break()

        self.make_overlap_summary()
        self.make_gps_details()
        self.make_gcp_details()
        self.make_orientation_details()
        self.make_processing_time_details()
