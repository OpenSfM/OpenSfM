# pyre-strict
"""Localization support for the OpenSfM quality report.

Provides unit formatting (metric/imperial) and string translation
driven by YAML catalogs in opensfm/data/locale/.
"""

import logging
import os
from typing import Any, Dict

import yaml

logger: logging.Logger = logging.getLogger(__name__)

# Conversion constants
_METERS_TO_FEET: float = 3.28084
_SQ_METERS_TO_SQ_MILES: float = 3.86102e-7
_CM_TO_INCHES: float = 0.393701

_LOCALE_DIR: str = os.path.join(os.path.dirname(__file__), "data", "locale")
_SUPPORTED_LANGUAGES: tuple = ("en", "fr", "es", "de", "it")


def _load_strings(language: str) -> Dict[str, str]:
    """Load a language YAML file, falling back to English on failure."""
    if language not in _SUPPORTED_LANGUAGES:
        logger.warning(
            f"Unsupported report language '{language}', falling back to 'en'."
        )
        language = "en"

    path = os.path.join(_LOCALE_DIR, f"{language}.yaml")
    if not os.path.isfile(path):
        logger.warning(f"Locale file not found: {path}, falling back to 'en'.")
        path = os.path.join(_LOCALE_DIR, "en.yaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class ReportLocale:
    """Provides translated strings and unit-formatted values for the report."""

    def __init__(self, language: str = "en", unit_system: str = "metric") -> None:
        self._language: str = language
        self._unit_system: str = unit_system
        self._strings: Dict[str, str] = _load_strings(language)
        # Always load English as fallback
        self._fallback: Dict[str, str] = (
            _load_strings("en") if language != "en" else self._strings
        )

    @property
    def language(self) -> str:
        return self._language

    @property
    def unit_system(self) -> str:
        return self._unit_system

    def t(self, key: str) -> str:
        """Translate a string key. Falls back to English, then the key itself."""
        result = self._strings.get(key)
        if result is not None:
            return result
        result = self._fallback.get(key)
        if result is not None:
            if self._language != "en":
                logger.debug(f"Missing translation for '{key}' in '{self._language}'")
            return result
        logger.warning(f"Unknown locale key: '{key}'")
        return key

    # ─────────────────────────────────────────────────────────────────────────
    # Unit formatting
    # ─────────────────────────────────────────────────────────────────────────

    def format_area(self, sq_meters: float, precision: int = 6) -> str:
        """Format an area value with appropriate units."""
        if self._unit_system == "imperial":
            value = sq_meters * _SQ_METERS_TO_SQ_MILES
            return f"{value:.{precision}f} mi\u00b2"
        else:
            value = sq_meters / 1e6
            return f"{value:.{precision}f} km\u00b2"

    def format_distance(self, meters: float, precision: int = 2) -> str:
        """Format a distance in meters or feet (with unit label)."""
        if self._unit_system == "imperial":
            return f"{meters * _METERS_TO_FEET:.{precision}f} ft"
        else:
            return f"{meters:.{precision}f} m"

    def format_distance_label(self, meters: float, precision: int = 2) -> str:
        """Format a distance with full unit word (e.g. '1.23 meters')."""
        if self._unit_system == "imperial":
            return f"{meters * _METERS_TO_FEET:.{precision}f} {self.t('unit_feet')}"
        else:
            return f"{meters:.{precision}f} {self.t('unit_meters')}"

    def format_gsd(self, meters_per_px: float, precision: int = 2) -> str:
        """Format ground sampling distance."""
        if self._unit_system == "imperial":
            value = meters_per_px * 100.0 * _CM_TO_INCHES
            return f"{value:.{precision}f} in/px"
        else:
            value = meters_per_px * 100.0
            return f"{value:.{precision}f} cm/px"

    def format_time(self, seconds: float, precision: int = 2) -> str:
        """Format a time duration."""
        return f"{seconds:.{precision}f} {self.t('unit_seconds_short')}"

    def distance_unit_label(self) -> str:
        """Return the full unit word for distances."""
        if self._unit_system == "imperial":
            return self.t("unit_feet")
        else:
            return self.t("unit_meters")

    def distance_unit_short(self) -> str:
        """Return short unit label for distances."""
        if self._unit_system == "imperial":
            return "ft"
        else:
            return "m"
