# pyre-strict
import logging
from typing import Optional

from opensfm import context


def setup() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s", level=logging.DEBUG
    )


def memory_available() -> Optional[int]:
    """Available memory in MB.

    Delegates to the platform-aware implementation. This used to shell out to
    `free -t -m` unconditionally, which on Windows printed a command-not-found
    error and returned None, silently disabling the memory-aware queue and
    process sizing in `features_processing` and `undistort`.
    """
    return context.memory_available()
