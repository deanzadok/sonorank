"""
Clarius SDK loading utilities.

Setup:
    1. Copy libcast.so and pyclariuscast.so (for your Python version) into this folder
    2. Connect PC to the probe's WiFi network (DIRECT-...)
    3. Open Clarius App on phone, connect to probe, set Cast Permission to "Research"

Reader's guide: all operating-system differences in locating and importing the
native Cast SDK belong here. Callers receive the imported ``pyclariuscast``
module without manipulating dynamic-library search paths themselves.
"""

import ctypes
import sys
from pathlib import Path

# Script directory (where SDK files should be located)
SCRIPT_DIR = Path(__file__).resolve().parent


def load_clarius_sdk(sdk_dir: str = None):
    """
    Load the Clarius Cast SDK.

    On Linux, loads libcast.so with RTLD_GLOBAL flag so pyclariuscast can find symbols.

    Args:
        sdk_dir: Directory containing libcast.so and pyclariuscast.so.
                 If None, uses the sonorank_realtime directory.

    Returns:
        pyclariuscast module
    """
    if sdk_dir is None:
        sdk_dir = SCRIPT_DIR
    else:
        sdk_dir = Path(sdk_dir)

    libcast_path = sdk_dir / "libcast.so"
    pyclariuscast_path = sdk_dir / "pyclariuscast.so"

    if sys.platform.startswith("linux"):
        if not libcast_path.exists():
            raise FileNotFoundError(
                f"libcast.so not found at {libcast_path}\n"
                "Copy libcast.so from the Clarius SDK to this directory."
            )
        if not pyclariuscast_path.exists():
            raise FileNotFoundError(
                f"pyclariuscast.so not found at {pyclariuscast_path}\n"
                "Copy pyclariuscast.so (for your Python version) from the Clarius SDK."
            )

        # Load libcast with RTLD_GLOBAL so pyclariuscast can find symbols
        ctypes.CDLL(str(libcast_path), ctypes.RTLD_GLOBAL)

        # Add sdk_dir to path so we can import pyclariuscast
        sys.path.insert(0, str(sdk_dir))

    import pyclariuscast
    return pyclariuscast


# Placeholder callbacks for unused Cast API events
def noop_raw_image(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    """No-op callback for raw images."""
    pass


def noop_spectrum_image(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    """No-op callback for spectrum images."""
    pass


def noop_imu_data(imu):
    """No-op callback for IMU data."""
    pass
