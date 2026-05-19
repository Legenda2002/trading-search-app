"""Runtime paths and tuning knobs.

When the app runs from source (development) all data lives under
``<repo>/data/``. When packaged with PyInstaller and shipped to an end user,
``sys.frozen`` flips to True and we switch to platform-specific writable
locations so the .exe in ``C:\\Program Files`` doesn't try to write into its
own directory (which fails without admin rights).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_frozen() -> bool:
    """True when the app is running from a PyInstaller-built binary."""
    return bool(getattr(sys, "frozen", False))


def _user_data_dir() -> Path:
    """Per-user writable directory for indexes, thumbnails and DB."""
    if not _is_frozen():
        return PROJECT_ROOT / "data"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "TradingSearch" / "data"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TradingSearch" / "data"
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "TradingSearch" / "data"


def _bundled_resources_dir() -> Path | None:
    """Read-only resources shipped inside the PyInstaller bundle.

    For ``--onefile`` builds PyInstaller exposes the extraction dir via
    ``sys._MEIPASS``. For ``--onedir`` builds, resources sit next to the
    executable. Returns None when running from source.
    """
    if not _is_frozen():
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "resources"
    return Path(sys.executable).parent / "resources"


DATA_DIR = _user_data_dir()
DB_PATH = DATA_DIR / "app.db"

ORIGINALS_DIR = DATA_DIR / "images" / "originals"
THUMBNAILS_DIR = DATA_DIR / "images" / "thumbnails"
INDEX_DIR = DATA_DIR / "index"
DESCRIPTORS_DIR = INDEX_DIR / "descriptors"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"
HF_CACHE_DIR = DATA_DIR / "hf_cache"

# Optional read-only HuggingFace cache shipped inside the bundle. Used as a
# fallback when the writable HF_CACHE_DIR is empty so end users don't need
# internet on first launch.
_bundle_resources = _bundled_resources_dir()
BUNDLED_HF_CACHE_DIR = (_bundle_resources / "hf_cache") if _bundle_resources else None

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_MATCH_THRESHOLD = 10
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_INLIERS = 8

# DINOv2-small via HuggingFace transformers. 21M params, 384-dim output.
DINOV2_MODEL_NAME = "facebook/dinov2-small"
EMBEDDING_DIM = 384

# Aspect ratio used to letterbox-pad query images before feeding DINOv2.
# Roughly matches the canonical 1024x560 landscape rendering used to index
# the XAUUSD library. Letterboxing avoids the aspect-ratio cliff where a
# user-supplied portrait screenshot (e.g. 529x619 from a Snipping Tool grab)
# drops the true source from rank 19 down to rank 4790 because DINOv2's
# square preprocess squashes candles differently. Bars are filled with the
# dark grey background most charts already use.
QUERY_LETTERBOX_ASPECT = 1024 / 560
QUERY_LETTERBOX_FILL = 20  # dark grey, blends with typical chart backgrounds

# Hybrid retrieval — how many embedding-nearest candidates to push through
# the ORB+RANSAC verification stage. Bigger = better recall, slower.
#
# 2000 lets smart-mode catch genuine duplicates that sit outside the very
# top-K of the embedding ranking (this is common for charts because DINOv2
# cosines cluster tightly). On the 17k XAUUSD library smart-mode now takes
# ~8-12 s on a 10-core CPU, but it is essentially "full library" coverage:
# the embedding pass touches every image, and ORB then verifies a much
# wider net than before.
HYBRID_TOP_CANDIDATES = 2000

# Inlier count above which we consider an ORB match an "exact" fragment.
EXACT_MATCH_INLIER_THRESHOLD = 50


def ensure_data_dirs() -> None:
    """Create runtime data directories used by the desktop app."""
    for path in (ORIGINALS_DIR, THUMBNAILS_DIR, DESCRIPTORS_DIR, HF_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def is_frozen() -> bool:
    """Public alias of the private helper, useful for diagnostics."""
    return _is_frozen()
