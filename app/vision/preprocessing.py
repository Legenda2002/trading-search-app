from pathlib import Path

import cv2
import numpy as np


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_grayscale_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return normalize_chart_image(image)


def normalize_chart_image(image: np.ndarray) -> np.ndarray:
    """Preprocess a chart image for feature detection.

    For trading screenshots we keep thin candle wicks intact (no blur) and
    use CLAHE (adaptive local contrast) instead of global histogram
    equalization, which would otherwise crush dark backgrounds together
    with bright candle bodies.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(image)
