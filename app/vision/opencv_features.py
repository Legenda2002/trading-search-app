import logging
from pathlib import Path

import cv2
import numpy as np

from app.vision.preprocessing import load_grayscale_image, normalize_chart_image

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, algorithm: str = "ORB", max_features: int = 8000) -> None:
        self.algorithm = algorithm.upper()
        self.max_features = max_features
        self.detector = self._create_detector()
        logger.debug(
            "FeatureExtractor created: algorithm=%s max_features=%d",
            self.algorithm,
            self.max_features,
        )

    def extract_from_path(self, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        image = load_grayscale_image(path)
        if image is None:
            logger.warning("Could not load image for extraction: %s", path)
            return np.empty((0, 2), dtype=np.float32), None
        points, descriptors = self.extract_from_image(image)
        logger.debug(
            "Extracted %d keypoints from %s using %s",
            len(points),
            path.name,
            self.algorithm,
        )
        return points, descriptors

    def extract_with_shape(
        self, path: Path
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
        image = load_grayscale_image(path)
        if image is None:
            logger.warning("Could not load image for extraction: %s", path)
            return np.empty((0, 2), dtype=np.float32), None, (0, 0)

        height, width = image.shape[:2]
        points, descriptors = self.extract_from_image(image)
        logger.debug(
            "Extracted %d keypoints from %s (%dx%d) using %s",
            len(points),
            path.name,
            width,
            height,
            self.algorithm,
        )
        return points, descriptors, (width, height)

    def extract_from_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        image = normalize_chart_image(image)
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        points = self._serialize_keypoints(keypoints)
        return points, descriptors

    def _create_detector(self):
        if self.algorithm == "AKAZE":
            return cv2.AKAZE_create()
        if self.algorithm == "ORB":
            return cv2.ORB_create(nfeatures=self.max_features)
        raise ValueError(f"Unsupported feature algorithm: {self.algorithm}")

    @staticmethod
    def _serialize_keypoints(keypoints) -> np.ndarray:
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
