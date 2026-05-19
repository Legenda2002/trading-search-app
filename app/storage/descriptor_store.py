from pathlib import Path

import numpy as np

from app.core.config import DESCRIPTORS_DIR


class DescriptorStore:
    def descriptor_path(self, image_id: int, algorithm: str) -> Path:
        return DESCRIPTORS_DIR / f"{image_id}_{algorithm.lower()}.npz"

    def save(
        self,
        *,
        image_id: int,
        algorithm: str,
        keypoints: np.ndarray,
        descriptors: np.ndarray | None,
    ) -> Path | None:
        if descriptors is None or len(descriptors) == 0:
            return None

        path = self.descriptor_path(image_id, algorithm)
        np.savez_compressed(
            path,
            keypoints=keypoints,
            descriptors=descriptors,
        )
        return path

    def load(self, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        if not path.exists():
            return None

        data = np.load(path)
        return data["keypoints"], data["descriptors"]
