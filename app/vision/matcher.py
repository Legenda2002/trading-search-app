from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchPair:
    query_index: int
    train_index: int
    distance: float


class DescriptorMatcher:
    def __init__(self, algorithm: str = "ORB", ratio: float = 0.75) -> None:
        self.algorithm = algorithm.upper()
        self.ratio = ratio
        norm = cv2.NORM_HAMMING if self.algorithm in {"ORB", "AKAZE"} else cv2.NORM_L2
        self.matcher = cv2.BFMatcher(norm)

    def match(
        self,
        query_descriptors: np.ndarray | None,
        target_descriptors: np.ndarray | None,
    ) -> list[MatchPair]:
        if query_descriptors is None or target_descriptors is None:
            return []
        if len(query_descriptors) < 2 or len(target_descriptors) < 2:
            return []

        knn_matches = self.matcher.knnMatch(query_descriptors, target_descriptors, k=2)

        good: list[MatchPair] = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < self.ratio * second.distance:
                good.append(
                    MatchPair(
                        query_index=best.queryIdx,
                        train_index=best.trainIdx,
                        distance=float(best.distance),
                    )
                )
        return good

    def good_match_count(
        self,
        query_descriptors: np.ndarray | None,
        target_descriptors: np.ndarray | None,
    ) -> int:
        return len(self.match(query_descriptors, target_descriptors))
