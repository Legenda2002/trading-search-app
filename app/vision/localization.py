import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.matcher import MatchPair

logger = logging.getLogger(__name__)


REASON_OK = "ok"
REASON_TOO_FEW_MATCHES = "too_few_matches"
REASON_HOMOGRAPHY_FAILED = "homography_failed"
REASON_TOO_FEW_INLIERS = "too_few_inliers"
REASON_PERSPECTIVE_TRANSFORM_FAILED = "perspective_transform_failed"
REASON_NON_FINITE_POLYGON = "non_finite_polygon"
REASON_POLYGON_NOT_CONVEX = "polygon_not_convex"
REASON_POLYGON_AREA_TOO_SMALL = "polygon_area_too_small"
REASON_POLYGON_SIDE_TOO_SHORT = "polygon_side_too_short"
REASON_POLYGON_AREA_MISMATCH = "polygon_area_mismatch"
REASON_POLYGON_OUTSIDE_BOUNDS = "polygon_outside_bounds"


@dataclass(frozen=True)
class Localization:
    homography: np.ndarray
    polygon: np.ndarray
    inlier_count: int
    confidence: float
    query_points: np.ndarray
    target_points: np.ndarray


@dataclass(frozen=True)
class LocalizationDiagnostic:
    """Verbose result of localize_with_reason().

    Always includes the rejection reason (or REASON_OK on success) so the
    debug benchmark can classify every candidate. Carries any intermediate
    artefacts that were produced before rejection so the caller can inspect
    them (e.g. number of inliers even when the polygon was rejected).
    """

    localization: Localization | None
    reason: str
    inlier_count: int = 0
    homography_succeeded: bool = False


class Localizer:
    def __init__(
        self,
        min_matches: int = 10,
        ransac_threshold: float = 5.0,
        min_inliers: int = 8,
        min_area_ratio: float = 0.1,
        max_area_ratio: float = 3.0,
        vertex_margin: float = 0.25,
    ) -> None:
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.vertex_margin = vertex_margin

    def localize(
        self,
        *,
        query_keypoints: np.ndarray,
        target_keypoints: np.ndarray,
        matches: list[MatchPair],
        query_size: tuple[int, int],
        target_size: tuple[int, int] | None = None,
    ) -> Localization | None:
        diag = self.localize_with_reason(
            query_keypoints=query_keypoints,
            target_keypoints=target_keypoints,
            matches=matches,
            query_size=query_size,
            target_size=target_size,
        )
        return diag.localization

    def localize_with_reason(
        self,
        *,
        query_keypoints: np.ndarray,
        target_keypoints: np.ndarray,
        matches: list[MatchPair],
        query_size: tuple[int, int],
        target_size: tuple[int, int] | None = None,
    ) -> LocalizationDiagnostic:
        if len(matches) < self.min_matches:
            return LocalizationDiagnostic(
                localization=None, reason=REASON_TOO_FEW_MATCHES
            )

        src = np.float32(
            [query_keypoints[match.query_index] for match in matches]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [target_keypoints[match.train_index] for match in matches]
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold)
        if homography is None or mask is None:
            return LocalizationDiagnostic(
                localization=None, reason=REASON_HOMOGRAPHY_FAILED
            )

        inlier_count = int(mask.sum())
        if inlier_count < self.min_inliers:
            return LocalizationDiagnostic(
                localization=None,
                reason=REASON_TOO_FEW_INLIERS,
                inlier_count=inlier_count,
                homography_succeeded=True,
            )

        width, height = query_size
        corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(corners, homography)
        except cv2.error:
            return LocalizationDiagnostic(
                localization=None,
                reason=REASON_PERSPECTIVE_TRANSFORM_FAILED,
                inlier_count=inlier_count,
                homography_succeeded=True,
            )

        polygon = transformed.reshape(-1, 2)
        polygon_reason = self._polygon_rejection_reason(
            polygon, query_size=query_size, target_size=target_size
        )
        if polygon_reason is not None:
            return LocalizationDiagnostic(
                localization=None,
                reason=polygon_reason,
                inlier_count=inlier_count,
                homography_succeeded=True,
            )

        confidence = inlier_count / max(len(matches), 1)
        mask_flat = mask.ravel().astype(bool)

        localization = Localization(
            homography=homography,
            polygon=polygon,
            inlier_count=inlier_count,
            confidence=confidence,
            query_points=src.reshape(-1, 2)[mask_flat],
            target_points=dst.reshape(-1, 2)[mask_flat],
        )
        return LocalizationDiagnostic(
            localization=localization,
            reason=REASON_OK,
            inlier_count=inlier_count,
            homography_succeeded=True,
        )

    def _polygon_rejection_reason(
        self,
        polygon: np.ndarray,
        *,
        query_size: tuple[int, int],
        target_size: tuple[int, int] | None,
    ) -> str | None:
        """Return a reason code if the polygon fails validation, else None."""
        if not np.all(np.isfinite(polygon)):
            return REASON_NON_FINITE_POLYGON

        if not self._is_convex(polygon):
            return REASON_POLYGON_NOT_CONVEX

        area = float(cv2.contourArea(polygon.astype(np.float32)))
        if area <= 4.0:
            return REASON_POLYGON_AREA_TOO_SMALL

        sides = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1)
        if sides.min() < 2.0:
            return REASON_POLYGON_SIDE_TOO_SHORT

        query_area = max(query_size[0] * query_size[1], 1)
        ratio = area / query_area
        if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
            return REASON_POLYGON_AREA_MISMATCH

        if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
            tw, th = target_size
            margin_x = tw * self.vertex_margin
            margin_y = th * self.vertex_margin
            for vx, vy in polygon:
                if not (
                    -margin_x <= vx <= tw + margin_x
                    and -margin_y <= vy <= th + margin_y
                ):
                    return REASON_POLYGON_OUTSIDE_BOUNDS

        return None

    @staticmethod
    def _is_convex(polygon: np.ndarray) -> bool:
        n = len(polygon)
        if n < 3:
            return False
        signs: list[bool] = []
        for index in range(n):
            p1 = polygon[index]
            p2 = polygon[(index + 1) % n]
            p3 = polygon[(index + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (
                p3[0] - p2[0]
            )
            signs.append(cross > 0)
        return all(signs) or not any(signs)
