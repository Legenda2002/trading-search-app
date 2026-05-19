from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChartImage:
    id: int
    original_path: Path
    stored_path: Path
    thumbnail_path: Path | None
    filename: str
    width: int
    height: int
    file_hash: str
    descriptor_path: Path | None
    algorithm: str | None
    keypoint_count: int


@dataclass(frozen=True)
class ImportResult:
    imported: int
    skipped: int


MATCH_TYPE_EXACT = "exact"
MATCH_TYPE_SIMILAR = "similar"
MATCH_TYPE_NONE = "none"


@dataclass(frozen=True)
class SearchResult:
    image: "ChartImage"
    score: float
    match_count: int
    inlier_count: int = 0
    localization_score: float = 0.0
    localized: bool = False
    homography: np.ndarray | None = field(default=None, repr=False)
    polygon: np.ndarray | None = field(default=None, repr=False)
    matched_query_points: np.ndarray | None = field(default=None, repr=False)
    matched_target_points: np.ndarray | None = field(default=None, repr=False)
    rejection_reason: str | None = None
    embedding_similarity: float = 0.0
    similarity_percent: float = 0.0
    match_type: str = MATCH_TYPE_NONE


@dataclass(frozen=True)
class SearchOutcome:
    query_keypoint_count: int
    results: list["SearchResult"]
