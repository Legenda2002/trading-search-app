import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Tell OpenCV not to use its own thread pool. We parallelize across images via
# ThreadPoolExecutor, and OpenCV's internal parallelism inside knnMatch ends up
# fighting our threads on the same cores, hurting throughput. Single-threaded
# cv2 + outer ThreadPool gives much better scaling on multi-core CPUs.
cv2.setNumThreads(1)

# One BFMatcher per worker thread — avoids reconstructing it on every match.
_thread_local = threading.local()


def _get_bf_matcher(norm: int) -> cv2.BFMatcher:
    cached = getattr(_thread_local, "bf", None)
    cached_norm = getattr(_thread_local, "bf_norm", None)
    if cached is not None and cached_norm == norm:
        return cached
    matcher = cv2.BFMatcher(norm)
    _thread_local.bf = matcher
    _thread_local.bf_norm = norm
    return matcher

from app.core.config import (
    MIN_INLIERS,
    MIN_MATCH_THRESHOLD,
    RANSAC_REPROJ_THRESHOLD,
)
from app.core.models import ChartImage, SearchOutcome, SearchResult
from app.storage.database import Database
from app.storage.descriptor_store import DescriptorStore
from app.vision.localization import (
    REASON_OK,
    REASON_TOO_FEW_MATCHES,
    Localization,
    Localizer,
)
from app.vision.matcher import DescriptorMatcher, MatchPair
from app.vision.opencv_features import FeatureExtractor

logger = logging.getLogger(__name__)


def _match_descriptors_threadsafe(
    query_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    ratio: float,
    norm: int,
) -> list[MatchPair]:
    """Run a Lowe-ratio knnMatch and return good MatchPair objects.

    Uses one cv2.BFMatcher per worker thread (via threading.local) so threads
    never share matcher state. cv2 releases the GIL inside knnMatch, so this
    scales across cores.
    """
    if query_descriptors is None or target_descriptors is None:
        return []
    if len(query_descriptors) < 2 or len(target_descriptors) < 2:
        return []

    bf = _get_bf_matcher(norm)
    knn = bf.knnMatch(query_descriptors, target_descriptors, k=2)

    good: list[MatchPair] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(
                MatchPair(
                    query_index=best.queryIdx,
                    train_index=best.trainIdx,
                    distance=float(best.distance),
                )
            )
    return good


def _coarse_match_count(
    coarse_query_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    ratio: float,
    norm: int,
) -> int:
    """Fast cousin of _match_descriptors_threadsafe used in the coarse stage.

    Same ratio test, but we only return the number of surviving matches —
    no MatchPair allocation, no Python attribute access per match. This is
    the hottest inner loop, called once per library image.
    """
    if coarse_query_descriptors is None or target_descriptors is None:
        return 0
    if len(coarse_query_descriptors) < 2 or len(target_descriptors) < 2:
        return 0

    bf = _get_bf_matcher(norm)
    knn = bf.knnMatch(coarse_query_descriptors, target_descriptors, k=2)

    count = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            count += 1
    return count


class SearchEngine:
    def __init__(
        self,
        database: Database,
        descriptor_store: DescriptorStore,
        algorithm: str = "ORB",
        top_k: int = 10,
        min_matches: int = MIN_MATCH_THRESHOLD,
        ransac_threshold: float = RANSAC_REPROJ_THRESHOLD,
        min_inliers: int = MIN_INLIERS,
        max_workers: int | None = None,
        localize_top_candidates: int = 100,
        coarse_query_keypoints: int = 128,
        fine_candidates: int = 400,
    ) -> None:
        self.database = database
        self.descriptor_store = descriptor_store
        self.algorithm = algorithm
        self.top_k = top_k
        self.extractor = FeatureExtractor(algorithm=algorithm)
        self.matcher = DescriptorMatcher(algorithm=algorithm)
        self.localizer = Localizer(
            min_matches=min_matches,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )

        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers if max_workers is not None else cpu_count
        self.localize_top_candidates = localize_top_candidates
        self.coarse_query_keypoints = coarse_query_keypoints
        self.fine_candidates = fine_candidates

        self._norm = (
            cv2.NORM_HAMMING
            if algorithm.upper() in {"ORB", "AKAZE"}
            else cv2.NORM_L2
        )

        self._descriptor_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    def invalidate_cache(self) -> None:
        """Clear the in-memory descriptor cache.

        Should be called after the library is mutated (re-indexed).
        """
        with self._cache_lock:
            self._descriptor_cache.clear()

    def warmup(self) -> int:
        """Eagerly preload all known descriptors into memory.

        Loads in parallel to saturate SSD bandwidth.
        Returns the number of entries loaded. Safe to call multiple times.
        """
        library = self.database.list_images()
        to_load: list[ChartImage] = []
        for image in library:
            if image.descriptor_path is None:
                continue
            if image.descriptor_path in self._descriptor_cache:
                continue
            to_load.append(image)

        if not to_load:
            logger.info(
                "SearchEngine warmup: nothing to do (%d cached)",
                len(self._descriptor_cache),
            )
            return len(self._descriptor_cache)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(self._load_features_cached, to_load, chunksize=64))

        logger.info(
            "SearchEngine warmup: %d descriptor sets cached",
            len(self._descriptor_cache),
        )
        return len(self._descriptor_cache)

    def search_by_image(
        self,
        query_image_path: Path,
        *,
        debug: bool = False,
        candidates: list[ChartImage] | None = None,
    ) -> SearchOutcome:
        logger.info("Search by image: %s", query_image_path)
        query_keypoints, query_descriptors, query_size = (
            self.extractor.extract_with_shape(query_image_path)
        )
        return self._search(
            query_keypoints,
            query_descriptors,
            query_size,
            debug=debug,
            library_override=candidates,
        )

    def search_by_array(
        self,
        image: np.ndarray,
        *,
        debug: bool = False,
        candidates: list[ChartImage] | None = None,
    ) -> SearchOutcome:
        if image is None or image.size == 0:
            logger.warning("Empty query array")
            return SearchOutcome(query_keypoint_count=0, results=[])

        height, width = image.shape[:2]
        logger.info("Search by array %dx%d", width, height)
        query_keypoints, query_descriptors = self.extractor.extract_from_image(image)
        return self._search(
            query_keypoints,
            query_descriptors,
            (width, height),
            debug=debug,
            library_override=candidates,
        )

    def _search(
        self,
        query_keypoints: np.ndarray,
        query_descriptors: np.ndarray | None,
        query_size: tuple[int, int],
        *,
        debug: bool = False,
        library_override: list[ChartImage] | None = None,
    ) -> SearchOutcome:
        query_kp_count = int(query_keypoints.shape[0]) if query_keypoints is not None else 0
        logger.info(
            "Query keypoints=%d size=%dx%d descriptors=%s",
            query_kp_count,
            query_size[0],
            query_size[1],
            "yes" if query_descriptors is not None and len(query_descriptors) else "no",
        )

        if query_descriptors is None or len(query_descriptors) == 0:
            logger.warning("Empty query descriptors, returning no results")
            return SearchOutcome(query_keypoint_count=query_kp_count, results=[])

        if library_override is not None:
            library = library_override
            logger.info(
                "Comparing against pre-selected %d candidates (workers=%d)",
                len(library),
                self.max_workers,
            )
        else:
            library = self.database.list_images()
            logger.info(
                "Comparing against %d indexed images (workers=%d)",
                len(library),
                self.max_workers,
            )

        candidates = self._prepare_candidates(library)
        if not candidates:
            return SearchOutcome(query_keypoint_count=query_kp_count, results=[])

        ratio = self.matcher.ratio
        norm = self._norm

        # In debug mode we still use the cascade so big queries finish in a
        # reasonable wall-clock, but we widen the fine pool drastically so the
        # top-K we care about is essentially never filtered out by coarse.
        coarse_n = min(self.coarse_query_keypoints, len(query_descriptors))
        effective_fine = max(self.fine_candidates, 2000) if debug else self.fine_candidates
        use_cascade = (
            len(candidates) > effective_fine
            and len(query_descriptors) > coarse_n * 1.5
        )

        if use_cascade:
            # Pick descriptors uniformly across the query array using a stride.
            # ORB returns descriptors grouped by pyramid level — naive [:N]
            # slicing biases the coarse subset toward a single scale/region and
            # misses the correct image when the query is large. Striding keeps
            # spatial and scale coverage diverse.
            stride = max(1, len(query_descriptors) // coarse_n)
            coarse_query = np.ascontiguousarray(query_descriptors[::stride][:coarse_n])
            logger.info(
                "Cascade enabled: coarse query kp=%d (stride=%d, from %d), fine pool=%d",
                len(coarse_query),
                stride,
                len(query_descriptors),
                self.fine_candidates,
            )

            def coarse_one(item: tuple[ChartImage, np.ndarray, np.ndarray]) -> int:
                _, _, target_descriptors = item
                return _coarse_match_count(
                    coarse_query, target_descriptors, ratio, norm
                )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                coarse_scores = list(
                    executor.map(coarse_one, candidates, chunksize=64)
                )

            scored_pairs = sorted(
                zip(coarse_scores, candidates),
                key=lambda pair: pair[0],
                reverse=True,
            )
            non_zero = sum(1 for s, _ in scored_pairs if s > 0)
            fine_subset = [
                pair[1] for pair in scored_pairs[:effective_fine] if pair[0] > 0
            ]
            logger.info(
                "Coarse done: %d candidates passed (>0 matches), %d kept for fine match",
                non_zero,
                len(fine_subset),
            )
        else:
            fine_subset = candidates
            logger.info(
                "Cascade skipped: fine matching all %d candidates", len(fine_subset)
            )

        def match_one(item: tuple[ChartImage, np.ndarray, np.ndarray]):
            image, target_keypoints, target_descriptors = item
            matches = _match_descriptors_threadsafe(
                query_descriptors, target_descriptors, ratio, norm
            )
            return image, target_keypoints, matches

        matched: list[tuple[ChartImage, np.ndarray, list[MatchPair]]] = []
        empty_match_images: list[ChartImage] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for image, target_keypoints, matches in executor.map(
                match_one, fine_subset, chunksize=16
            ):
                if matches:
                    matched.append((image, target_keypoints, matches))
                elif debug:
                    empty_match_images.append(image)

        logger.info(
            "Fine done: %d candidates have non-empty matches "
            "(%d candidates had zero matches)",
            len(matched),
            len(empty_match_images),
        )

        matched.sort(key=lambda triple: len(triple[2]), reverse=True)

        scored: list[SearchResult] = []
        localize_budget = (
            len(matched) if debug else self.localize_top_candidates
        )
        for index, (image, target_keypoints, matches) in enumerate(matched):
            allow_localize = index < localize_budget
            result = self._build_result(
                image=image,
                matches=matches,
                query_keypoints=query_keypoints,
                target_keypoints=target_keypoints,
                query_kp_count=query_kp_count,
                query_size=query_size,
                allow_localize=allow_localize,
                track_reason=debug,
            )
            scored.append(result)

        scored.sort(
            key=lambda result: (
                1 if result.localized else 0,
                result.score,
                result.localization_score,
            ),
            reverse=True,
        )

        if debug:
            for image in empty_match_images:
                scored.append(
                    SearchResult(
                        image=image,
                        score=0.0,
                        match_count=0,
                        rejection_reason="no_descriptor_matches",
                    )
                )

        top_k = len(scored) if debug else self.top_k
        top = scored[:top_k]
        logger.info("Top %d of %d candidates returned", len(top), len(scored))
        return SearchOutcome(query_keypoint_count=query_kp_count, results=top)

    def _prepare_candidates(
        self, library: list[ChartImage]
    ) -> list[tuple[ChartImage, np.ndarray, np.ndarray]]:
        uncached = [
            image
            for image in library
            if image.descriptor_path is not None
            and image.descriptor_path not in self._descriptor_cache
        ]
        if uncached:
            logger.info(
                "Loading %d uncached descriptor sets in parallel", len(uncached)
            )
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(executor.map(self._load_features_cached, uncached, chunksize=64))

        candidates: list[tuple[ChartImage, np.ndarray, np.ndarray]] = []
        missing = 0
        for image in library:
            if image.descriptor_path is None:
                missing += 1
                continue
            cached = self._descriptor_cache.get(image.descriptor_path)
            if cached is None:
                missing += 1
                continue
            target_keypoints, target_descriptors = cached
            candidates.append((image, target_keypoints, target_descriptors))
        if missing:
            logger.debug("Skipped %d library entries without descriptors", missing)
        return candidates

    def _load_features_cached(
        self, image: ChartImage
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if image.descriptor_path is None:
            return None
        path = image.descriptor_path
        cached = self._descriptor_cache.get(path)
        if cached is not None:
            return cached

        loaded = self.descriptor_store.load(path)
        if loaded is None:
            return None
        keypoints, descriptors = loaded
        if descriptors is None or len(descriptors) == 0:
            return None

        with self._cache_lock:
            self._descriptor_cache[path] = (keypoints, descriptors)
        return keypoints, descriptors

    def _build_result(
        self,
        *,
        image: ChartImage,
        matches: list[MatchPair],
        query_keypoints: np.ndarray,
        target_keypoints: np.ndarray,
        query_kp_count: int,
        query_size: tuple[int, int],
        allow_localize: bool = True,
        track_reason: bool = False,
    ) -> SearchResult:
        match_count = len(matches)
        score = match_count / max(query_kp_count, 1)

        if not allow_localize or query_size[0] <= 0 or query_size[1] <= 0:
            return SearchResult(
                image=image,
                score=score,
                match_count=match_count,
                rejection_reason="not_localized" if track_reason else None,
            )

        diagnostic = self.localizer.localize_with_reason(
            query_keypoints=query_keypoints,
            target_keypoints=target_keypoints,
            matches=matches,
            query_size=query_size,
            target_size=(image.width, image.height),
        )
        localization = diagnostic.localization

        if localization is None:
            return SearchResult(
                image=image,
                score=score,
                match_count=match_count,
                inlier_count=diagnostic.inlier_count,
                rejection_reason=diagnostic.reason if track_reason else None,
            )

        return SearchResult(
            image=image,
            score=score,
            match_count=match_count,
            inlier_count=localization.inlier_count,
            localization_score=localization.confidence,
            localized=True,
            homography=localization.homography,
            polygon=localization.polygon,
            matched_query_points=localization.query_points,
            matched_target_points=localization.target_points,
            rejection_reason=REASON_OK if track_reason else None,
        )
