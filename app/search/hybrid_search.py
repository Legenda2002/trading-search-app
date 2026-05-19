"""Hybrid retrieval: DINOv2 embeddings for candidate shortlisting,
ORB+RANSAC for precise verification and localisation.

Three modes:

* ``smart``   – embedding → top-K candidates → ORB verify → unified score.
* ``exact``   – plain ORB over the whole library (the original Phase 1 path).
* ``similar`` – embedding only, no ORB. Fastest, surfaces visually-close
                charts even when the rendering style differs.

Scoring is calibrated so the UI can report a single 0-100 % similarity
number that means roughly the same thing across modes. The calibration
window is intentionally tight for trading charts, where DINOv2 cosines
between unrelated images already sit at 0.70-0.85:

* 90-100 % – ORB localisation succeeded with many inliers — the query is
             genuinely a crop of the candidate (or one pixel shift away).
* 70-89 %  – Homography survived RANSAC with few supporting inliers — a
             partial overlap or a near-duplicate.
* 40-74 %  – ORB found keypoint matches AND embeddings agree above the
             noise floor — visually similar pattern in a different render.
* 30-39 %  – Embedding-only similarity, no ORB structure.
* dropped  – Below the cosine noise floor (~0.84). Almost certainly two
             unrelated charts that merely both contain candles.

These are heuristics, not probabilities. The calibration constants were
picked from offline benchmarks on the XAUUSD library.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import (
    EXACT_MATCH_INLIER_THRESHOLD,
    HYBRID_TOP_CANDIDATES,
)
from app.core.models import (
    MATCH_TYPE_EXACT,
    MATCH_TYPE_NONE,
    MATCH_TYPE_SIMILAR,
    ChartImage,
    SearchOutcome,
    SearchResult,
)
from app.search.search_engine import SearchEngine
from app.storage.database import Database
from app.storage.embedding_store import EmbeddingStore
from app.vision.embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)


MODE_SMART = "smart"
MODE_EXACT = "exact"
MODE_SIMILAR = "similar"
ALLOWED_MODES = {MODE_SMART, MODE_EXACT, MODE_SIMILAR}

# DINOv2 cosines on trading charts naturally cluster between 0.70 and 0.86
# even for completely unrelated images, because both are dark candlestick
# panels with similar axes. Two truly similar patterns sit at 0.88-0.94, and
# near-duplicate renders reach 0.96+. We map (0.82, 0.96) → (0, 1) so the
# UI percentage actually means something — anything below 0.82 reads as 0 %
# instead of the previous misleading "71 % похожее" on noise.
EMBED_CALIBRATION_LOW = 0.82
EMBED_CALIBRATION_HIGH = 0.96

# Hard floor for cosine similarity before we claim "похожее". Below this
# embedding-only matches are dropped from the results panel entirely — they
# are statistically indistinguishable from random chart pairs.
EMBED_MIN_COSINE_FOR_SIMILAR = 0.84


class SimilarityScorer:
    """Translate ORB inliers + embedding cosine into a UI-friendly percent."""

    def __init__(
        self,
        exact_inliers: int = EXACT_MATCH_INLIER_THRESHOLD,
        cal_low: float = EMBED_CALIBRATION_LOW,
        cal_high: float = EMBED_CALIBRATION_HIGH,
        min_cosine_for_similar: float = EMBED_MIN_COSINE_FOR_SIMILAR,
    ) -> None:
        self.exact_inliers = exact_inliers
        self.cal_low = cal_low
        self.cal_high = cal_high
        self.min_cosine_for_similar = min_cosine_for_similar

    def calibrate(self, cosine: float) -> float:
        span = max(1e-6, self.cal_high - self.cal_low)
        return max(0.0, min(1.0, (float(cosine) - self.cal_low) / span))

    def embedding_only_percent(self, cosine: float) -> float:
        # Pure-embedding results never beat ORB-localised ones, so the
        # ceiling sits at 65 %. cos ≈ 0.96 maxes out, cos ≈ 0.84 lands at
        # the lower edge of the displayed range.
        return 65.0 * self.calibrate(cosine)

    def score(
        self,
        *,
        inliers: int,
        localized: bool,
        match_count: int,
        cosine: float,
    ) -> tuple[str, float]:
        norm = self.calibrate(cosine)

        if localized and inliers >= self.exact_inliers:
            # Strong, geometrically consistent match: 90-100 %.
            pct = 90.0 + min(10.0, inliers / 30.0)
            return MATCH_TYPE_EXACT, min(100.0, pct)

        if localized and inliers >= 12:
            # Homography survived RANSAC but with few supporting inliers —
            # likely a partial overlap or a near-duplicate region.
            pct = 70.0 + 15.0 * norm
            return MATCH_TYPE_EXACT, min(89.0, pct)

        if match_count > 0 and cosine >= self.min_cosine_for_similar:
            # ORB keypoints matched AND embedding agrees enough — visually
            # related chart in a slightly different rendering/window. We are
            # deliberately conservative here: trading charts give noisy ORB
            # matches very easily, so without a homography we cap at 75 %.
            pct = 35.0 + 35.0 * norm
            match_type = MATCH_TYPE_SIMILAR if pct >= 40.0 else MATCH_TYPE_NONE
            return match_type, min(75.0, pct)

        # Pure embedding match — needs to clear the noise floor or it's not
        # worth showing at all.
        if cosine < self.min_cosine_for_similar:
            return MATCH_TYPE_NONE, 0.0

        pct = self.embedding_only_percent(cosine)
        if pct >= 30.0:
            return MATCH_TYPE_SIMILAR, pct
        return MATCH_TYPE_NONE, pct


class HybridSearchEngine:
    def __init__(
        self,
        database: Database,
        embedding_store: EmbeddingStore,
        embedding_extractor: EmbeddingExtractor,
        orb_engine: SearchEngine,
        *,
        top_candidates: int = HYBRID_TOP_CANDIDATES,
        default_mode: str = MODE_SMART,
        scorer: SimilarityScorer | None = None,
    ) -> None:
        self.database = database
        self.embedding_store = embedding_store
        self.embedding_extractor = embedding_extractor
        self.orb_engine = orb_engine
        self.top_candidates = top_candidates
        self.default_mode = default_mode
        self.scorer = scorer or SimilarityScorer()
        self._library_cache: dict[int, ChartImage] = {}

    @property
    def top_k(self) -> int:
        return self.orb_engine.top_k

    def invalidate_cache(self) -> None:
        self.orb_engine.invalidate_cache()
        self._library_cache.clear()

    def warmup(self) -> None:
        self.orb_engine.warmup()
        self.embedding_extractor.warmup()
        self._refresh_library_cache()
        ids, _ = self.embedding_store.load_all()
        logger.info(
            "HybridSearchEngine warmup: %d embeddings loaded, %d images cached",
            len(ids),
            len(self._library_cache),
        )

    def _refresh_library_cache(self) -> None:
        library = self.database.list_images()
        self._library_cache = {int(img.id): img for img in library}

    def _ensure_library_cache(self) -> None:
        if not self._library_cache:
            self._refresh_library_cache()

    def search_by_image(
        self,
        query_image_path: Path,
        *,
        mode: str | None = None,
        debug: bool = False,
    ) -> SearchOutcome:
        with Image.open(query_image_path) as pil_image:
            array = np.asarray(pil_image.convert("RGB"))
        return self.search_by_array(array, mode=mode, debug=debug)

    def search_by_array(
        self,
        image: np.ndarray,
        *,
        mode: str | None = None,
        debug: bool = False,
    ) -> SearchOutcome:
        if image is None or image.size == 0:
            logger.warning("HybridSearchEngine: empty query array")
            return SearchOutcome(query_keypoint_count=0, results=[])

        effective_mode = (mode or self.default_mode).lower()
        if effective_mode not in ALLOWED_MODES:
            logger.warning(
                "Unknown mode %r, falling back to %s", effective_mode, MODE_SMART
            )
            effective_mode = MODE_SMART

        logger.info(
            "Hybrid search mode=%s, query_shape=%s",
            effective_mode,
            image.shape,
        )

        if effective_mode == MODE_EXACT:
            orb_outcome = self.orb_engine.search_by_array(image, debug=debug)
            # Try to fetch a query embedding even in exact mode so the scorer
            # can distinguish "ORB-matched and visually similar" from
            # "ORB-matched but the images look nothing alike".
            try:
                query_emb = self.embedding_extractor.extract_from_image(image)
            except Exception:
                logger.exception("Embedding extraction failed in exact mode")
                query_emb = None
            return self._enrich_exact_outcome(orb_outcome, query_emb)

        ids, vectors = self.embedding_store.load_all()
        if len(ids) == 0:
            logger.warning(
                "EmbeddingStore is empty — falling back to ORB-only search"
            )
            orb_outcome = self.orb_engine.search_by_array(image, debug=debug)
            return self._enrich_exact_outcome(orb_outcome)

        try:
            query_emb = self.embedding_extractor.extract_from_image(image)
        except Exception:
            logger.exception("Embedding extraction failed, falling back to ORB")
            return self.orb_engine.search_by_array(image, debug=debug)

        scores = vectors @ query_emb.astype(np.float32, copy=False)
        top_k = min(self.top_candidates, len(scores))
        if top_k <= 0:
            return SearchOutcome(query_keypoint_count=0, results=[])
        # argpartition is O(N); for 17k vs 200 it's noticeably faster than
        # full argsort, and we sort only the surviving slice afterwards.
        cut = np.argpartition(-scores, top_k - 1)[:top_k]
        cut = cut[np.argsort(-scores[cut])]
        shortlisted_ids = ids[cut]
        shortlisted_cos = scores[cut]
        id_to_cos = {
            int(image_id): float(cos)
            for image_id, cos in zip(shortlisted_ids, shortlisted_cos)
        }

        self._ensure_library_cache()
        candidate_images: list[ChartImage] = []
        for image_id in shortlisted_ids:
            chart_image = self._library_cache.get(int(image_id))
            if chart_image is not None:
                candidate_images.append(chart_image)
        logger.info(
            "Embedding stage: %d candidates (top cosines=%.3f..%.3f)",
            len(candidate_images),
            float(shortlisted_cos[0]) if len(shortlisted_cos) else 0.0,
            float(shortlisted_cos[-1]) if len(shortlisted_cos) else 0.0,
        )

        if effective_mode == MODE_SIMILAR:
            return self._build_similar_outcome(candidate_images, id_to_cos, debug=debug)

        orb_outcome = self.orb_engine.search_by_array(
            image, debug=debug, candidates=candidate_images
        )
        return self._merge_smart_outcome(
            candidate_images, id_to_cos, orb_outcome, debug=debug
        )

    def _enrich_exact_outcome(
        self,
        orb_outcome: SearchOutcome,
        query_emb: np.ndarray | None = None,
    ) -> SearchOutcome:
        """Add calibrated similarity_percent / match_type to ORB-only results.

        When a query embedding is provided we look up each result's embedding
        in the store and feed the cosine into the scorer; this is what makes
        the "слабая похожесть" filter work in MODE_EXACT too. When no
        embedding is available (e.g. extractor failed) we route the scorer
        through a localisation-only branch by pretending cosine is 1.0 so
        ORB-matched results are not silently dropped.
        """
        enriched: list[SearchResult] = []
        for result in orb_outcome.results:
            cos = 0.0
            if query_emb is not None:
                stored = self.embedding_store.get(int(result.image.id))
                if stored is not None:
                    cos = float(np.dot(stored, query_emb))

            scoring_cos = cos if query_emb is not None else 1.0
            match_type, pct = self.scorer.score(
                inliers=result.inlier_count,
                localized=result.localized,
                match_count=result.match_count,
                cosine=scoring_cos,
            )
            enriched.append(
                replace(
                    result,
                    embedding_similarity=cos,
                    similarity_percent=pct,
                    match_type=match_type,
                )
            )
        return SearchOutcome(
            query_keypoint_count=orb_outcome.query_keypoint_count,
            results=enriched,
        )

    def _build_similar_outcome(
        self,
        candidates: list[ChartImage],
        id_to_cos: dict[int, float],
        *,
        debug: bool,
    ) -> SearchOutcome:
        results: list[SearchResult] = []
        for chart_image in candidates:
            cos = id_to_cos.get(int(chart_image.id), 0.0)
            # Drop noise — see EMBED_MIN_COSINE_FOR_SIMILAR comment in the
            # module header for why anything lower is just "both are charts".
            if cos < self.scorer.min_cosine_for_similar:
                continue
            pct = self.scorer.embedding_only_percent(cos)
            match_type = MATCH_TYPE_SIMILAR if pct >= 30.0 else MATCH_TYPE_NONE
            if match_type == MATCH_TYPE_NONE:
                continue
            results.append(
                SearchResult(
                    image=chart_image,
                    score=cos,
                    match_count=0,
                    embedding_similarity=cos,
                    similarity_percent=pct,
                    match_type=match_type,
                )
            )
        results.sort(key=lambda r: r.similarity_percent, reverse=True)
        top_k = len(results) if debug else self.top_k
        return SearchOutcome(query_keypoint_count=0, results=results[:top_k])

    def _merge_smart_outcome(
        self,
        candidates: list[ChartImage],
        id_to_cos: dict[int, float],
        orb_outcome: SearchOutcome,
        *,
        debug: bool,
    ) -> SearchOutcome:
        enriched: list[SearchResult] = []
        seen_ids: set[int] = set()
        for result in orb_outcome.results:
            cos = id_to_cos.get(int(result.image.id), 0.0)
            match_type, pct = self.scorer.score(
                inliers=result.inlier_count,
                localized=result.localized,
                match_count=result.match_count,
                cosine=cos,
            )
            enriched.append(
                replace(
                    result,
                    embedding_similarity=cos,
                    similarity_percent=pct,
                    match_type=match_type,
                )
            )
            seen_ids.add(int(result.image.id))

        # Fill the remaining shortlist with embedding-only results so the
        # smart mode surfaces visually similar charts even when ORB found
        # nothing for them. We only allow this for embeddings clearly above
        # the noise floor — otherwise the panel fills up with chart-pair
        # noise that misled users into trusting "71 % похожее" matches.
        for chart_image in candidates:
            image_id = int(chart_image.id)
            if image_id in seen_ids:
                continue
            cos = id_to_cos.get(image_id, 0.0)
            if cos < self.scorer.min_cosine_for_similar:
                continue
            pct = self.scorer.embedding_only_percent(cos)
            if pct < 30.0:
                continue
            enriched.append(
                SearchResult(
                    image=chart_image,
                    score=cos,
                    match_count=0,
                    embedding_similarity=cos,
                    similarity_percent=pct,
                    match_type=MATCH_TYPE_SIMILAR,
                )
            )

        # Drop ``MATCH_TYPE_NONE`` rows so the panel doesn't show 35 % grey
        # noise alongside a real 92 % exact match. Debug callers still see
        # the full list.
        if not debug:
            enriched = [r for r in enriched if r.match_type != MATCH_TYPE_NONE]

        enriched.sort(
            key=lambda r: (
                # exact > similar > none
                {MATCH_TYPE_EXACT: 2, MATCH_TYPE_SIMILAR: 1, MATCH_TYPE_NONE: 0}.get(
                    r.match_type, 0
                ),
                r.inlier_count,
                r.similarity_percent,
            ),
            reverse=True,
        )
        top_k = len(enriched) if debug else self.top_k
        return SearchOutcome(
            query_keypoint_count=orb_outcome.query_keypoint_count,
            results=enriched[:top_k],
        )
