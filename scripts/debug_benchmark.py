"""End-to-end debug benchmark for the ORB+RANSAC pipeline.

Runs two test suites against the indexed library:

  Suite A — IN-LIBRARY CROPS
    Randomly picks N images from the library, crops the centre region of each,
    and uses that crop as a query. The expected top-1 is the source image.
    A miss here means the ORB pipeline has a real bug (style is identical).

  Suite B — EXTERNAL SCREENSHOTS
    User-provided fragments captured from MetaTrader / TradingView / IDE
    viewers / etc. No expected answer — we just print the full diagnostic so
    you can see why localisation fails and decide if AI embeddings are needed.

For every query the report shows:
    query keypoints       — how many ORB features were extracted
    top-N candidates      — each with score, matches, inliers, loc-score, and a
                            rejection_reason code from app/vision/localization.py
                            (too_few_matches, too_few_inliers, polygon_*, ...)

Usage:
    python -m scripts.debug_benchmark \\
        --in-library 5 \\
        --external-dir data/csv/external_queries \\
        --top 10

    # Or just Suite A
    python -m scripts.debug_benchmark --in-library 5
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from app.core.logging_config import configure_logging
from app.core.models import SearchOutcome
from app.search.hybrid_search import (
    ALLOWED_MODES,
    MODE_SMART,
    HybridSearchEngine,
)
from app.search.search_engine import SearchEngine
from app.storage.database import Database
from app.storage.descriptor_store import DescriptorStore
from app.storage.embedding_store import EmbeddingStore
from app.vision.embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)

ALGORITHM = "ORB"
ENGINE_ORB = "orb"
ENGINE_HYBRID = "hybrid"

# Reason codes by category — used to colour-code the verdict.
HARD_FAILURES = {
    "no_descriptor_matches",  # zero good ORB matches at all
    "too_few_matches",        # had some matches but below threshold
    "too_few_inliers",        # RANSAC ran but couldn't find geometry
    "homography_failed",
    "perspective_transform_failed",
}
POLYGON_FAILURES = {
    "non_finite_polygon",
    "polygon_not_convex",
    "polygon_area_too_small",
    "polygon_side_too_short",
    "polygon_area_mismatch",
    "polygon_outside_bounds",
}


def build_in_library_queries(
    library_root: Path,
    n: int,
    output_dir: Path,
    seed: int,
) -> list[tuple[Path, str]]:
    """Pick N library images and crop their centre to use as queries.

    Returns a list of (query_path, expected_filename) pairs.
    """
    rng = random.Random(seed)
    all_pngs = sorted(library_root.rglob("*.png"))
    if not all_pngs:
        raise SystemExit(f"No PNG files under {library_root}")

    picks = rng.sample(all_pngs, min(n, len(all_pngs)))
    output_dir.mkdir(parents=True, exist_ok=True)
    queries: list[tuple[Path, str]] = []
    for path in picks:
        with Image.open(path) as img:
            W, H = img.size
            crop = img.crop((W // 4, H // 4, 3 * W // 4, 3 * H // 4))
            query_path = output_dir / f"crop_{path.stem}.png"
            crop.save(query_path)
        queries.append((query_path, path.name))
    return queries


def collect_external_queries(folder: Path | None) -> list[tuple[Path, str]]:
    """All images in folder become queries with no expected answer."""
    if folder is None or not folder.is_dir():
        return []
    queries: list[tuple[Path, str]] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        for path in sorted(folder.glob(ext)):
            queries.append((path, ""))
    return queries


def format_row(rank: int, result, is_expected: bool) -> str:
    flag = "<<<" if is_expected else "   "
    return (
        f"  {rank:>2}. {flag}  "
        f"sim={result.similarity_percent:5.1f}%  "
        f"type={result.match_type:<7}  "
        f"emb={result.embedding_similarity:5.3f}  "
        f"matches={result.match_count:4}  "
        f"inliers={result.inlier_count:4}  "
        f"loc-score={result.localization_score:5.3f}  "
        f"reason={(result.rejection_reason or '-'):<28}  "
        f"{result.image.filename}"
    )


def print_query_report(
    title: str,
    query_path: Path,
    expected: str,
    outcome: SearchOutcome,
    top_n: int,
    elapsed: float,
) -> dict:
    """Print the per-query block and return summary metrics."""
    print()
    print("=" * 100)
    print(f"{title}")
    print(f"  query file       : {query_path}")
    if expected:
        print(f"  expected top-1   : {expected}")
    print(f"  query keypoints  : {outcome.query_keypoint_count}")
    print(f"  search time      : {elapsed:.2f}s")
    print(f"  candidates kept  : {len(outcome.results)}")

    top = outcome.results[:top_n]
    print(f"  top-{top_n}:")
    expected_rank: int | None = None
    expected_result = None
    for index, result in enumerate(top, start=1):
        is_expected = bool(expected) and (result.image.filename == expected)
        print(format_row(index, result, is_expected))
        if is_expected and expected_rank is None:
            expected_rank = index
            expected_result = result

    if expected and expected_rank is None:
        # search the full result list (not just the top_n cutoff)
        for index, result in enumerate(outcome.results, start=1):
            if result.image.filename == expected:
                expected_rank = index
                expected_result = result
                print(f"  expected found at rank {index} (outside top-{top_n})")
                print(format_row(index, result, True))
                break

    if expected and expected_rank is None:
        print("  expected: NOT FOUND in candidates")

    return {
        "expected_present": bool(expected),
        "expected_rank": expected_rank,
        "expected_localized": (
            bool(expected_result and expected_result.localized)
            if expected_result
            else False
        ),
        "expected_reason": (
            expected_result.rejection_reason if expected_result else None
        ),
        "top1_localized": top[0].localized if top else False,
        "top1_reason": top[0].rejection_reason if top else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("samples/library"),
        help="Root folder used to sample in-library crops",
    )
    parser.add_argument(
        "--in-library",
        type=int,
        default=5,
        help="How many in-library crops to generate for Suite A",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=None,
        help="Folder with external screenshots for Suite B (optional)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top-N candidates to print per query",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for picking in-library crops",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose engine logging",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file path. Output is also written to stdout.",
    )
    parser.add_argument(
        "--engine",
        choices=[ENGINE_ORB, ENGINE_HYBRID],
        default=ENGINE_ORB,
        help="Which retrieval pipeline to benchmark (default: orb).",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(ALLOWED_MODES),
        default=MODE_SMART,
        help="Mode for the hybrid engine (only used with --engine hybrid).",
    )
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.WARNING)

    # Force unbuffered stdout so progress is visible when piped/teed.
    # Mirror everything to args.out if given by monkey-patching builtins.print.
    import builtins
    out_file = open(args.out, "w", buffering=1) if args.out else None
    real_print = builtins.print

    def emit(*items, **kwargs):
        text = " ".join(str(x) for x in items)
        real_print(text, flush=True, **kwargs)
        if out_file is not None:
            out_file.write(text + "\n")
            out_file.flush()

    builtins.print = emit  # type: ignore[assignment]

    database = Database()
    descriptor_store = DescriptorStore()
    orb_engine = SearchEngine(
        database,
        descriptor_store,
        algorithm=ALGORITHM,
        top_k=args.top,
    )

    library_size = len(database.list_images())
    if library_size == 0:
        print("Library is empty. Run the indexer first.", file=sys.stderr)
        return 2

    engine: SearchEngine | HybridSearchEngine
    if args.engine == ENGINE_HYBRID:
        embedding_store = EmbeddingStore()
        if len(embedding_store) == 0:
            print(
                "Embedding store is empty. Run `python -m scripts.build_embeddings`"
                " before --engine hybrid.",
                file=sys.stderr,
            )
            return 2
        extractor = EmbeddingExtractor()
        engine = HybridSearchEngine(
            database=database,
            embedding_store=embedding_store,
            embedding_extractor=extractor,
            orb_engine=orb_engine,
            default_mode=args.mode,
        )
        print(
            f"Library: {library_size} images   workers={orb_engine.max_workers}   "
            f"engine=hybrid   mode={args.mode}   embeddings={len(embedding_store)}"
        )
    else:
        engine = orb_engine
        print(
            f"Library: {library_size} images   workers={orb_engine.max_workers}   "
            f"engine=orb"
        )

    print("Warming up descriptor cache...")
    warmup_start = time.perf_counter()
    engine.warmup()
    print(f"  warmup: {time.perf_counter() - warmup_start:.2f}s")

    crop_dir = Path("data/csv/debug_crops")
    in_library_queries = (
        build_in_library_queries(args.library, args.in_library, crop_dir, args.seed)
        if args.in_library > 0
        else []
    )
    external_queries = collect_external_queries(args.external_dir)

    suite_a_stats: list[dict] = []
    suite_b_stats: list[dict] = []

    if in_library_queries:
        print()
        print("#" * 100)
        print(f"# SUITE A — IN-LIBRARY CROPS (centre 50% of source) — {len(in_library_queries)} queries")
        print("#" * 100)
        for index, (query, expected) in enumerate(in_library_queries, start=1):
            start = time.perf_counter()
            outcome = engine.search_by_image(query, debug=True)
            elapsed = time.perf_counter() - start
            stats = print_query_report(
                title=f"[A.{index}] {query.name}",
                query_path=query,
                expected=expected,
                outcome=outcome,
                top_n=args.top,
                elapsed=elapsed,
            )
            suite_a_stats.append(stats)

    if external_queries:
        print()
        print("#" * 100)
        print(f"# SUITE B — EXTERNAL SCREENSHOTS — {len(external_queries)} queries")
        print("#" * 100)
        for index, (query, _) in enumerate(external_queries, start=1):
            start = time.perf_counter()
            outcome = engine.search_by_image(query, debug=True)
            elapsed = time.perf_counter() - start
            stats = print_query_report(
                title=f"[B.{index}] {query.name}",
                query_path=query,
                expected="",
                outcome=outcome,
                top_n=args.top,
                elapsed=elapsed,
            )
            suite_b_stats.append(stats)

    print()
    print("#" * 100)
    print("# SUMMARY")
    print("#" * 100)

    if suite_a_stats:
        ranks = [s["expected_rank"] for s in suite_a_stats]
        found = sum(1 for r in ranks if r is not None)
        rank1 = sum(1 for r in ranks if r == 1)
        localized = sum(1 for s in suite_a_stats if s["expected_localized"])
        reasons = Counter(
            s["expected_reason"] or "missing"
            for s in suite_a_stats
            if not s["expected_localized"]
        )
        print(f"\nSuite A (in-library) — {len(suite_a_stats)} queries:")
        print(f"  expected found in top-{args.top}        : {found}/{len(suite_a_stats)}")
        print(f"  expected at rank 1                : {rank1}/{len(suite_a_stats)}")
        print(f"  expected correctly localised      : {localized}/{len(suite_a_stats)}")
        if reasons:
            print("  rejection reasons for expected (when not localised):")
            for reason, count in reasons.most_common():
                print(f"    {reason:<32} {count}")

    if suite_b_stats:
        localized_top1 = sum(1 for s in suite_b_stats if s["top1_localized"])
        reasons = Counter(s["top1_reason"] or "missing" for s in suite_b_stats)
        print(f"\nSuite B (external) — {len(suite_b_stats)} queries:")
        print(f"  top-1 successfully localised      : {localized_top1}/{len(suite_b_stats)}")
        print("  top-1 reason distribution:")
        for reason, count in reasons.most_common():
            print(f"    {reason:<32} {count}")

    print()
    print("VERDICT GUIDE")
    print("  - Suite A miss / no inliers / no_descriptor_matches")
    print("       --> bug in ORB pipeline (style is identical)")
    print("  - Suite A passes, Suite B fails with no_descriptor_matches / too_few_matches")
    print("       --> style mismatch limitation; AI embedding hybrid would help")
    print("  - Suite B has matches but polygon_* rejections")
    print("       --> features match but geometry is distorted; AI hybrid + tuning would help")

    if out_file is not None:
        out_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
