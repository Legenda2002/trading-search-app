"""Benchmark search latency on the populated library.

Measures:
  1. Warmup time (parallel .npz load of the whole descriptor cache).
  2. Per-query latency for a list of fragment images (cache hit).

Usage:
  python -m scripts.benchmark_search --queries data/csv/xauusd_m15_fragment.png \
                                     data/csv/xauusd_real_fragment.png
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from app.core.logging_config import configure_logging
from app.search.search_engine import SearchEngine
from app.storage.database import Database
from app.storage.descriptor_store import DescriptorStore

logger = logging.getLogger(__name__)

ALGORITHM = "ORB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        nargs="+",
        required=True,
        help="Fragment images to use as queries",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    database = Database()
    descriptor_store = DescriptorStore()
    engine = SearchEngine(
        database,
        descriptor_store,
        algorithm=ALGORITHM,
        top_k=args.top,
    )

    library_size = len(database.list_images())
    print(f"\nLibrary: {library_size} images, workers={engine.max_workers}\n")

    print("Warming up descriptor cache...")
    warmup_start = time.perf_counter()
    cached = engine.warmup()
    warmup_secs = time.perf_counter() - warmup_start
    print(f"  Cached {cached} descriptor sets in {warmup_secs:.2f}s\n")

    for query_path in args.queries:
        if not query_path.is_file():
            print(f"  [skip] {query_path} not found")
            continue

        print(f"=== query: {query_path.name} ===")
        for trial in range(2):
            start = time.perf_counter()
            outcome = engine.search_by_image(query_path)
            elapsed = time.perf_counter() - start
            if not outcome.results:
                print(f"  trial {trial + 1}: {elapsed:.2f}s -- no results")
                continue
            best = outcome.results[0]
            print(
                f"  trial {trial + 1}: {elapsed:.2f}s   "
                f"top-1={best.image.filename}   "
                f"score={best.score:.3f}   "
                f"matches={best.match_count}   "
                f"inliers={best.inlier_count}   "
                f"loc-score={best.localization_score:.3f}   "
                f"localized={'yes' if best.localized else 'no'}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
