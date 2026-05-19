"""End-to-end CLI workflow:

1. Import the given images folder into the local library.
2. Load a query fragment.
3. Run the OpenCV-based search.
4. Print top-N results with score, matches and target keypoints.
5. Open the best (or selected) match with the system image viewer.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from app.core.config import DB_PATH, DESCRIPTORS_DIR, ensure_data_dirs
from app.core.logging_config import configure_logging
from app.core.models import SearchOutcome
from app.indexing.indexer import Indexer
from app.search.search_engine import SearchEngine
from app.storage.database import Database
from app.storage.descriptor_store import DescriptorStore
from app.storage.image_store import ImageStore

logger = logging.getLogger(__name__)

ALGORITHM = "ORB"


def reset_storage() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
        logger.info("Removed database %s", DB_PATH)
    if DESCRIPTORS_DIR.exists():
        for path in DESCRIPTORS_DIR.glob("*.npz"):
            path.unlink()
        logger.info("Cleared descriptor cache %s", DESCRIPTORS_DIR)


def open_in_viewer(path: Path) -> None:
    try:
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
    except OSError as error:
        logger.warning("Could not open %s with system viewer: %s", path, error)


def print_results(outcome: SearchOutcome) -> None:
    print()
    print(f"Query keypoints: {outcome.query_keypoint_count}")
    print(f"Top {len(outcome.results)} matches:")
    header = (
        f"{'rank':>4}  {'loc':>3}  {'score':>6}  {'matches':>7}  "
        f"{'inliers':>7}  {'loc-score':>9}  filename"
    )
    print(header)
    print("-" * len(header))
    for index, result in enumerate(outcome.results, start=1):
        marker = "yes" if result.localized else "no"
        print(
            f"{index:>4}  "
            f"{marker:>3}  "
            f"{result.score:>6.3f}  "
            f"{result.match_count:>7}  "
            f"{result.inlier_count:>7}  "
            f"{result.localization_score:>9.3f}  "
            f"{result.image.filename}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("samples/library"),
        help="Folder with chart images to import",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=Path("samples/query/query.png"),
        help="Path to the fragment image used as a search query",
    )
    parser.add_argument("--top", type=int, default=10, help="Top-N matches to print")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete previous SQLite DB and descriptor cache before running",
    )
    parser.add_argument(
        "--open-best",
        action="store_true",
        help="Open the highest-scoring match with the system image viewer",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    if not args.images.is_dir():
        logger.error("Images folder does not exist: %s", args.images)
        return 2
    if not args.query.is_file():
        logger.error("Query fragment does not exist: %s", args.query)
        return 2

    ensure_data_dirs()
    if args.reset:
        reset_storage()

    database = Database()
    image_store = ImageStore()
    descriptor_store = DescriptorStore()
    indexer = Indexer(database, image_store, descriptor_store, algorithm=ALGORITHM)
    search_engine = SearchEngine(
        database,
        descriptor_store,
        algorithm=ALGORITHM,
        top_k=args.top,
    )

    logger.info("=== Step 1: import folder ===")
    import_result = indexer.import_folder(
        args.images,
        progress=lambda i, t, n: logger.debug("[%d/%d] %s", i, t, n),
    )
    logger.info(
        "Import done: imported=%d skipped=%d",
        import_result.imported,
        import_result.skipped,
    )

    logger.info("=== Step 2-3: search by fragment ===")
    outcome = search_engine.search_by_image(args.query)

    logger.info("=== Step 4: top results ===")
    print_results(outcome)

    if not outcome.results:
        logger.warning("No matches found")
        return 1

    best = outcome.results[0]
    logger.info(
        "Best match: %s score=%.3f matches=%d inliers=%d loc-score=%.3f localized=%s",
        best.image.filename,
        best.score,
        best.match_count,
        best.inlier_count,
        best.localization_score,
        best.localized,
    )
    if best.localized and best.polygon is not None:
        logger.info(
            "Bounding polygon (target coords): %s",
            best.polygon.round(1).tolist(),
        )

    if args.open_best:
        logger.info("Opening best match in system viewer")
        open_in_viewer(best.image.stored_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
