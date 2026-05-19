"""Backfill DINOv2 embeddings for every image currently in the library.

Run once after upgrading to Phase 2A. Subsequent imports compute embeddings
automatically as part of indexing. Re-running this script is safe — already
computed embeddings are skipped.

Usage:
    venv/bin/python -m scripts.build_embeddings
    venv/bin/python -m scripts.build_embeddings --batch-size 32 --rebuild
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import EMBEDDINGS_PATH, ensure_data_dirs
from app.core.logging_config import configure_logging
from app.core.models import ChartImage
from app.storage.database import Database
from app.storage.embedding_store import EmbeddingStore
from app.vision.embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)


def _load_image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as pil:
            return np.asarray(pil.convert("RGB"))
    except (OSError, ValueError) as error:
        logger.warning("Skipping %s: %s", path, error)
        return None


def _iter_pending(
    images: list[ChartImage],
    known_ids: set[int],
    rebuild: bool,
) -> list[ChartImage]:
    if rebuild:
        return [img for img in images if img.stored_path.exists()]
    return [
        img
        for img in images
        if int(img.id) not in known_ids and img.stored_path.exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images encoded per forward pass (default: 16)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute embeddings for every image, ignoring the existing store.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N images (useful for smoke tests). 0 = no limit.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="How often to log progress lines (default: every 500 images).",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_data_dirs()

    database = Database()
    store = EmbeddingStore(path=EMBEDDINGS_PATH)
    extractor = EmbeddingExtractor()

    library = database.list_images()
    known_ids = store.known_ids()
    if args.rebuild:
        logger.info("Rebuild requested: re-encoding %d images", len(library))
        known_ids = set()
    pending = _iter_pending(library, known_ids, args.rebuild)
    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        logger.info(
            "Nothing to do: library has %d images, %d already embedded",
            len(library),
            len(store),
        )
        return 0

    logger.info(
        "Embedding %d images (library=%d, already=%d, batch_size=%d)",
        len(pending),
        len(library),
        len(store),
        args.batch_size,
    )

    start_time = time.time()
    last_log = start_time
    last_save = start_time
    batch_ids: list[int] = []
    batch_arrays: list[np.ndarray] = []
    processed = 0
    saved_since = 0

    def _flush_batch() -> None:
        nonlocal saved_since
        if not batch_arrays:
            return
        try:
            vectors = extractor.extract_batch(batch_arrays, batch_size=args.batch_size)
        except Exception:
            logger.exception("Batch encoding failed for ids=%s", batch_ids)
            batch_ids.clear()
            batch_arrays.clear()
            return
        store.extend(np.asarray(batch_ids, dtype=np.int64), vectors)
        saved_since += len(batch_ids)
        batch_ids.clear()
        batch_arrays.clear()

    try:
        for image in pending:
            array = _load_image(image.stored_path)
            if array is None:
                continue
            batch_ids.append(int(image.id))
            batch_arrays.append(array)
            processed += 1
            if len(batch_arrays) >= args.batch_size:
                _flush_batch()

            now = time.time()
            if processed % args.log_every == 0 or now - last_log > 30:
                elapsed = now - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = (len(pending) - processed) / rate if rate else 0.0
                logger.info(
                    "Progress %d/%d (%.1f imgs/s, eta %.1f min)",
                    processed,
                    len(pending),
                    rate,
                    remaining / 60.0,
                )
                last_log = now

            # Periodic checkpoint so a crash doesn't lose hours of work.
            if saved_since >= 500 or now - last_save > 120:
                _flush_batch()
                store.save()
                saved_since = 0
                last_save = time.time()

        _flush_batch()
        store.save()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, flushing progress to disk")
        _flush_batch()
        store.save()
        return 130

    total = time.time() - start_time
    logger.info(
        "Done: %d new embeddings in %.1f min, store now has %d entries",
        processed,
        total / 60.0,
        len(store),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
