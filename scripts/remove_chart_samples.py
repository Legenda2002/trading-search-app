"""One-off cleanup: remove the ``chart_NNN.png`` synthetic samples from the
index, including their copies on disk, thumbnails, ORB descriptors and
DINOv2 embeddings.

Usage::

    python -m scripts.remove_chart_samples [--dry-run] [--prefix chart_]

The script is idempotent — re-running it after the first pass is a no-op.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

from app.core.config import DB_PATH, EMBEDDINGS_PATH, ensure_data_dirs
from app.storage.database import Database
from app.storage.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="chart_",
        help="Filename prefix to match (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed, don't touch the disk or DB",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_data_dirs()

    database = Database()
    targets = [img for img in database.list_images() if img.filename.startswith(args.prefix)]
    if not targets:
        logger.info("Nothing matches prefix %r — index is already clean", args.prefix)
        return 0

    logger.info("Found %d images matching prefix %r:", len(targets), args.prefix)
    for img in targets:
        logger.info("  id=%d  %s", img.id, img.filename)

    if args.dry_run:
        logger.info("--dry-run set: no changes applied")
        return 0

    removed_ids = [int(img.id) for img in targets]

    # --- 1. Drop files associated with each row ---------------------------
    paths_to_remove: list[Path] = []
    for img in targets:
        for candidate in (img.stored_path, img.thumbnail_path, img.descriptor_path):
            if candidate is None:
                continue
            paths_to_remove.append(Path(candidate))

    deleted = 0
    for path in paths_to_remove:
        try:
            if path.exists():
                path.unlink()
                deleted += 1
        except OSError as error:
            logger.warning("Failed to remove %s: %s", path, error)
    logger.info("Deleted %d files from disk", deleted)

    # --- 2. Drop SQLite rows ----------------------------------------------
    placeholders = ",".join("?" * len(removed_ids))
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            f"DELETE FROM images WHERE id IN ({placeholders})",
            removed_ids,
        )
        logger.info("Removed %d rows from images table", cursor.rowcount)

    # --- 3. Rebuild embeddings.npz without those IDs -----------------------
    store = EmbeddingStore(path=EMBEDDINGS_PATH)
    ids, vectors = store.load_all()
    if ids.size == 0:
        logger.info("Embedding store is empty, nothing to filter")
        return 0

    keep_mask = ~np.isin(ids, np.array(removed_ids, dtype=np.int64))
    if keep_mask.all():
        logger.info("No embeddings matched removed IDs — store unchanged")
        return 0

    new_ids = ids[keep_mask]
    new_vectors = vectors[keep_mask]
    tmp_path = EMBEDDINGS_PATH.with_name(
        EMBEDDINGS_PATH.stem + ".tmp" + EMBEDDINGS_PATH.suffix
    )
    np.savez(str(tmp_path), ids=new_ids, vectors=new_vectors)
    tmp_path.replace(EMBEDDINGS_PATH)
    logger.info(
        "Rebuilt embedding store: %d -> %d vectors (removed %d)",
        len(ids),
        len(new_ids),
        len(ids) - len(new_ids),
    )

    logger.info("Cleanup done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
