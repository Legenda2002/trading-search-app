"""Rewrite absolute paths inside the SQLite index after moving the project.

When you copy ``trading-search-app/`` from one machine to another (e.g. Linux →
Windows, or to a different user folder), the paths stored in ``data/app.db``
still point at the old machine and the app can't find the images / ORB
descriptors / thumbnails it indexed. This script fixes that in-place.

Usage::

    # Activate your venv first
    python -m scripts.migrate_paths

It auto-detects the new project root (= the folder containing this script's
parent) and rewrites every ``original_path``, ``stored_path``,
``thumbnail_path`` and ``descriptor_path`` row in the ``images`` table so they
point inside the *current* project. The DB is backed up to ``app.db.bak`` next
to itself before any changes are made.

Idempotent — running it twice is safe (the second run finds no rows to update).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

from app.core.config import DB_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)


def _detect_old_roots(db_path: Path) -> set[str]:
    """Look at a few existing rows and infer the old project root prefixes."""
    roots: set[str] = set()
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT stored_path, thumbnail_path, descriptor_path FROM images LIMIT 200"
        )
        for row in cur.fetchall():
            for value in row:
                if not value:
                    continue
                marker = "/data/"
                idx = value.find(marker)
                if idx > 0:
                    roots.add(value[:idx])
                    continue
                marker_win = "\\data\\"
                idx = value.find(marker_win)
                if idx > 0:
                    roots.add(value[:idx])
    return roots


def _rewrite(value: str | None, old_roots: set[str], new_root: str) -> str | None:
    if not value:
        return value
    for old in old_roots:
        if value.startswith(old):
            return new_root + value[len(old):]
    return value


def migrate(db_path: Path, new_root: Path, *, dry_run: bool = False) -> int:
    if not db_path.exists():
        logger.error("DB not found at %s", db_path)
        return 0

    old_roots = _detect_old_roots(db_path)
    if not old_roots:
        logger.info("No old-root prefixes detected — DB looks already migrated.")
        return 0

    new_root_str = str(new_root)
    logger.info("Detected old roots: %s", sorted(old_roots))
    logger.info("Rewriting to new root: %s", new_root_str)

    if not dry_run:
        backup = db_path.with_suffix(db_path.suffix + ".bak")
        shutil.copy2(db_path, backup)
        logger.info("Backup created: %s", backup)

    updated = 0
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT id, original_path, stored_path, thumbnail_path, descriptor_path FROM images"
        )
        rows = cur.fetchall()
        for image_id, orig, stored, thumb, desc in rows:
            new_orig = _rewrite(orig, old_roots, new_root_str)
            new_stored = _rewrite(stored, old_roots, new_root_str)
            new_thumb = _rewrite(thumb, old_roots, new_root_str)
            new_desc = _rewrite(desc, old_roots, new_root_str)
            if (new_orig, new_stored, new_thumb, new_desc) == (orig, stored, thumb, desc):
                continue
            updated += 1
            if not dry_run:
                cur.execute(
                    "UPDATE images SET original_path=?, stored_path=?, thumbnail_path=?, descriptor_path=? WHERE id=?",
                    (new_orig, new_stored, new_thumb, new_desc, image_id),
                )
        if not dry_run:
            con.commit()

    logger.info("Updated %d/%d rows", updated, len(rows))
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite DB (default: %(default)s)",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Destination project root (default: auto-detected from this file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    new_root = args.new_root.resolve()
    if not new_root.exists():
        logger.error("New root does not exist: %s", new_root)
        return 1

    migrate(args.db, new_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
