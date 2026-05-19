import sqlite3
from pathlib import Path

from app.core.config import DB_PATH, ensure_data_dirs
from app.core.models import ChartImage


class Database:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        ensure_data_dirs()
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_path TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    thumbnail_path TEXT,
                    filename TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    descriptor_path TEXT,
                    algorithm TEXT,
                    keypoint_count INTEGER NOT NULL DEFAULT 0,
                    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash)"
            )

    def upsert_image(
        self,
        *,
        original_path: Path,
        stored_path: Path,
        thumbnail_path: Path | None,
        filename: str,
        width: int,
        height: int,
        file_hash: str,
    ) -> tuple[int, bool]:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM images WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
            if existing:
                return int(existing["id"]), False

            cursor = connection.execute(
                """
                INSERT INTO images (
                    original_path,
                    stored_path,
                    thumbnail_path,
                    filename,
                    width,
                    height,
                    file_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(original_path),
                    str(stored_path),
                    str(thumbnail_path) if thumbnail_path else None,
                    filename,
                    width,
                    height,
                    file_hash,
                ),
            )
            return int(cursor.lastrowid), True

    def update_descriptor(
        self,
        image_id: int,
        *,
        descriptor_path: Path | None,
        algorithm: str,
        keypoint_count: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE images
                SET descriptor_path = ?, algorithm = ?, keypoint_count = ?
                WHERE id = ?
                """,
                (
                    str(descriptor_path) if descriptor_path else None,
                    algorithm,
                    keypoint_count,
                    image_id,
                ),
            )

    def list_images(self) -> list[ChartImage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM images
                ORDER BY imported_at DESC, id DESC
                """
            ).fetchall()
        return [self._row_to_image(row) for row in rows]

    def get_image(self, image_id: int) -> ChartImage | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM images WHERE id = ?",
                (image_id,),
            ).fetchone()
        return self._row_to_image(row) if row else None

    @staticmethod
    def _row_to_image(row: sqlite3.Row) -> ChartImage:
        return ChartImage(
            id=int(row["id"]),
            original_path=Path(row["original_path"]),
            stored_path=Path(row["stored_path"]),
            thumbnail_path=Path(row["thumbnail_path"]) if row["thumbnail_path"] else None,
            filename=str(row["filename"]),
            width=int(row["width"]),
            height=int(row["height"]),
            file_hash=str(row["file_hash"]),
            descriptor_path=Path(row["descriptor_path"]) if row["descriptor_path"] else None,
            algorithm=str(row["algorithm"]) if row["algorithm"] else None,
            keypoint_count=int(row["keypoint_count"]),
        )
