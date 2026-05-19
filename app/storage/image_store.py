import hashlib
import shutil
from pathlib import Path

from PIL import Image

from app.core.config import ORIGINALS_DIR, SUPPORTED_IMAGE_EXTENSIONS, THUMBNAILS_DIR


class ImageStore:
    def is_supported_image(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    def iter_images(self, folder: Path) -> list[Path]:
        return sorted(
            path for path in folder.rglob("*") if self.is_supported_image(path)
        )

    def import_image(self, source_path: Path) -> tuple[Path, Path | None, int, int, str]:
        file_hash = self._sha256(source_path)
        stored_path = self._copy_original(source_path, file_hash)
        thumbnail_path, width, height = self._create_thumbnail(stored_path, file_hash)
        return stored_path, thumbnail_path, width, height, file_hash

    def _copy_original(self, source_path: Path, file_hash: str) -> Path:
        target_name = f"{file_hash[:16]}{source_path.suffix.lower()}"
        target_path = ORIGINALS_DIR / target_name
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        return target_path

    def _create_thumbnail(
        self,
        source_path: Path,
        file_hash: str,
        max_size: tuple[int, int] = (320, 220),
    ) -> tuple[Path | None, int, int]:
        try:
            with Image.open(source_path) as image:
                width, height = image.size
                image.thumbnail(max_size)
                thumbnail_path = THUMBNAILS_DIR / f"{file_hash[:16]}.jpg"
                image.convert("RGB").save(thumbnail_path, "JPEG", quality=85)
                return thumbnail_path, width, height
        except OSError:
            return None, 0, 0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
