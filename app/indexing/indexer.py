import logging
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from app.core.models import ImportResult
from app.storage.database import Database
from app.storage.descriptor_store import DescriptorStore
from app.storage.embedding_store import EmbeddingStore
from app.storage.image_store import ImageStore
from app.vision.embedding_extractor import EmbeddingExtractor
from app.vision.opencv_features import FeatureExtractor

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[int, int, str], None]


class Indexer:
    def __init__(
        self,
        database: Database,
        image_store: ImageStore,
        descriptor_store: DescriptorStore,
        algorithm: str = "ORB",
        embedding_extractor: EmbeddingExtractor | None = None,
        embedding_store: EmbeddingStore | None = None,
        embedding_save_every: int = 50,
    ) -> None:
        self.database = database
        self.image_store = image_store
        self.descriptor_store = descriptor_store
        self.algorithm = algorithm
        self.extractor = FeatureExtractor(algorithm=algorithm)
        self.embedding_extractor = embedding_extractor
        self.embedding_store = embedding_store
        self.embedding_save_every = max(1, embedding_save_every)
        self._embedding_dirty_count = 0

    def import_folder(
        self,
        folder: Path,
        progress: ProgressCallback | None = None,
    ) -> ImportResult:
        paths = self.image_store.iter_images(folder)
        total = len(paths)
        logger.info("Importing folder %s (%d candidate images)", folder, total)

        imported = 0
        skipped = 0

        for index, source_path in enumerate(paths, start=1):
            if progress is not None:
                progress(index, total, source_path.name)
            logger.debug("[%d/%d] processing %s", index, total, source_path.name)

            try:
                image_id, was_new = self._import_single(source_path)
            except (OSError, ValueError, Image.UnidentifiedImageError) as error:
                logger.warning("Skipping %s: %s", source_path.name, error)
                skipped += 1
                continue

            if was_new:
                imported += 1
                self._index_image(image_id, source_path)
            else:
                logger.debug("Skipping duplicate %s (id=%d)", source_path.name, image_id)
                skipped += 1

        if self.embedding_store is not None and self._embedding_dirty_count:
            self.embedding_store.save()
            self._embedding_dirty_count = 0

        logger.info(
            "Import finished: imported=%d skipped=%d total=%d",
            imported,
            skipped,
            total,
        )
        return ImportResult(imported=imported, skipped=skipped)

    def _import_single(self, source_path: Path) -> tuple[int, bool]:
        stored_path, thumbnail_path, width, height, file_hash = (
            self.image_store.import_image(source_path)
        )
        return self.database.upsert_image(
            original_path=source_path,
            stored_path=stored_path,
            thumbnail_path=thumbnail_path,
            filename=source_path.name,
            width=width,
            height=height,
            file_hash=file_hash,
        )

    def _index_image(self, image_id: int, source_path: Path) -> None:
        keypoints, descriptors = self.extractor.extract_from_path(source_path)
        descriptor_path = self.descriptor_store.save(
            image_id=image_id,
            algorithm=self.algorithm,
            keypoints=keypoints,
            descriptors=descriptors,
        )
        keypoint_count = int(keypoints.shape[0]) if keypoints is not None else 0
        self.database.update_descriptor(
            image_id,
            descriptor_path=descriptor_path,
            algorithm=self.algorithm,
            keypoint_count=keypoint_count,
        )
        logger.info(
            "Indexed id=%d %s keypoints=%d descriptor=%s",
            image_id,
            source_path.name,
            keypoint_count,
            descriptor_path.name if descriptor_path else "<none>",
        )
        self._maybe_embed(image_id, source_path)

    def _maybe_embed(self, image_id: int, source_path: Path) -> None:
        if self.embedding_extractor is None or self.embedding_store is None:
            return
        if self.embedding_store.has(image_id):
            return

        try:
            # Indexing uses the image as-is — the canonical landscape charts
            # already match the model's expected aspect, and query-side
            # letterboxing handles odd-shaped screenshots later.
            vector = self.embedding_extractor.extract_from_path(
                source_path, normalize_aspect=False
            )
        except Exception:
            logger.exception(
                "Embedding failed for id=%d %s", image_id, source_path.name
            )
            return

        self.embedding_store.append(image_id, np.asarray(vector, dtype=np.float32))
        self._embedding_dirty_count += 1
        if self._embedding_dirty_count >= self.embedding_save_every:
            self.embedding_store.save()
            self._embedding_dirty_count = 0
