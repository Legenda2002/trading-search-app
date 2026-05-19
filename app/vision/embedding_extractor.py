"""DINOv2 embedding extractor for visual similarity retrieval.

Loads ``facebook/dinov2-small`` from HuggingFace once, then produces
L2-normalised 384-dim feature vectors. CPU only — model weights are ~85 MB
and one forward pass on a 224×224 chart takes ~70 ms on a 10-core laptop.

Thread safety:
* The model itself is read-only once loaded, but ``torch.nn.Module`` ops
  share intermediate buffers, so concurrent forward() calls from multiple
  threads are not safe. We serialise inference with an internal lock.
* For batch indexing we expose ``extract_batch`` which processes the entire
  batch under a single lock acquire.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from app.core.config import (
    BUNDLED_HF_CACHE_DIR,
    DINOV2_MODEL_NAME,
    EMBEDDING_DIM,
    HF_CACHE_DIR,
    QUERY_LETTERBOX_ASPECT,
    QUERY_LETTERBOX_FILL,
    ensure_data_dirs,
)

logger = logging.getLogger(__name__)


def _configure_hf_env(cache_dir: Path) -> None:
    """Point HuggingFace to a project-local cache and run offline if cached.

    We try offline first; on a cache miss the caller can retry with the
    environment unmodified (or fall back to the global HF cache). This keeps
    repeated runs fast and avoids surprise network hits at search time.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _letterbox_to_aspect(
    image: np.ndarray,
    target_aspect: float = QUERY_LETTERBOX_ASPECT,
    fill: int = QUERY_LETTERBOX_FILL,
    tolerance: float = 0.05,
) -> np.ndarray:
    """Pad ``image`` symmetrically so its width/height ratio == target.

    Empirically this single transform recovers ~75x recall on portrait
    screenshots of landscape charts: with raw input the true source falls
    to rank ~4790 in a 17k library; padded to the canonical 16:9 aspect it
    bounces back to rank ~60, well inside the 2000-candidate shortlist.

    The padding colour matches the dark grey background most candlestick
    renderers already use, so DINOv2 does not get fooled into thinking the
    bars are part of the price action.
    """
    if image is None or image.size == 0:
        return image
    if image.ndim < 2:
        return image
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    current = w / h
    if abs(current - target_aspect) < tolerance:
        return image

    if current < target_aspect:
        new_w = int(round(h * target_aspect))
        pad_total = max(0, new_w - w)
        left = pad_total // 2
        right = pad_total - left
        pad_widths = [(0, 0), (left, right)]
    else:
        new_h = int(round(w / target_aspect))
        pad_total = max(0, new_h - h)
        top = pad_total // 2
        bottom = pad_total - top
        pad_widths = [(top, bottom), (0, 0)]
    if image.ndim == 3:
        pad_widths.append((0, 0))
    return np.pad(image, pad_widths, mode="constant", constant_values=fill)


def _seed_cache_from_bundle(cache_dir: Path) -> None:
    """Populate the user cache from a read-only bundled cache on first run.

    PyInstaller ships ``data/hf_cache/`` alongside the executable so the
    desktop client doesn't need to download ~85 MB of DINOv2 weights on first
    launch. We mirror the bundle into the user-writable cache only if it
    actually contains model files we don't have yet.
    """
    if BUNDLED_HF_CACHE_DIR is None:
        return
    if not BUNDLED_HF_CACHE_DIR.is_dir():
        return

    import shutil  # noqa: WPS433 — local import keeps cold-start cheap

    for src in BUNDLED_HF_CACHE_DIR.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(BUNDLED_HF_CACHE_DIR)
        dst = cache_dir / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError as error:
            logger.warning("Failed to seed %s from bundle: %s", rel, error)
    logger.info("HF cache seeded from bundle at %s", BUNDLED_HF_CACHE_DIR)


class EmbeddingExtractor:
    def __init__(
        self,
        model_name: str = DINOV2_MODEL_NAME,
        cache_dir: Path = HF_CACHE_DIR,
        device: str = "cpu",
    ) -> None:
        ensure_data_dirs()
        _configure_hf_env(cache_dir)
        _seed_cache_from_bundle(cache_dir)

        # Imports are deferred so the rest of the app still starts even if
        # torch is missing (e.g. someone running scripts that don't need
        # embeddings).
        import torch  # noqa: WPS433
        from transformers import AutoImageProcessor, AutoModel  # noqa: WPS433

        self._torch = torch
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        self._lock = threading.Lock()

        logger.info("Loading DINOv2 model %s on %s", model_name, self.device)
        # ``backend="torchvision"`` selects the fast image processor and
        # silences the legacy ``use_fast`` deprecation warning. We try the
        # local cache first because the sandbox often blocks HuggingFace and
        # falling through to network would also block on a long timeout.
        self.processor, self.model = self._load_model(model_name)
        self.model.eval()
        self.model.to(self.device)

        self.dim = int(getattr(self.model.config, "hidden_size", EMBEDDING_DIM))
        if self.dim != EMBEDDING_DIM:
            logger.warning(
                "Model hidden size %d differs from configured EMBEDDING_DIM %d",
                self.dim,
                EMBEDDING_DIM,
            )

    def _load_model(self, model_name: str):
        from transformers import AutoImageProcessor, AutoModel  # noqa: WPS433

        # First try strictly local — fast and never touches the network.
        try:
            processor = AutoImageProcessor.from_pretrained(
                model_name,
                backend="torchvision",
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(model_name, local_files_only=True)
            logger.info("Loaded %s from local cache", model_name)
            return processor, model
        except (OSError, ValueError) as error:
            logger.info(
                "Local cache miss for %s (%s), retrying with network",
                model_name,
                error.__class__.__name__,
            )

        processor = AutoImageProcessor.from_pretrained(
            model_name,
            backend="torchvision",
        )
        model = AutoModel.from_pretrained(model_name)
        logger.info("Downloaded %s from HuggingFace Hub", model_name)
        return processor, model

    def warmup(self) -> None:
        """Run a tiny forward pass to JIT-init torch's CPU kernels."""
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self.extract_from_image(dummy)
            logger.info("EmbeddingExtractor warmup done")
        except Exception:
            logger.exception("EmbeddingExtractor warmup failed")

    def _preprocess(self, images: Sequence[np.ndarray]):
        rgb_images: list[np.ndarray] = []
        for img in images:
            if img is None or img.size == 0:
                raise ValueError("Empty image passed to embedding extractor")
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            # Force a writable, contiguous copy — torch.from_numpy emits a
            # noisy warning every time we hand it a read-only PIL array.
            rgb_images.append(np.array(img, dtype=np.uint8, copy=True))
        return self.processor(images=rgb_images, return_tensors="pt")

    def _forward(self, inputs) -> np.ndarray:
        torch = self._torch
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            cls = outputs.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls.detach().cpu().numpy().astype(np.float32, copy=False)

    def extract_from_image(
        self,
        image: np.ndarray,
        *,
        normalize_aspect: bool = True,
    ) -> np.ndarray:
        """Return a single L2-normalised float32 vector of length ``self.dim``.

        ``normalize_aspect`` letterbox-pads the input to the library's
        canonical 16:9 aspect ratio before DINOv2 sees it. Enabled by
        default for queries — see ``_letterbox_to_aspect`` for the
        rationale. Indexing paths flip it off so library embeddings stay
        exactly as they were when the store was built.
        """
        prepared = _letterbox_to_aspect(image) if normalize_aspect else image
        with self._lock:
            inputs = self._preprocess([prepared])
            vecs = self._forward(inputs)
        return vecs[0]

    def extract_from_path(
        self,
        path: Path,
        *,
        normalize_aspect: bool = True,
    ) -> np.ndarray:
        from PIL import Image  # noqa: WPS433

        with Image.open(path) as pil_image:
            pil_image = pil_image.convert("RGB")
            array = np.asarray(pil_image)
        return self.extract_from_image(array, normalize_aspect=normalize_aspect)

    def extract_batch(
        self,
        images: Iterable[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Encode an iterable of HWC numpy images. Returns ``(N, dim)`` float32."""
        buffer: list[np.ndarray] = []
        out_chunks: list[np.ndarray] = []
        with self._lock:
            for img in images:
                buffer.append(img)
                if len(buffer) >= batch_size:
                    inputs = self._preprocess(buffer)
                    out_chunks.append(self._forward(inputs))
                    buffer.clear()
            if buffer:
                inputs = self._preprocess(buffer)
                out_chunks.append(self._forward(inputs))
        if not out_chunks:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(out_chunks, axis=0)

    def extract_paths(
        self,
        paths: Sequence[Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Convenience wrapper that loads paths with PIL and batch-encodes them."""
        from PIL import Image  # noqa: WPS433

        def _iter():
            for path in paths:
                with Image.open(path) as pil_image:
                    pil_image = pil_image.convert("RGB")
                    yield np.asarray(pil_image)

        return self.extract_batch(_iter(), batch_size=batch_size)
