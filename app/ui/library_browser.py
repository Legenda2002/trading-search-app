"""Modal dialog showing every chart currently in the local index.

For a 17k-image library a naive QListWidget that pre-loads thumbnails up
front freezes the UI for ~30 s. Instead we expose the library through a
``QAbstractListModel`` whose ``data(role=DecorationRole)`` is the only place
that touches disk. Qt only asks for the icon of *visible* rows, so the
panel paints instantly and progressively populates as the user scrolls.

Filtering goes through a ``QSortFilterProxyModel`` so we keep the icon
cache in the source model and never re-decode the same JPEG twice.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.models import ChartImage
from app.storage.database import Database

logger = logging.getLogger(__name__)


THUMB_SIZE = QSize(180, 120)


def _make_placeholder_icon() -> QIcon:
    """Solid-colour tile painted instead of an unread thumbnail."""
    pm = QPixmap(THUMB_SIZE)
    pm.fill(QColor(38, 44, 56))
    return QIcon(pm)


class LibraryModel(QAbstractListModel):
    """Virtual model: decodes thumbnails off the UI thread.

    ``data(DecorationRole)`` returns a placeholder immediately and queues a
    background decode if the row is not cached yet. A worker pool reads the
    JPEG, scales it, and emits ``icon_ready`` which we splice back into the
    cache on the GUI thread; the model then emits ``dataChanged`` to tell
    the view to repaint that single row.
    """

    icon_ready = Signal(int, QIcon)

    def __init__(self, images: list[ChartImage], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._images: list[ChartImage] = images
        self._icon_cache: dict[int, QIcon] = {}
        self._pending: set[int] = set()
        self._lock = threading.Lock()
        self._placeholder = _make_placeholder_icon()
        # Decoding JPEGs is bound by libjpeg, not Python, so a handful of
        # worker threads is enough to keep the visible rows up to date.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lib-thumb")
        self.icon_ready.connect(self._on_icon_ready)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._images)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = index.row()
        if not 0 <= row < len(self._images):
            return None
        image = self._images[row]

        if role == Qt.DisplayRole:
            return image.filename
        if role == Qt.DecorationRole:
            cached = self._icon_cache.get(row)
            if cached is not None:
                return cached
            self._schedule_load(row, image)
            return self._placeholder
        if role == Qt.ToolTipRole:
            return (
                f"{image.filename}\n"
                f"{image.width}×{image.height}\n"
                f"Ключевых точек: {image.keypoint_count}"
            )
        if role == Qt.UserRole:
            return str(image.stored_path)
        if role == Qt.UserRole + 1:
            # Pre-lowercased filename for the proxy filter — much cheaper
            # than calling .lower() on every keystroke for 17k rows.
            return image.filename.lower()
        return None

    def _schedule_load(self, row: int, image: ChartImage) -> None:
        with self._lock:
            if row in self._pending:
                return
            self._pending.add(row)
        thumb_path = str(image.thumbnail_path or image.stored_path)
        self._executor.submit(self._decode_in_background, row, thumb_path)

    def _decode_in_background(self, row: int, thumb_path: str) -> None:
        try:
            pixmap = QPixmap(thumb_path)
            if pixmap.isNull():
                icon = self._placeholder
            else:
                scaled = pixmap.scaled(
                    THUMB_SIZE,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                icon = QIcon(scaled)
        except Exception:  # noqa: BLE001 — defensive: never kill the pool
            logger.exception("Failed to decode thumbnail %s", thumb_path)
            icon = self._placeholder
        # Marshal back to the GUI thread; Qt forbids mutating models from
        # worker threads.
        self.icon_ready.emit(row, icon)

    def _on_icon_ready(self, row: int, icon: QIcon) -> None:
        with self._lock:
            self._pending.discard(row)
        self._icon_cache[row] = icon
        if 0 <= row < len(self._images):
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, [Qt.DecorationRole])

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class _FilenameFilterProxy(QSortFilterProxyModel):
    """Filters by substring against the pre-lowercased filename role."""

    NAME_ROLE = Qt.UserRole + 1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._needle = ""

    def set_needle(self, text: str) -> None:
        self._needle = text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if not self._needle:
            return True
        index = self.sourceModel().index(source_row, 0, source_parent)
        name = self.sourceModel().data(index, self.NAME_ROLE) or ""
        return self._needle in name


class LibraryBrowserDialog(QDialog):
    """Grid-style browser of every chart in the SQLite index."""

    image_chosen = Signal(Path)

    def __init__(self, database: Database, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("База графиков")
        self.resize(1100, 720)
        self.setModal(True)

        self.database = database

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("libraryLabel")
        top_row.addWidget(self.summary_label)

        top_row.addStretch(1)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Фильтр по имени файла…")
        self.search_field.setClearButtonEnabled(True)
        self.search_field.setMinimumWidth(280)
        # Debounce the filter so we don't run a 17k-row scan on every
        # keystroke. The 200 ms window feels instant but coalesces fast
        # typing into a single pass.
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(200)
        self._filter_timer.timeout.connect(self._apply_pending_filter)
        self.search_field.textChanged.connect(self._on_filter_text_changed)
        top_row.addWidget(self.search_field)

        layout.addLayout(top_row)

        # Build the model with the full library — it's just a list of small
        # dataclasses, ~17k entries cost ~3 MB and load instantly. Decoding
        # the JPEG thumbnails is what was slow, and that now happens
        # lazily inside LibraryModel.data().
        images = database.list_images()
        self.source_model = LibraryModel(images, parent=self)
        self.proxy_model = _FilenameFilterProxy(self)
        self.proxy_model.setSourceModel(self.source_model)

        self.list_view = QListView()
        self.list_view.setModel(self.proxy_model)
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setIconSize(THUMB_SIZE)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setMovement(QListView.Static)
        self.list_view.setSpacing(8)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setWordWrap(True)
        # A grid size larger than the icon avoids reflow churn while
        # scrolling and gives each cell room for the filename underneath.
        self.list_view.setGridSize(
            QSize(THUMB_SIZE.width() + 24, THUMB_SIZE.height() + 56)
        )
        # Per-pixel scrolling keeps the lazy decoder demand-driven rather
        # than spiking on each wheel notch.
        self.list_view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list_view.doubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self.list_view, stretch=1)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch(1)

        self.open_button = QPushButton("Открыть в поиске")
        self.open_button.setObjectName("primary")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._on_open_clicked)
        bottom_row.addWidget(self.open_button)

        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.reject)
        bottom_row.addWidget(self.close_button)

        layout.addLayout(bottom_row)

        selection_model = self.list_view.selectionModel()
        selection_model.selectionChanged.connect(self._refresh_open_button)

        self._refresh_summary()
        logger.info("Library browser opened with %d images", len(images))

    def _on_filter_text_changed(self, _text: str) -> None:
        self._filter_timer.start()

    def _apply_pending_filter(self) -> None:
        self.proxy_model.set_needle(self.search_field.text())
        self._refresh_summary()
        self._refresh_open_button()

    def _refresh_summary(self) -> None:
        total = self.source_model.rowCount()
        shown = self.proxy_model.rowCount()
        if shown == total:
            self.summary_label.setText(f"Всего в базе: {total} графиков")
        else:
            self.summary_label.setText(
                f"Показано: {shown} из {total} (фильтр: «{self.search_field.text()}»)"
            )

    def _refresh_open_button(self) -> None:
        self.open_button.setEnabled(self.list_view.selectionModel().hasSelection())

    def _on_double_clicked(self, index: QModelIndex) -> None:
        self._emit_choice(index)

    def _on_open_clicked(self) -> None:
        indexes = self.list_view.selectionModel().selectedIndexes()
        if not indexes:
            return
        self._emit_choice(indexes[0])

    def _emit_choice(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        path = self.proxy_model.data(index, Qt.UserRole)
        if not path:
            return
        self.image_chosen.emit(Path(path))
        self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self.source_model.shutdown()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        # Triggered by both accept() and reject(); make sure the thread
        # pool stops accepting work so the parent shutdown is instant.
        self.source_model.shutdown()
        super().done(result)
