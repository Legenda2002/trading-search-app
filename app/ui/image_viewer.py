import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

logger = logging.getLogger(__name__)


OVERLAY_COLOR = QColor(0, 200, 80, 230)
OVERLAY_FILL_COLOR = QColor(0, 200, 80, 35)
SELECTION_COLOR = QColor(40, 130, 220, 235)
SELECTION_FILL_COLOR = QColor(40, 130, 220, 55)
BACKGROUND_COLOR = QColor(30, 30, 33)

MIN_SELECTION_PX = 5


class ImageViewer(QWidget):
    """Image preview with a green overlay polygon and a blue draggable selection.

    All overlays and selections are stored in *image coordinates*, so they
    follow the picture across zoom/resize, and the bounding polygon stays
    aligned with the homography-localized area.
    """

    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)

        self._pixmap: QPixmap | None = None
        self._source_path: Path | None = None
        self._overlay_polygon: np.ndarray | None = None

        self._selection_image_rect: QRectF | None = None
        self._dragging: bool = False
        self._drag_start_image: QPointF | None = None
        self._drag_current_image: QPointF | None = None

    def has_image(self) -> bool:
        return self._pixmap is not None

    def has_selection(self) -> bool:
        return self._selection_image_rect is not None

    def source_path(self) -> Path | None:
        return self._source_path

    def load_image(self, path: Path) -> bool:
        return self.load_image_with_overlay(path, polygon=None)

    def load_image_with_overlay(
        self, path: Path, polygon: np.ndarray | None
    ) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            logger.warning("Cannot load image %s", path)
            self._pixmap = None
            self._source_path = None
            self._overlay_polygon = None
            self._reset_selection_state(emit=True)
            self.unsetCursor()
            self.update()
            return False

        self._pixmap = pixmap
        self._source_path = Path(path)
        self._overlay_polygon = polygon if polygon is not None else None
        self._reset_selection_state(emit=True)
        self.setCursor(Qt.CrossCursor)
        self.update()
        logger.debug(
            "Loaded image %s (%dx%d) with overlay=%s",
            path,
            pixmap.width(),
            pixmap.height(),
            polygon is not None,
        )
        return True

    def clear_image(self) -> None:
        self._pixmap = None
        self._source_path = None
        self._overlay_polygon = None
        self._reset_selection_state(emit=True)
        self.unsetCursor()
        self.update()

    def clear_selection(self) -> None:
        self._reset_selection_state(emit=True)
        self.update()

    def get_selected_crop(self) -> np.ndarray | None:
        if self._source_path is None or self._selection_image_rect is None:
            return None

        image = cv2.imread(str(self._source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Cannot reload source image for crop: %s", self._source_path)
            return None

        height, width = image.shape[:2]
        rect = self._selection_image_rect
        x1 = max(0, int(round(rect.x())))
        y1 = max(0, int(round(rect.y())))
        x2 = min(width, int(round(rect.x() + rect.width())))
        y2 = min(height, int(round(rect.y() + rect.height())))

        if x2 - x1 < MIN_SELECTION_PX or y2 - y1 < MIN_SELECTION_PX:
            logger.debug("Selection too small after clamping: %dx%d", x2 - x1, y2 - y1)
            return None

        crop = image[y1:y2, x1:x2].copy()
        logger.info(
            "Cropped selection x=%d y=%d w=%d h=%d from %s",
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            self._source_path.name,
        )
        return crop

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        if self._pixmap is None:
            painter.setPen(QColor(140, 147, 164))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Фрагмент ещё не загружен\n\n"
                "1.  «Открыть» — выбрать файл\n"
                "2.  «Вставить» или Ctrl+V — из буфера обмена\n"
                "3.  «База графиков» — взять из библиотеки\n\n"
                "Затем выберите режим и нажмите «Найти».",
            )
            return

        displayed = self._compute_displayed_rect()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(displayed, self._pixmap, QRectF(self._pixmap.rect()))

        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_polygon(painter, displayed)
        self._draw_selection(painter, displayed)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._pixmap is None:
            return
        point = self._widget_to_image(event.position(), clamp=False)
        if point is None:
            return
        self._dragging = True
        self._drag_start_image = point
        self._drag_current_image = point
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging or self._pixmap is None:
            return
        point = self._widget_to_image(event.position(), clamp=True)
        if point is None:
            return
        self._drag_current_image = point
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._dragging:
            return
        self._dragging = False

        rect = self._dragging_rect_image()
        self._drag_start_image = None
        self._drag_current_image = None

        if rect is None or rect.width() < MIN_SELECTION_PX or rect.height() < MIN_SELECTION_PX:
            previous = self._selection_image_rect
            self._selection_image_rect = None
            if previous is not None:
                self.selection_changed.emit(None)
        else:
            self._selection_image_rect = rect
            self.selection_changed.emit(rect)

        self.update()

    def _reset_selection_state(self, *, emit: bool) -> None:
        was_set = self._selection_image_rect is not None
        self._selection_image_rect = None
        self._dragging = False
        self._drag_start_image = None
        self._drag_current_image = None
        if emit and was_set:
            self.selection_changed.emit(None)

    def _draw_polygon(self, painter: QPainter, displayed: QRectF) -> None:
        if self._overlay_polygon is None or len(self._overlay_polygon) < 3:
            return
        widget_points = [
            self._image_to_widget(QPointF(float(p[0]), float(p[1])), displayed)
            for p in self._overlay_polygon
        ]
        qpolygon = QPolygonF(widget_points)

        pen = QPen(OVERLAY_COLOR)
        pen.setWidthF(max(2.0, displayed.width() / 250))
        pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(OVERLAY_FILL_COLOR)
        painter.drawPolygon(qpolygon)

    def _draw_selection(self, painter: QPainter, displayed: QRectF) -> None:
        rect = self._current_selection_image_rect()
        if rect is None or rect.isEmpty():
            return
        top_left = self._image_to_widget(rect.topLeft(), displayed)
        bottom_right = self._image_to_widget(rect.bottomRight(), displayed)
        rect_widget = QRectF(top_left, bottom_right).normalized()

        pen = QPen(SELECTION_COLOR)
        pen.setWidthF(max(1.5, displayed.width() / 350))
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(SELECTION_FILL_COLOR)
        painter.drawRect(rect_widget)

    def _current_selection_image_rect(self) -> QRectF | None:
        if self._dragging:
            return self._dragging_rect_image()
        return self._selection_image_rect

    def _dragging_rect_image(self) -> QRectF | None:
        if self._drag_start_image is None or self._drag_current_image is None:
            return None
        return QRectF(self._drag_start_image, self._drag_current_image).normalized()

    def _compute_displayed_rect(self) -> QRectF:
        if self._pixmap is None:
            return QRectF()
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        ww = self.width()
        wh = self.height()
        if pw == 0 or ph == 0 or ww == 0 or wh == 0:
            return QRectF()
        scale = min(ww / pw, wh / ph)
        dw = pw * scale
        dh = ph * scale
        dx = (ww - dw) / 2.0
        dy = (wh - dh) / 2.0
        return QRectF(dx, dy, dw, dh)

    def _widget_to_image(self, pos: QPointF, *, clamp: bool) -> QPointF | None:
        if self._pixmap is None:
            return None
        displayed = self._compute_displayed_rect()
        if displayed.isEmpty():
            return None
        scale = displayed.width() / self._pixmap.width()
        if scale <= 0:
            return None
        x = (pos.x() - displayed.x()) / scale
        y = (pos.y() - displayed.y()) / scale
        pw = float(self._pixmap.width())
        ph = float(self._pixmap.height())
        if clamp:
            x = max(0.0, min(pw, x))
            y = max(0.0, min(ph, y))
        else:
            if x < 0 or y < 0 or x > pw or y > ph:
                return None
        return QPointF(x, y)

    def _image_to_widget(self, point: QPointF, displayed: QRectF) -> QPointF:
        if self._pixmap is None or displayed.isEmpty():
            return QPointF()
        scale = displayed.width() / self._pixmap.width()
        return QPointF(
            displayed.x() + point.x() * scale,
            displayed.y() + point.y() * scale,
        )
