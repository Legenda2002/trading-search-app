from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.models import (
    MATCH_TYPE_EXACT,
    MATCH_TYPE_NONE,
    MATCH_TYPE_SIMILAR,
    SearchResult,
)


_MATCH_TYPE_MARKER = {
    MATCH_TYPE_EXACT: "[E]",
    MATCH_TYPE_SIMILAR: "[S]",
    MATCH_TYPE_NONE: "[ ]",
}

_MATCH_TYPE_COLOR = {
    MATCH_TYPE_EXACT: QColor(70, 170, 70),
    MATCH_TYPE_SIMILAR: QColor(60, 120, 200),
    MATCH_TYPE_NONE: QColor(140, 140, 140),
}


class ResultPanel(QWidget):
    result_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title = QLabel("Лучшие совпадения")
        self.title.setObjectName("sectionTitle")
        layout.addWidget(self.title)

        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget, stretch=1)

        self.preview = QLabel("Выберите результат для предпросмотра")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(220)
        layout.addWidget(self.preview)

    def set_results(self, results: list[SearchResult]) -> None:
        self.list_widget.clear()
        self.preview.setText("Выберите результат для предпросмотра")
        self.preview.setPixmap(QPixmap())

        if not results:
            placeholder = QListWidgetItem("Совпадений не найдено")
            placeholder.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(placeholder)
            return

        for index, result in enumerate(results, start=1):
            marker = _MATCH_TYPE_MARKER.get(result.match_type, "[ ]")
            if result.match_type == MATCH_TYPE_EXACT:
                detail = (
                    f"точное   {result.inlier_count} inliers, "
                    f"emb {result.embedding_similarity:.2f}"
                )
            elif result.match_type == MATCH_TYPE_SIMILAR:
                detail = (
                    f"похожее  emb {result.embedding_similarity:.2f}, "
                    f"совп. {result.match_count}"
                )
            else:
                detail = (
                    f"—        emb {result.embedding_similarity:.2f}, "
                    f"совп. {result.match_count}"
                )

            percent = f"{result.similarity_percent:5.1f}%"
            label = (
                f"{marker} {index:>2}. {percent}  "
                f"{result.image.filename:<40s}  {detail}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, result)
            color = _MATCH_TYPE_COLOR.get(result.match_type)
            if color is not None:
                item.setForeground(QBrush(color))
            self.list_widget.addItem(item)

    def _on_selection_changed(self) -> None:
        items = self.list_widget.selectedItems()
        if not items:
            return
        result = items[0].data(Qt.UserRole)
        if result is None:
            return
        self._update_preview(result)
        self.result_selected.emit(result)

    def _update_preview(self, result: SearchResult) -> None:
        path = result.image.thumbnail_path or result.image.stored_path
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.preview.setText(
                f"Не удалось загрузить превью для {result.image.filename}"
            )
            return
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.width(),
                self.preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
