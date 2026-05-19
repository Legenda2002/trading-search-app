import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.core.models import ImportResult, SearchOutcome, SearchResult
from app.indexing.indexer import Indexer
from app.search.hybrid_search import (
    MODE_EXACT,
    MODE_SIMILAR,
    MODE_SMART,
    HybridSearchEngine,
)
from app.search.search_engine import SearchEngine
from app.storage.database import Database
from app.storage.descriptor_store import DescriptorStore
from app.storage.embedding_store import EmbeddingStore
from app.storage.image_store import ImageStore
from app.ui.image_viewer import ImageViewer
from app.ui.result_panel import ResultPanel
from app.vision.embedding_extractor import EmbeddingExtractor
from app.workers.indexing_worker import IndexingWorker
from app.workers.search_worker import SearchWorker

logger = logging.getLogger(__name__)


ALGORITHM = "ORB"

SEARCH_MODES = [
    (MODE_SMART, "Умный"),
    (MODE_EXACT, "Точный"),
    (MODE_SIMILAR, "Похожие"),
]

# Mode descriptions shown as tooltip on each segmented button.
MODE_TOOLTIPS = {
    MODE_SMART:
        "Умный режим (рекомендуется)\n\n"
        "Шаг 1. ИИ сравнивает запрос со ВСЕМИ графиками базы\n"
        "        и отбирает 2000 самых близких.\n"
        "Шаг 2. ORB-движок перепроверяет геометрию на этих\n"
        "        2000 кандидатах и находит дубликаты.\n\n"
        "Лучший компромисс между скоростью и полнотой\n"
        "перебора. Время поиска: 8-12 секунд.",
    MODE_EXACT:
        "Точный — поиск по ВСЕЙ базе (для дубликатов)\n\n"
        "ORB-движок перебирает каждый графический файл\n"
        "в базе без пропусков. Самый медленный (30-60 секунд\n"
        "на 17 000 графиков), но гарантирует 100% покрытие.\n\n"
        "Используйте, когда нужно точно убедиться, что\n"
        "ничего не пропущено — например, для поиска\n"
        "дубликатов или конкретного фрагмента из базы.",
    MODE_SIMILAR:
        "Похожие — только ИИ-анализ\n\n"
        "ИИ сравнивает запрос со ВСЕМИ графиками базы\n"
        "по визуальному сходству, без проверки ORB.\n"
        "Самый быстрый (1-2 секунды), находит визуально\n"
        "похожие графики в любом стиле (другие цвета,\n"
        "другой фон), но может пропустить точные дубликаты.",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Поиск фрагментов трейдинговых графиков")
        self.resize(1280, 820)

        self.database = Database()
        self.image_store = ImageStore()
        self.descriptor_store = DescriptorStore()
        self.embedding_store = EmbeddingStore()
        # The embedding extractor loads ~85 MB of weights and pins ~750 MB of
        # RAM, so we build it lazily on the first call to a search/index path
        # that actually needs it (typically warmup).
        self._embedding_extractor: EmbeddingExtractor | None = None
        self.indexer = Indexer(
            self.database,
            self.image_store,
            self.descriptor_store,
            algorithm=ALGORITHM,
        )
        self.orb_engine = SearchEngine(
            self.database,
            self.descriptor_store,
            algorithm=ALGORITHM,
            top_k=10,
        )
        # ``search_engine`` is the public handle used by workers. It can be
        # swapped to a HybridSearchEngine once the embedding extractor is
        # ready; until then we run plain ORB so the UI is responsive even if
        # torch fails to load.
        self.search_engine = self.orb_engine
        self.hybrid_engine: HybridSearchEngine | None = None
        self.current_mode = MODE_SMART

        self._indexing_thread: QThread | None = None
        self._indexing_worker: IndexingWorker | None = None
        self._search_thread: QThread | None = None
        self._search_worker: SearchWorker | None = None

        self._paste_temp_path = (
            Path(tempfile.gettempdir()) / "trading_search_paste.png"
        )
        # Path of the image currently treated as the search query. We keep
        # this separate from ``viewer.source_path()`` because clicking on a
        # result swaps the viewer to a library image — but the *query* must
        # stay the user's original fragment for the next "Найти" click.
        self._query_path: Path | None = None
        # True while the viewer is showing the active query image. Flips to
        # False as soon as the user previews a result so we know to reload
        # the query before running another search.
        self._viewer_shows_query: bool = False

        self._build_ui()
        self._register_shortcuts()
        self._refresh_library_label()
        self._kick_off_warmup()

        # Poll the warmup thread once a second so the AI-status indicator in
        # the status bar flips to "готова" as soon as the hybrid engine is up.
        self._ai_status_timer = QTimer(self)
        self._ai_status_timer.setInterval(1000)
        self._ai_status_timer.timeout.connect(self._refresh_ai_status_label)
        self._ai_status_timer.start()

    def _kick_off_warmup(self) -> None:
        """Preload descriptors + embeddings + DINOv2 in the background.

        Until warmup finishes, searches still work (ORB-only fallback), but
        the first hybrid call would otherwise block the UI for ~5 seconds
        while torch initialises.
        """
        def _run() -> None:
            try:
                self.orb_engine.warmup()
            except Exception:
                logger.exception("ORB warmup failed")

            if self._embedding_extractor is None:
                try:
                    extractor = EmbeddingExtractor()
                    extractor.warmup()
                except Exception:
                    logger.exception("Embedding extractor warmup failed")
                    return
                self._embedding_extractor = extractor

            if self.hybrid_engine is None:
                self.hybrid_engine = HybridSearchEngine(
                    database=self.database,
                    embedding_store=self.embedding_store,
                    embedding_extractor=self._embedding_extractor,
                    orb_engine=self.orb_engine,
                    default_mode=self.current_mode,
                )

            try:
                self.hybrid_engine.warmup()
            except Exception:
                logger.exception("Hybrid warmup failed")
                return

            # Swap the active engine. Setting it after warmup means searches
            # initiated during startup get the ORB-only engine instead of
            # blocking on the still-loading model.
            self.search_engine = self.hybrid_engine
            # Wire the embedding pipeline into the indexer so future imports
            # automatically populate the embedding store.
            self.indexer.embedding_extractor = self._embedding_extractor
            self.indexer.embedding_store = self.embedding_store
            logger.info("Hybrid search engine is now active (mode=%s)", self.current_mode)

        thread = threading.Thread(target=_run, name="search-warmup", daemon=True)
        thread.start()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.viewer = ImageViewer()
        self.viewer.selection_changed.connect(self._on_selection_changed)
        left.addWidget(self.viewer, stretch=1)

        self.library_label = QLabel()
        self.library_label.setObjectName("libraryLabel")
        self.library_label.setAlignment(Qt.AlignLeft)
        left.addWidget(self.library_label)

        root.addLayout(left, stretch=3)

        self.result_panel = ResultPanel()
        self.result_panel.result_selected.connect(self._on_result_selected)
        root.addWidget(self.result_panel, stretch=2)

        toolbar = QToolBar("Главная панель")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        self.import_action = QPushButton("Импорт папки")
        self.import_action.setToolTip("Добавить папку с PNG-графиками в индекс")
        self.import_action.clicked.connect(self._on_import_clicked)
        toolbar.addWidget(self.import_action)

        self.library_action = QPushButton("База графиков")
        self.library_action.setToolTip("Открыть список всех графиков в базе")
        self.library_action.clicked.connect(self._on_browse_library_clicked)
        toolbar.addWidget(self.library_action)

        toolbar.addSeparator()

        self.open_action = QPushButton("Открыть")
        self.open_action.setToolTip("Открыть файл-фрагмент для поиска")
        self.open_action.clicked.connect(self._on_open_clicked)
        toolbar.addWidget(self.open_action)

        self.paste_action = QPushButton("Вставить  (Ctrl+V)")
        self.paste_action.setToolTip(
            "Вставить изображение из буфера обмена.\n"
            "Сделайте скриншот (Print Screen) и нажмите эту кнопку или Ctrl+V.\n"
            "После этого выберите режим и нажмите «Найти»."
        )
        self.paste_action.clicked.connect(self._on_paste_clicked)
        toolbar.addWidget(self.paste_action)

        toolbar.addSeparator()

        self.find_action = QPushButton("Найти  (Enter)")
        self.find_action.setObjectName("primary")
        self.find_action.setEnabled(False)
        self.find_action.setToolTip(
            "Запустить поиск по загруженному изображению в выбранном режиме.\n\n"
            "Если выделить мышью прямоугольник — поиск пойдёт только по нему."
        )
        self.find_action.clicked.connect(self._on_find_clicked)
        toolbar.addWidget(self.find_action)

        # Spacer pushes the mode switch to the right edge.
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        toolbar.addWidget(QLabel("Режим:"))
        toolbar.addWidget(self._build_mode_switch())

        # Status bar: progress on the left, AI indicator on the right.
        self.setStatusBar(QStatusBar(self))
        self.statusBar().setSizeGripEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setMaximumWidth(220)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.ai_status_label = QLabel("ИИ: загружается…")
        self.ai_status_label.setObjectName("statusDot")
        self.statusBar().addPermanentWidget(self.ai_status_label)
        self.statusBar().showMessage(
            "Готово. Загрузите фрагмент («Открыть» или Ctrl+V), "
            "выберите режим и нажмите «Найти»."
        )

    def _on_import_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с изображениями графиков"
        )
        if not folder:
            return
        self._start_indexing(Path(folder))

    def _on_open_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть фрагмент для поиска",
            filter="Изображения (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        chosen = Path(path)
        if self.viewer.load_image(chosen):
            self._set_active_query(chosen, label=chosen.name)

    def _on_paste_clicked(self) -> None:
        clipboard = QApplication.clipboard()
        qimage = clipboard.image()
        if qimage.isNull():
            QMessageBox.information(
                self,
                "Вставка из буфера",
                "В буфере обмена нет изображения.\n"
                "Сделайте скриншот (Print Screen) и попробуйте снова.",
            )
            return

        if not qimage.save(str(self._paste_temp_path), "PNG"):
            QMessageBox.critical(
                self,
                "Вставка из буфера",
                f"Не удалось сохранить изображение в {self._paste_temp_path}",
            )
            return

        logger.info(
            "Pasted image %dx%d, saved to %s",
            qimage.width(),
            qimage.height(),
            self._paste_temp_path,
        )
        self.viewer.load_image(self._paste_temp_path)
        self._set_active_query(
            self._paste_temp_path,
            label=f"вставленное изображение {qimage.width()}×{qimage.height()}",
        )

    def _set_active_query(self, path: Path, *, label: str) -> None:
        """Mark *path* as the active search query.

        Called after every explicit query load (open, paste, library pick).
        Keeps ``self._query_path`` in sync with what the viewer shows so the
        next "Найти" click searches the right image, even after the user
        previews a result.
        """
        self._query_path = path
        self._viewer_shows_query = True
        self.find_action.setEnabled(True)
        mode_name = {
            MODE_SMART: "Умный",
            MODE_EXACT: "Точный",
            MODE_SIMILAR: "Похожие",
        }.get(self.current_mode, self.current_mode)
        self.statusBar().showMessage(
            f"Загружено: {label}.   "
            f"Режим: «{mode_name}».   "
            f"Нажмите «Найти» (или Enter) для поиска. "
            f"Можно сначала выделить прямоугольник мышью."
        )

    def _on_find_clicked(self) -> None:
        """Run the search using the active query image.

        Always uses ``self._query_path`` (the last image the user explicitly
        loaded), so previewing a result no longer leaks into the next
        search. If the viewer still shows the query and the user drew a
        rectangle, the crop is searched instead of the full image.
        """
        if self._query_path is None:
            QMessageBox.information(
                self,
                "Нет фрагмента",
                "Сначала загрузите изображение: «Открыть» или «Вставить» (Ctrl+V).",
            )
            return

        if not self._query_path.exists():
            QMessageBox.warning(
                self,
                "Запрос недоступен",
                f"Не удалось найти файл запроса:\n{self._query_path}\n\n"
                f"Загрузите фрагмент заново.",
            )
            self._query_path = None
            self.find_action.setEnabled(False)
            return

        # Use the user's mouse selection only if the viewer is still showing
        # the query. After previewing a result the selection is gone anyway,
        # so we restore the query view and search the full image.
        if self._viewer_shows_query:
            crop = self.viewer.get_selected_crop()
            if crop is not None:
                logger.info("Searching by selected region shape=%s", crop.shape)
                self._start_search_array(crop)
                return
        else:
            logger.info(
                "Restoring viewer to query image before search: %s",
                self._query_path,
            )
            self.viewer.load_image(self._query_path)
            self._viewer_shows_query = True

        self._start_search_path(self._query_path)

    def _on_browse_library_clicked(self) -> None:
        # Lazy import keeps the dialog out of the startup path.
        from app.ui.library_browser import LibraryBrowserDialog

        dialog = LibraryBrowserDialog(self.database, parent=self)
        dialog.image_chosen.connect(self._on_library_image_chosen)
        dialog.exec()

    def _on_library_image_chosen(self, path: Path) -> None:
        if self.viewer.load_image(path):
            self._set_active_query(path, label=path.name)

    def _build_mode_switch(self) -> QWidget:
        """Three-segment toggle for picking the search mode.

        Visually grouped (joined buttons with shared rounded corners) so the
        user sees all three options at once and switches with a single click,
        no dropdown indirection. Single-line labels keep the toolbar compact.
        """
        container = QFrame()
        container.setObjectName("modeSwitch")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.mode_buttons: list[QPushButton] = []
        self.mode_group = QButtonGroup(container)
        self.mode_group.setExclusive(True)

        for index, (mode_id, title) in enumerate(SEARCH_MODES):
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.setProperty("modeId", mode_id)
            btn.setToolTip(MODE_TOOLTIPS[mode_id])
            btn.setCursor(Qt.PointingHandCursor)
            # Distinct objectName per position so the stylesheet can round
            # only the outer corners of the segmented control.
            if index == 0:
                btn.setObjectName("modeSegmentLeft")
            elif index == len(SEARCH_MODES) - 1:
                btn.setObjectName("modeSegmentRight")
            else:
                btn.setObjectName("modeSegmentMiddle")
            if mode_id == MODE_SMART:
                btn.setChecked(True)
            btn.clicked.connect(
                lambda _checked=False, m=mode_id: self._set_mode(m)
            )
            self.mode_group.addButton(btn, index)
            self.mode_buttons.append(btn)
            layout.addWidget(btn)

        return container

    def _set_mode(self, mode: str) -> None:
        if mode == self.current_mode:
            return
        self.current_mode = mode
        for btn in self.mode_buttons:
            btn.setChecked(btn.property("modeId") == mode)
        if self.hybrid_engine is not None:
            self.hybrid_engine.default_mode = mode
            logger.info("Search mode changed to %s", mode)
        else:
            logger.info(
                "Search mode preselected to %s (hybrid engine not ready yet)",
                mode,
            )
        # Surface the change in the status bar so it's obvious it took effect.
        ru_name = {
            MODE_SMART: "Умный (ИИ + ORB)",
            MODE_EXACT: "Точный (только ORB)",
            MODE_SIMILAR: "Похожие (только ИИ)",
        }.get(mode, mode)
        self.statusBar().showMessage(f"Режим поиска: {ru_name}", 4000)

    def _on_selection_changed(self, selection) -> None:
        has_selection = selection is not None
        if has_selection:
            self.statusBar().showMessage(
                f"Выделено: {int(selection.width())}×{int(selection.height())} px. "
                f"Нажмите «Найти» — поиск пойдёт только по этой области."
            )
        elif self.viewer.has_image():
            self.statusBar().showMessage(
                "Выделение сброшено. Поиск будет идти по всему изображению."
            )

    def _start_indexing(self, folder: Path) -> None:
        self._set_busy(True, "Импорт и индексирование…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        thread = QThread(self)
        worker = IndexingWorker(self.indexer, folder)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_indexing_progress)
        worker.finished.connect(self._on_indexing_finished)
        worker.failed.connect(self._on_indexing_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._indexing_thread = thread
        self._indexing_worker = worker
        thread.start()

    def _start_search_path(self, query_path: Path) -> None:
        worker = SearchWorker(self.search_engine, query_path=query_path)
        self._run_search_worker(worker, self._busy_message_for_mode())

    def _start_search_array(self, image: np.ndarray) -> None:
        worker = SearchWorker(self.search_engine, query_array=image)
        self._run_search_worker(
            worker, self._busy_message_for_mode(region=True)
        )

    def _busy_message_for_mode(self, *, region: bool = False) -> str:
        """Status text shown while a search is running.

        We surface the total library size and the mode-specific strategy so
        the user can see at a glance that the whole base is being checked.
        """
        total = len(self.database.list_images())
        what = "выделенной области" if region else "всему изображению"
        if self.current_mode == MODE_EXACT:
            return (
                f"Точный поиск по {what}: перебираю все {total} "
                f"графиков ORB-движком (30-60 секунд)…"
            )
        if self.current_mode == MODE_SIMILAR:
            return (
                f"Поиск похожих по {what}: ИИ сравнивает запрос со всеми "
                f"{total} графиками базы…"
            )
        # Smart
        return (
            f"Умный поиск по {what}: ИИ сравнил все {total} графиков, "
            f"ORB проверяет 2000 лучших кандидатов…"
        )

    def _run_search_worker(self, worker: SearchWorker, message: str) -> None:
        self._set_busy(True, message)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_search_finished)
        worker.failed.connect(self._on_search_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._search_thread = thread
        self._search_worker = worker
        thread.start()

    def _on_indexing_progress(self, current: int, total: int, name: str) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"Индексирование {current}/{total}: {name}")

    def _on_indexing_finished(self, result: ImportResult) -> None:
        self._set_busy(False)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(
            f"Импортировано: {result.imported} новых, пропущено: {result.skipped}"
        )
        self._refresh_library_label()
        self.search_engine.invalidate_cache()
        self._kick_off_warmup()

    def _on_indexing_failed(self, message: str) -> None:
        self._set_busy(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Ошибка индексирования", message)

    def _on_search_finished(self, outcome: SearchOutcome) -> None:
        self._set_busy(False)
        self.progress_bar.setVisible(False)
        self.result_panel.set_results(outcome.results)

        exact_count = sum(1 for r in outcome.results if r.match_type == "exact")
        similar_count = sum(1 for r in outcome.results if r.match_type == "similar")

        if not outcome.results:
            message = (
                "Похожих графиков в базе не найдено. "
                "Попробуйте другой режим или убедитесь, что в базе есть нужные графики."
            )
        else:
            parts = []
            if exact_count:
                parts.append(f"точных: {exact_count}")
            if similar_count:
                parts.append(f"похожих: {similar_count}")
            summary = " · ".join(parts) if parts else f"всего {len(outcome.results)}"
            message = (
                f"Найдено — {summary}.   "
                f"Ключевые точки запроса: {outcome.query_keypoint_count}"
            )
        self.statusBar().showMessage(message)
        logger.info(message)

    def _on_search_failed(self, message: str) -> None:
        self._set_busy(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Ошибка поиска", message)

    def _on_result_selected(self, result: SearchResult) -> None:
        self.viewer.load_image_with_overlay(
            result.image.stored_path,
            result.polygon,
        )
        # Preview only — the active query (self._query_path) is untouched, so
        # the next "Найти" still searches the user's original fragment.
        self._viewer_shows_query = False

        type_ru = {"exact": "точное", "similar": "похожее", "none": "—"}.get(
            result.match_type, result.match_type
        )
        message = (
            f"Просмотр результата: {result.image.filename}   "
            f"сходство {result.similarity_percent:.0f}%   "
            f"тип: {type_ru}   "
            f"emb {result.embedding_similarity:.3f}   "
            f"совпадений {result.match_count}   "
            f"inliers {result.inlier_count}"
        )
        self.statusBar().showMessage(message)
        logger.info("Result selected: %s", message)

    def _register_shortcuts(self) -> None:
        # ApplicationShortcut context makes Ctrl+V work no matter which child
        # widget (toolbar button, list, etc.) currently has focus.
        self._paste_shortcut = QShortcut(QKeySequence.Paste, self)
        self._paste_shortcut.setContext(Qt.ApplicationShortcut)
        self._paste_shortcut.activated.connect(self._on_paste_clicked)

        # Plain Ctrl+V duplicate so the binding still works on platforms
        # where QKeySequence.Paste resolves to e.g. Shift+Insert.
        self._paste_shortcut_ctrl_v = QShortcut(QKeySequence("Ctrl+V"), self)
        self._paste_shortcut_ctrl_v.setContext(Qt.ApplicationShortcut)
        self._paste_shortcut_ctrl_v.activated.connect(self._on_paste_clicked)

        # Enter / Return triggers the primary "Найти" action so the workflow
        # paste -> choose mode -> Enter feels natural.
        self._find_shortcut_return = QShortcut(QKeySequence(Qt.Key_Return), self)
        self._find_shortcut_return.setContext(Qt.ApplicationShortcut)
        self._find_shortcut_return.activated.connect(self._on_find_clicked)
        self._find_shortcut_enter = QShortcut(QKeySequence(Qt.Key_Enter), self)
        self._find_shortcut_enter.setContext(Qt.ApplicationShortcut)
        self._find_shortcut_enter.activated.connect(self._on_find_clicked)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self.import_action.setEnabled(not busy)
        self.library_action.setEnabled(not busy)
        self.open_action.setEnabled(not busy)
        self.paste_action.setEnabled(not busy)
        self.find_action.setEnabled(not busy and self.viewer.has_image())
        for btn in getattr(self, "mode_buttons", ()):
            btn.setEnabled(not busy)
        if message is not None:
            self.statusBar().showMessage(message)

    def _refresh_library_label(self) -> None:
        count = len(self.database.list_images())
        self.library_label.setText(f"База: {count} изображений в индексе")

    def _refresh_ai_status_label(self) -> None:
        if isinstance(self.search_engine, HybridSearchEngine):
            self.ai_status_label.setText("ИИ-движок: готов")
            self.ai_status_label.setStyleSheet("color: #3ec27b;")
            self._ai_status_timer.stop()
        elif self._embedding_extractor is None:
            self.ai_status_label.setText("ИИ-движок: загружается…")
            self.ai_status_label.setStyleSheet("color: #f0a050;")
        else:
            self.ai_status_label.setText("ИИ-движок: прогрев…")
            self.ai_status_label.setStyleSheet("color: #f0a050;")
