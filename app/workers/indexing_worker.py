import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from app.core.models import ImportResult
from app.indexing.indexer import Indexer

logger = logging.getLogger(__name__)


class IndexingWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(ImportResult)
    failed = Signal(str)

    def __init__(self, indexer: Indexer, folder: Path) -> None:
        super().__init__()
        self.indexer = indexer
        self.folder = folder

    def run(self) -> None:
        try:
            result = self.indexer.import_folder(
                self.folder,
                progress=lambda current, total, name: self.progress.emit(
                    current, total, name
                ),
            )
        except Exception as error:
            logger.exception("Indexing worker failed")
            self.failed.emit(str(error))
            return
        self.finished.emit(result)
