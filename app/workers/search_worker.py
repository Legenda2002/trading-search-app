import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from app.search.search_engine import SearchEngine

logger = logging.getLogger(__name__)


class SearchWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        search_engine: SearchEngine,
        *,
        query_path: Path | None = None,
        query_array: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if query_path is None and query_array is None:
            raise ValueError("SearchWorker needs either query_path or query_array")
        self.search_engine = search_engine
        self.query_path = query_path
        self.query_array = query_array

    def run(self) -> None:
        try:
            if self.query_array is not None:
                logger.info("Running search by array shape=%s", self.query_array.shape)
                outcome = self.search_engine.search_by_array(self.query_array)
            else:
                assert self.query_path is not None
                outcome = self.search_engine.search_by_image(self.query_path)
        except Exception as error:
            logger.exception("Search worker failed")
            self.failed.emit(str(error))
            return
        self.finished.emit(outcome)
