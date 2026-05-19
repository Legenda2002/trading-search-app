import argparse
import logging
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.core.config import ensure_data_dirs
from app.core.logging_config import configure_logging
from app.ui.main_window import MainWindow
from app.ui.style import APP_STYLESHEET


def main() -> int:
    parser = argparse.ArgumentParser(description="Поиск фрагментов трейдинговых графиков")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Включить подробное DEBUG-логирование",
    )
    args, qt_args = parser.parse_known_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger(__name__).info("Запуск приложения")

    ensure_data_dirs()
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Поиск фрагментов трейдинговых графиков")
    app.setApplicationDisplayName("Поиск фрагментов трейдинговых графиков")
    app.setOrganizationName("Trading Search")
    app.setFont(QFont("Inter", 10))
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
