import logging
import sys

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)
