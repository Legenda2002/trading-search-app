"""Parse MetaTrader-5 CSV exports into in-memory OHLC bars.

Expected header (MT5 default):
    <DATE> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>

Date column may be ``YYYY.MM.DD`` (Daily) or ``YYYY.MM.DD HH:MM`` (intraday).
Values are separated by tabs or any whitespace.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OhlcBar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


_WHITESPACE = re.compile(r"\s+")


def parse_mt5_csv(path: Path) -> list[OhlcBar]:
    bars: list[OhlcBar] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        rows = list(reader)

    if not rows:
        return bars

    if "<" in rows[0][0]:
        rows = rows[1:]

    for raw in rows:
        if len(raw) == 1:
            tokens = _WHITESPACE.split(raw[0].strip())
        else:
            tokens = [cell.strip() for cell in raw if cell.strip()]

        if len(tokens) < 5:
            continue

        date_token = tokens[0]
        time_token = ""
        offset = 1
        if not _is_number(tokens[1]):
            time_token = tokens[1]
            offset = 2

        try:
            open_p = float(tokens[offset])
            high_p = float(tokens[offset + 1])
            low_p = float(tokens[offset + 2])
            close_p = float(tokens[offset + 3])
        except (IndexError, ValueError):
            continue

        timestamp = f"{date_token} {time_token}".strip()
        bars.append(
            OhlcBar(
                timestamp=timestamp,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
            )
        )

    logger.info("Parsed %d bars from %s", len(bars), path)
    return bars


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
