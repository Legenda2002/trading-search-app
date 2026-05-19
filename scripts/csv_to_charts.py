"""Convert an MT5 OHLC CSV into a folder of candlestick PNG images.

Cuts the price history with a sliding window so a single CSV with thousands of
bars yields a library of overlapping chart fragments. The resulting folder can
be passed directly to ``Import folder`` in the desktop app.

Example:

    python -m scripts.csv_to_charts \\
        --csv data/csv/XAUUSD_Daily.csv \\
        --output samples/library/xauusd_daily \\
        --window 60 --step 30
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.chart_render.candlestick import render_chart, save_chart
from app.chart_render.ohlc import OhlcBar, parse_mt5_csv
from app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def slice_bars(bars: list[OhlcBar], window: int, step: int) -> list[list[OhlcBar]]:
    if window <= 0:
        raise ValueError("window must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if len(bars) < window:
        return [bars] if bars else []

    slices: list[list[OhlcBar]] = []
    for start in range(0, len(bars) - window + 1, step):
        slices.append(bars[start : start + window])
    return slices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to MT5 CSV export")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Folder where PNG charts will be written",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Number of candles in each chart image",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=30,
        help="Sliding window step (in candles)",
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument(
        "--max-charts",
        type=int,
        default=None,
        help="Optional limit on number of charts to render",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix (default: CSV stem)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    if not args.csv.is_file():
        logger.error("CSV file not found: %s", args.csv)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.csv.stem

    bars = parse_mt5_csv(args.csv)
    if not bars:
        logger.error("No bars parsed from %s", args.csv)
        return 1

    slices = slice_bars(bars, args.window, args.step)
    if args.max_charts is not None:
        slices = slices[: args.max_charts]

    logger.info(
        "Rendering %d charts (window=%d step=%d) to %s",
        len(slices),
        args.window,
        args.step,
        args.output,
    )

    for index, bar_window in enumerate(slices):
        image = render_chart(
            bar_window,
            width=args.width,
            height=args.height,
        )
        start_ts = _safe_token(bar_window[0].timestamp)
        end_ts = _safe_token(bar_window[-1].timestamp)
        filename = f"{prefix}_{index:05d}_{start_ts}_{end_ts}.png"
        save_chart(image, args.output / filename)
        if (index + 1) % 50 == 0 or index == len(slices) - 1:
            logger.info("  ... rendered %d / %d", index + 1, len(slices))

    logger.info("Done. Output folder: %s", args.output)
    return 0


def _safe_token(timestamp: str) -> str:
    return (
        timestamp.replace(".", "")
        .replace(":", "")
        .replace(" ", "_")
        .replace("/", "-")
    )


if __name__ == "__main__":
    raise SystemExit(main())
