"""Render a list of OHLC bars into a TradingView-style PNG image.

The renderer intentionally adds rich visual structure (price scale on the
right, time labels on the bottom, dense grid, body outlines) so the resulting
images have enough texture for ORB-based feature matching.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.chart_render.ohlc import OhlcBar


BG_COLOR = (24, 26, 32)
GRID_MAJOR = (60, 64, 75)
GRID_MINOR = (38, 41, 50)
WICK_COLOR = (178, 181, 190)
BULL_BODY = (8, 153, 129)
BEAR_BODY = (242, 54, 69)
TEXT_COLOR = (170, 175, 188)
TITLE_COLOR = (220, 224, 235)

LEFT_MARGIN = 14
RIGHT_MARGIN = 64
TOP_MARGIN = 28
BOTTOM_MARGIN = 30

_FONT = ImageFont.load_default()


def render_chart(
    bars: list[OhlcBar],
    *,
    width: int = 1024,
    height: int = 560,
    title: str | None = None,
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(image)

    if not bars:
        return image

    plot_left = LEFT_MARGIN
    plot_right = width - RIGHT_MARGIN
    plot_top = TOP_MARGIN
    plot_bottom = height - BOTTOM_MARGIN
    plot_w = max(plot_right - plot_left, 10)
    plot_h = max(plot_bottom - plot_top, 10)

    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    price_max = max(highs)
    price_min = min(lows)
    price_range = max(price_max - price_min, 1e-9)
    padding = price_range * 0.06
    price_max += padding
    price_min -= padding
    price_range = price_max - price_min

    def y_for(price: float) -> float:
        return plot_top + plot_h - ((price - price_min) / price_range) * plot_h

    _draw_grid(draw, bars, plot_left, plot_right, plot_top, plot_bottom)
    _draw_price_scale(draw, plot_right, plot_top, plot_bottom, price_min, price_max, y_for)
    _draw_time_scale(draw, bars, plot_left, plot_right, plot_bottom, height)
    _draw_candles(draw, bars, plot_left, plot_right, plot_h, y_for)

    if title:
        draw.text((LEFT_MARGIN, 8), title, fill=TITLE_COLOR, font=_FONT)

    draw.rectangle(
        [plot_left, plot_top, plot_right, plot_bottom],
        outline=GRID_MAJOR,
        width=1,
    )
    return image


def save_chart(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")


def _draw_grid(draw, bars, left, right, top, bottom) -> None:
    plot_h = bottom - top
    plot_w = right - left

    minor_step = max(plot_h // 12, 8)
    for y in range(top, bottom + 1, minor_step):
        draw.line([(left, y), (right, y)], fill=GRID_MINOR, width=1)

    major_step = max(plot_h // 6, 16)
    for y in range(top, bottom + 1, major_step):
        draw.line([(left, y), (right, y)], fill=GRID_MAJOR, width=1)

    columns = max(len(bars), 1)
    column_step = plot_w / columns
    minor_every = max(columns // 12, 1)
    major_every = max(columns // 6, 1)
    for index in range(columns + 1):
        x = int(left + index * column_step)
        if index % major_every == 0:
            draw.line([(x, top), (x, bottom)], fill=GRID_MAJOR, width=1)
        elif index % minor_every == 0:
            draw.line([(x, top), (x, bottom)], fill=GRID_MINOR, width=1)


def _draw_price_scale(draw, plot_right, top, bottom, price_min, price_max, y_for) -> None:
    levels = 8
    for i in range(levels + 1):
        price = price_min + (price_max - price_min) * (i / levels)
        y = y_for(price)
        draw.line([(plot_right, y), (plot_right + 4, y)], fill=GRID_MAJOR, width=1)
        label = _format_price(price)
        draw.text((plot_right + 6, int(y) - 5), label, fill=TEXT_COLOR, font=_FONT)


def _draw_time_scale(draw, bars, left, right, bottom, height) -> None:
    columns = max(len(bars), 1)
    column_step = (right - left) / columns
    ticks = min(8, columns)
    if ticks < 2:
        return
    for i in range(ticks):
        idx = int(i * (columns - 1) / (ticks - 1))
        x = int(left + (idx + 0.5) * column_step)
        draw.line([(x, bottom), (x, bottom + 4)], fill=GRID_MAJOR, width=1)
        label = _format_timestamp(bars[idx].timestamp)
        draw.text((x - 28, bottom + 6), label, fill=TEXT_COLOR, font=_FONT)


def _draw_candles(draw, bars, plot_left, plot_right, plot_h, y_for) -> None:
    columns = max(len(bars), 1)
    column_width = (plot_right - plot_left) / columns
    body_width = max(2.0, column_width * 0.7)

    for index, bar in enumerate(bars):
        center_x = plot_left + (index + 0.5) * column_width
        wick_top = y_for(bar.high)
        wick_bottom = y_for(bar.low)
        body_top = y_for(max(bar.open, bar.close))
        body_bottom = y_for(min(bar.open, bar.close))
        body_color = BULL_BODY if bar.close >= bar.open else BEAR_BODY

        draw.line(
            [(center_x, wick_top), (center_x, wick_bottom)],
            fill=WICK_COLOR,
            width=1,
        )

        if body_bottom - body_top < 1.5:
            body_bottom = body_top + 1.5
        draw.rectangle(
            [center_x - body_width / 2, body_top, center_x + body_width / 2, body_bottom],
            fill=body_color,
            outline=WICK_COLOR,
        )


def _format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 10:
        return f"{price:.2f}"
    return f"{price:.4f}"


def _format_timestamp(timestamp: str) -> str:
    parts = timestamp.split()
    return parts[0] if parts else timestamp
