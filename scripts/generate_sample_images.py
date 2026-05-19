"""Generate synthetic candlestick-style chart images for quick MVP testing.

Reproducible by seed: re-running with the same seed reproduces the same chart,
which lets us crop a query fragment that must match one of the library images.
"""
import argparse
import logging
import random
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def generate_chart(width: int = 480, height: int = 320, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), color=(245, 245, 248))
    draw = ImageDraw.Draw(image)

    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=(225, 225, 232))
    for y in range(0, height, 30):
        draw.line([(0, y), (width, y)], fill=(225, 225, 232))

    candles = 40
    step = width / candles
    price = height / 2
    for i in range(candles):
        change = rng.gauss(0, 12)
        open_p = price
        price += change
        price = max(15.0, min(price, height - 15.0))
        close_p = price

        body_top = min(open_p, close_p)
        body_bottom = max(open_p, close_p)
        wick_top = body_top - rng.uniform(2, 12)
        wick_bottom = body_bottom + rng.uniform(2, 12)
        cx = int(i * step + step / 2)

        color = (45, 160, 80) if close_p < open_p else (200, 60, 60)
        draw.line([(cx, wick_top), (cx, wick_bottom)], fill=(80, 80, 80), width=1)
        draw.rectangle(
            [cx - 4, body_top, cx + 4, body_bottom],
            fill=color,
            outline=(40, 40, 40),
        )

    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate sample chart images")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("samples/library"),
        help="Folder to write generated chart images to",
    )
    parser.add_argument(
        "--query-out",
        type=Path,
        default=Path("samples/query/query.png"),
        help="Path to save the cropped query fragment",
    )
    parser.add_argument(
        "--fragment-source-index",
        type=int,
        default=3,
        help="Which generated chart to crop the query fragment from (1-based)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    args.library_dir.mkdir(parents=True, exist_ok=True)
    args.query_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d sample charts into %s", args.count, args.library_dir)
    for i in range(1, args.count + 1):
        image = generate_chart(seed=i)
        target = args.library_dir / f"chart_{i:03d}.png"
        image.save(target, "PNG")
        logger.debug("Wrote %s (%dx%d)", target, *image.size)

    source_index = max(1, min(args.fragment_source_index, args.count))
    source = generate_chart(seed=source_index)
    width, height = source.size
    crop_box = (width // 4, height // 4, width // 4 + 200, height // 4 + 140)
    fragment = source.crop(crop_box)
    fragment.save(args.query_out, "PNG")
    logger.info(
        "Wrote query fragment %s cropped from chart_%03d.png at %s",
        args.query_out,
        source_index,
        crop_box,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
