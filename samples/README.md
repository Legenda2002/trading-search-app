# Samples

Folder layout used by the test workflow:

```
samples/
  library/    # chart images to import into the local library
  query/      # query fragment images
```

## Generate synthetic samples

```bash
python -m scripts.generate_sample_images --count 10
```

This creates:

- `samples/library/chart_001.png` ... `chart_010.png` — candlestick-style charts.
- `samples/query/query.png` — a fragment cropped from `chart_003.png`.

Since the generator is deterministic per seed, the cropped fragment is
guaranteed to exist in the library, so the search should rank the source chart
near the top.

## Use your own samples

Drop any `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.webp` files into:

- `samples/library/` — your chart library.
- `samples/query/` — small fragment images you want to search for.
