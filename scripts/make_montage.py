#!/usr/bin/env python3
"""Builds a PNG contact sheet of input images next to their GPU edge maps.

Used by run.sh to produce visual proof of execution. Requires Pillow.

Example:
  python3 scripts/make_montage.py --input_dir=data/synthetic_small \
      --edges_dir=output/small_4streams/edges --output=proof/small.png
"""

import argparse
import pathlib
import sys

from PIL import Image, ImageDraw

EXTENSIONS = {".pgm", ".ppm", ".pnm"}


def find_pairs(input_dir, edges_dir):
  """Returns (input_path, edges_path) for every input that has an output."""
  pairs = []
  for path in sorted(input_dir.rglob("*")):
    if path.suffix.lower() not in EXTENSIONS:
      continue
    edges = (edges_dir / path.relative_to(input_dir)).with_suffix(".pgm")
    if edges.exists():
      pairs.append((path, edges))
  return pairs


def fit(image, tile):
  """Scales `image` to fit in a tile x tile box, preserving aspect ratio."""
  image = image.convert("RGB")
  scale = min(tile / image.width, tile / image.height)
  size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
  # Downscaling uses area averaging so 1-pixel edges stay visible; small
  # images are enlarged with nearest-neighbor to keep edges crisp.
  image = image.resize(size, Image.BOX if scale < 1 else Image.NEAREST)
  canvas = Image.new("RGB", (tile, tile), (40, 40, 40))
  canvas.paste(image, ((tile - image.width) // 2, (tile - image.height) // 2))
  return canvas


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input_dir", required=True, type=pathlib.Path)
  parser.add_argument("--edges_dir", required=True, type=pathlib.Path)
  parser.add_argument("--output", required=True, type=pathlib.Path)
  parser.add_argument("--count", type=int, default=8,
                      help="number of input/output pairs to show")
  parser.add_argument("--tile", type=int, default=240)
  parser.add_argument("--columns", type=int, default=2,
                      help="pairs per row")
  parser.add_argument("--title", default="")
  args = parser.parse_args()

  pairs = find_pairs(args.input_dir, args.edges_dir)
  if not pairs:
    print(f"no input/output pairs found in {args.edges_dir}", file=sys.stderr)
    return 1
  # Spread the samples evenly across the whole batch.
  step = max(1, len(pairs) // args.count)
  sample = pairs[::step][:args.count]

  tile, cols, header, label = args.tile, args.columns, 28, 16
  rows = (len(sample) + cols - 1) // cols
  width = cols * 2 * tile + (cols - 1) * 8
  height = header + rows * (tile + label)
  sheet = Image.new("RGB", (width, height), "white")
  draw = ImageDraw.Draw(sheet)
  draw.text((6, 6), f"{args.title}: {len(pairs)} images processed, "
                    f"{len(sample)} shown (left: input, right: GPU edges)",
            fill="black")
  for i, (src, edges) in enumerate(sample):
    x = (i % cols) * (2 * tile + 8)
    y = header + (i // cols) * (tile + label)
    sheet.paste(fit(Image.open(src), tile), (x, y))
    sheet.paste(fit(Image.open(edges), tile), (x + tile, y))
    draw.text((x + 2, y + tile + 2), str(src.relative_to(args.input_dir)),
              fill="black")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  sheet.save(args.output)
  print(f"wrote {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
