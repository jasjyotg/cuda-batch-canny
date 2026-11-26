#!/usr/bin/env bash
# Downloads volumes of the USC-SIPI image database and converts the TIFF
# files to the 8-bit PGM/PPM format read by canny_batch.
#
#   textures: 155 grayscale images, 512x512 and 1024x1024   (~30 MB)
#   aerials :  38 mostly RGB images, 512x512 up to 2250x2250 (~86 MB)
#
# Usage: scripts/fetch_sipi.sh [volume ...]   (default: textures aerials)
# Output: data/sipi/<volume>/*.pgm|*.ppm
set -euo pipefail
cd "$(dirname "$0")/.."

VOLUMES=("$@")
if [[ ${#VOLUMES[@]} -eq 0 ]]; then VOLUMES=(textures aerials); fi
BASE_URL="https://sipi.usc.edu/database"
RAW_DIR="data/sipi_raw"
OUT_DIR="data/sipi"
mkdir -p "${RAW_DIR}" "${OUT_DIR}"

download() {  # download <url> <dest>
  if command -v wget >/dev/null; then
    wget -q --show-progress -O "$2" "$1"
  else
    curl -fL --progress-bar -o "$2" "$1"
  fi
}

for volume in "${VOLUMES[@]}"; do
  archive="${RAW_DIR}/${volume}.tar.gz"
  if [[ ! -s "${archive}" ]]; then
    echo "Downloading ${BASE_URL}/${volume}.tar.gz"
    download "${BASE_URL}/${volume}.tar.gz" "${archive}"
  fi
  mkdir -p "${RAW_DIR}/${volume}"
  tar -xzf "${archive}" -C "${RAW_DIR}/${volume}"
done

echo "Converting TIFF -> PGM/PPM into ${OUT_DIR}/"
if python3 -c "import PIL" 2>/dev/null; then
  python3 - "${RAW_DIR}" "${OUT_DIR}" <<'EOF'
import pathlib
import sys

from PIL import Image

raw, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
count = 0
for path in sorted(raw.rglob("*.tif*")):
    volume = path.relative_to(raw).parts[0]
    image = Image.open(path)
    gray = image.mode in ("1", "L", "P", "I;16", "I", "F")
    image = image.convert("L" if gray else "RGB")
    dest = out / volume / (path.stem + (".pgm" if gray else ".ppm"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    count += 1
print(f"converted {count} images")
EOF
elif command -v convert >/dev/null; then
  find "${RAW_DIR}" -name '*.tif*' | sort | while read -r tif; do
    volume="$(echo "${tif#${RAW_DIR}/}" | cut -d/ -f1)"
    name="$(basename "${tif%.*}")"
    mkdir -p "${OUT_DIR}/${volume}"
    if [[ "$(identify -format '%[channels]' "${tif}" 2>/dev/null)" == gray* ]]; then
      convert "${tif}" -depth 8 "${OUT_DIR}/${volume}/${name}.pgm"
    else
      convert "${tif}" -depth 8 "${OUT_DIR}/${volume}/${name}.ppm"
    fi
  done
else
  echo "Need Python Pillow (pip install pillow) or ImageMagick to convert TIFF." >&2
  exit 1
fi
find "${OUT_DIR}" -name '*.p?m' | wc -l | xargs echo "SIPI images ready:"
