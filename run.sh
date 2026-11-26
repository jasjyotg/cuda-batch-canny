#!/usr/bin/env bash
# Builds the project, prepares the datasets and runs canny_batch on each of
# them, saving everything a reviewer needs as proof of execution under
# proof/<timestamp>/ (console log, nvidia-smi, per-image CSVs, summaries and
# input/output montages).
#
# Usage: ./run.sh            full demo (synthetic data + SIPI if available)
#        FETCH_SIPI=1 ./run.sh   also download the USC SIPI volumes first
#        QUICK=1 ./run.sh        fewer/smaller images, for a fast smoke test
set -euo pipefail
cd "$(dirname "$0")"

STAMP="$(date +%Y%m%d_%H%M%S)"
PROOF="proof/${STAMP}"
mkdir -p "${PROOF}"
# Mirror all console output into the proof log.
exec > >(tee "${PROOF}/console.log") 2>&1

echo "=== cuda-batch-canny run ${STAMP} on $(hostname) ==="
echo "--- nvidia-smi ---"
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo "--- nvcc ---"
"${CUDA_PATH:-/usr/local/cuda}/bin/nvcc" --version || nvcc --version || true
nvidia-smi > "${PROOF}/nvidia-smi.txt" 2>&1 || true

echo "--- build ---"
make -j"$(nproc)"

echo "--- datasets ---"
if [[ "${QUICK:-0}" == "1" ]]; then
  SMALL_N=100; COLOR_N=20; LARGE_N=3
else
  SMALL_N=500; COLOR_N=100; LARGE_N=12
fi
gen() {  # gen <dir> <expected count> <generator flags...>
  local dir="$1" count="$2"; shift 2
  local have
  have="$(find "${dir}" -name '*.p?m' 2>/dev/null | wc -l)"
  if [[ "${have}" -lt "${count}" ]]; then
    ./bin/generate_synthetic --output_dir="${dir}" --count="${count}" "$@"
  else
    echo "${dir}: ${have} images already present"
  fi
}
gen data/synthetic_small "${SMALL_N}" --prefix=small --width=256 --height=256 --seed=1000
gen data/synthetic_color "${COLOR_N}" --prefix=color --width=640 --height=480 --color --seed=5000
gen data/synthetic_large "${LARGE_N}" --prefix=large --width=3840 --height=2160 --shapes=60 --seed=9000
if [[ "${FETCH_SIPI:-0}" == "1" ]]; then
  ./scripts/fetch_sipi.sh || echo "WARNING: SIPI download failed; continuing with synthetic data"
fi

# run_case <name> <input dir> [extra canny_batch flags...]
run_case() {
  local name="$1" input="$2"; shift 2
  local out="output/${name}"
  echo
  echo "################ ${name}: $(find "${input}" -name '*.p?m' | wc -l) images from ${input} ################"
  rm -rf "${out}"
  local status=0
  ./bin/canny_batch --input_dir="${input}" --output_dir="${out}" --verify "$@" || status=$?
  cp "${out}/results.csv" "${PROOF}/${name}_results.csv" 2>/dev/null || true
  cp "${out}/summary.txt" "${PROOF}/${name}_summary.txt" 2>/dev/null || true
  if python3 -c "import PIL" 2>/dev/null; then
    python3 scripts/make_montage.py --input_dir="${input}" --edges_dir="${out}/edges" \
        --output="${PROOF}/${name}_montage.png" --title="${name}" || true
  fi
  if [[ ${status} -ne 0 ]]; then
    echo "WARNING: ${name} exited with status ${status}"
  fi
}

# 1) Hundreds of small images: single stream vs. multi-stream overlap.
run_case small_1stream data/synthetic_small --streams=1 --quiet
run_case small_4streams data/synthetic_small --streams=4
# 2) Color (RGB -> gray on the GPU).
run_case color data/synthetic_color --streams=4
# 3) Tens of large 4K images, also exporting gradient-magnitude images.
run_case large_4k data/synthetic_large --streams=4 --save_gradient
# 4) Real photographs from the USC SIPI database, if downloaded.
if [[ -d data/sipi ]]; then
  run_case sipi data/sipi --streams=4
fi

echo
echo "=== done; proof artifacts in ${PROOF}/ ==="
ls -la "${PROOF}"
