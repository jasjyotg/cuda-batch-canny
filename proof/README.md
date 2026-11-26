# Proof of execution

Each `./run.sh` invocation creates `proof/<timestamp>/` containing:

| File | Contents |
|---|---|
| `console.log` | full console output: nvidia-smi, nvcc version, GPU description, one line per processed image, summaries |
| `nvidia-smi.txt` | GPU / driver used for the run |
| `<case>_results.csv` | per-image metrics (size, GPU latency, edge pixels, hysteresis sweeps, CPU time, mismatched pixels) |
| `<case>_summary.txt` | throughput, host time split, verification verdict, speedup vs. CPU |
| `<case>_montage.png` | sample inputs (left) next to their GPU edge maps (right) |

Cases: `small_1stream`, `small_4streams` (500 × 256²), `color` (100 × 640×480 RGB),
`large_4k` (12 × 3840×2160) and, with `FETCH_SIPI=1`, `sipi` (USC-SIPI textures + aerials).
