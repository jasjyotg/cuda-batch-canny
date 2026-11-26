# cuda-batch-canny: batched multi-stream Canny edge detection in CUDA

`canny_batch` finds edges in **hundreds of small images or tens of large
images in a single execution**. Every image goes through a complete Canny
edge detector implemented as eight custom CUDA kernels. Several images are in
flight at once on separate CUDA streams, so file I/O, host-device copies and
compute overlap. An optional `--verify` mode re-runs every image through a
CPU reference implementation and checks that the GPU output is
**bit-identical**.

The project was written for the *CUDA at Scale for the Enterprise* course
project. It uses hand-written kernels rather than NPP so that it can fuse
stages, control numerical behaviour exactly, and show stream-level
concurrency.

---

## 1. What the program does

```
 PGM/PPM files ──► pinned host buffer ──H2D──► ToGray ──► Gaussian (row) ──► Gaussian (col)
                                                                                   │
  edge PGM files ◄── pinned host buffer ◄──D2H── Finalize ◄── Hysteresis ×N ◄── DoubleThreshold
                                                     ▲                               ▲
                                           edge count (reduction)        NonMaxSuppression + Sobel
                                                                         (+ max-gradient reduction)
```

For each image:

1. **Grayscale**: RGB is converted to BT.601 luma. Grayscale input is cast
   to float.
2. **Gaussian blur**: a separable filter (σ = 1.4 by default, radius
   ⌈3σ⌉). Two passes cost O(r) per pixel instead of O(r²).
3. **Sobel gradient**: computes the magnitude and a direction quantized to
   4 bins.
4. **Non-maximum suppression**: thins edges to 1 px. The same kernel also
   computes the image's maximum gradient.
5. **Double threshold**: pixels are labelled strong (≥ `high`·max), weak
   (≥ `low`·max) or none. Thresholds are relative to each image's max
   gradient, so one setting works across images with very different
   contrast.
6. **Hysteresis**: weak pixels that are 8-connected to strong pixels
   become edges.
7. **Finalize**: labels become a 0/255 edge map, edge pixels are counted
   and an optional gradient-magnitude image is rendered.

## 2. GPU kernels and CUDA techniques

| Kernel | Work | Techniques |
|---|---|---|
| `ToGrayKernel` | u8 RGB/gray → float | grid-stride loop, coalesced access |
| `GaussianRowKernel` / `GaussianColKernel` | separable blur | **dynamic shared-memory tiles with an apron** of radius r; weights in **`__constant__` memory** (broadcast to every thread of a warp); replicate borders |
| `SobelKernel` | gx, gy, magnitude, direction | 18×18 shared tile for a 16×16 block (1-px halo) |
| `NonMaxSuppressionKernel` | edge thinning + per-image max | shared tile + **warp-shuffle block reduction** (`__shfl_down_sync`) + one `atomicMax` per block on the float's bit pattern |
| `DoubleThresholdKernel` | label strong/weak/none | reads the max from device memory, so the host never synchronizes |
| `HysteresisKernel` | edge tracking | **self-terminating iterative sweeps**, in-tile propagation in shared memory, persistent-style grid (see §4) |
| `FinalizeKernel` | 0/255 map, edge count, gradient image | block reduction + 64-bit `atomicAdd` |

### Host pipeline (`GpuCannyPipeline`)

* **N slots (`--streams`, default 4).** Each slot has its own
  `cudaStreamNonBlocking` stream, **pinned (page-locked) staging buffers**
  (needed for real asynchronous `cudaMemcpyAsync`), device buffers and a
  pair of CUDA events.
* **Round-robin scheduling.** For image *i*, the host *first reads the
  file from disk*, then waits only on slot `i % N`, writes that slot's
  finished result, and queues the new image. While the host does file I/O,
  the GPU keeps working on the other N−1 images.
* **No synchronization inside an image.** The thresholds use the max
  gradient stored on the device, and hysteresis decides on the device when
  it has converged. The host only synchronizes once per image, on that
  slot's completion event.
* **Buffers are allocated before the batch starts.** The program reads
  every file header first and sizes all slots for the largest image.
  Growing a buffer later would mean calling `cudaFree`, which synchronizes
  the whole device and destroys the overlap.
* **Buffer reuse.** Three float buffers cover six logical stages
  (gray→magnitude, row-blur→NMS, blur).

## 3. Repository layout

```
├── Makefile               build (auto-detects GPU arch), data, run, format, lint
├── run.sh                 end-to-end demo that writes proof artifacts to proof/<timestamp>/
├── INSTALL                build requirements and instructions
├── src/
│   ├── main.cc            CLI, directory scan, stream scheduling loop, CSV + summary
│   ├── canny_gpu.h/.cu    CUDA kernels + multi-stream GpuCannyPipeline
│   ├── canny_cpu.h/.cc    single-threaded reference (used only by --verify)
│   ├── canny_ops.h        __host__ __device__ per-pixel math shared by CPU and GPU
│   ├── canny_params.h/.cc parameters, validation, Gaussian weights
│   ├── image_io.h/.cc     dependency-free PGM/PPM reader/writer
│   ├── flag_parser.h/.cc  small gflags-style command-line parser
│   ├── cuda_check.h       CUDA_CHECK error macros
│   ├── timer.h            steady-clock stopwatch
│   └── tools/generate_synthetic.cc   reproducible synthetic dataset generator
├── scripts/
│   ├── fetch_sipi.sh      download USC-SIPI volumes and convert TIFF → PGM/PPM
│   └── make_montage.py    input-vs-edges contact sheet (Pillow)
├── data/                  datasets (generated or downloaded; git-ignored)
├── proof/                 committed evidence of execution (logs, CSVs, montages)
├── bin/  lib/             build output / third-party libs (none needed)
```

## 4. Build and run

Requirements: NVIDIA GPU and CUDA Toolkit ≥ 11 (C++17), g++ ≥ 9, make.
Optional: Python 3 + Pillow (montages, SIPI conversion). See `INSTALL`.

```bash
make                 # builds bin/canny_batch and bin/generate_synthetic
./run.sh             # generates data, runs all cases, writes proof/<timestamp>/
FETCH_SIPI=1 ./run.sh   # also downloads and processes the USC-SIPI images
QUICK=1 ./run.sh        # smaller smoke test
```

Run directly on any folder of 8-bit PGM/PPM images (searched recursively):

```bash
./bin/canny_batch --input_dir=data/synthetic_small --output_dir=output/small \
                  --streams=4 --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--input_dir` | (required) | directory searched recursively for `.pgm/.ppm/.pnm` |
| `--output_dir` | `output` | edge maps go to `edges/`, plus `results.csv` and `summary.txt` |
| `--sigma` | 1.4 | Gaussian σ in pixels, (0, 5] |
| `--low`, `--high` | 0.05, 0.12 | hysteresis thresholds as fractions of the image's max gradient |
| `--hysteresis_iters` | 100 | maximum GPU hysteresis sweeps; sweeps after convergence return immediately |
| `--streams` | 4 | CUDA streams, i.e. images processed concurrently (1–32) |
| `--device` | 0 | CUDA device ordinal |
| `--max_images` | 0 (all) | limit the batch size |
| `--save_gradient` | false | also write gradient-magnitude images |
| `--verify` | false | run the CPU reference on every image and compare pixel by pixel |
| `--verify_tolerance` | 0.0001 | maximum mismatched-pixel fraction for PASS |
| `--quiet` | false | hide per-image log lines |

Exit codes: `0` success, `1` I/O or usage error, `2` verification failed.

## 5. Data

* **Synthetic (always available, no network).** `bin/generate_synthetic`
  draws reproducible scenes: a gradient background; random disks, rings,
  rotated rectangles and triangles; and Gaussian sensor noise. `run.sh`
  creates three sets:
  * `synthetic_small`: **500** × 256×256 grayscale
  * `synthetic_color`: **100** × 640×480 RGB
  * `synthetic_large`: **12** × 3840×2160 (4K) grayscale, 8.3 MP each
* **USC-SIPI (real images).** `scripts/fetch_sipi.sh` downloads two
  volumes. *Textures* has 155 grayscale images at 512² and 1024². *Aerials*
  has 38 mostly RGB images, from 512² up to 2250². The script converts both
  to PGM/PPM.

## 6. Proof of execution

`run.sh` saves the following under `proof/<timestamp>/`:

* `console.log`: the full console output, including `nvidia-smi`,
  `nvcc --version`, the GPU name and compute capability as reported by the
  program, and **one log line per image** (slot/stream, size, GPU latency,
  edge density, hysteresis sweeps). This shows that every image of a
  dataset was processed by one execution of `canny_batch`.
* `nvidia-smi.txt`: the GPU and driver that were used.
* `<case>_results.csv`: per-image metrics, plus CPU time and mismatched
  pixels from `--verify`.
* `<case>_summary.txt`: throughput (images/s, MPix/s), time split
  (disk read / write / waiting for the GPU), verification verdict and
  end-to-end speedup over the CPU reference.
* `<case>_montage.png`: sample inputs next to their GPU edge maps.

The cases are `small_1stream` and `small_4streams` (the same 500 images, to
measure what stream overlap is worth), `color`, `large_4k` and, if
downloaded, `sipi`.

## 7. Verification: making GPU and CPU agree bit for bit

Comparing floating-point pipelines usually needs a tolerance. This project
aims for exact equality, which took the following decisions:

* **One source of truth for the math.** `canny_ops.h` holds every per-pixel
  formula (luma, Sobel, magnitude, direction, NMS test, classification) as
  `__host__ __device__` functions. The CPU reference and the kernels call
  the same code, with the same accumulation order.
* **FMA contraction.** By default nvcc fuses `a*b + c` into one FMA
  instruction, which rounds once instead of twice, while the x86-64 host
  build does not. Blurred values can then differ in the last bit, and a
  pixel sitting exactly on a threshold can flip between edge and
  non-edge. The project compiles with `--fmad=false`, trading a little
  throughput for reproducible results.
* **No `atan2`.** Direction is binned by comparing |gy| against |gx|·tan(22.5°)
  and |gx|·tan(67.5°). This needs only multiplications and comparisons,
  which are correctly rounded on both host and device. The `atan2`
  implementations on host and device can differ by an ULP.
* **The Gaussian weights are computed once** on the host in double
  precision and uploaded. Both sides use the identical float table.

## 8. Challenges and lessons learned

* **Hysteresis is the one part that is not naturally parallel.** It is a
  connected-components problem: a weak pixel becomes an edge if any chain
  of weak pixels links it to a strong one. A naive GPU version repeats
  "promote weak pixels next to strong ones" until nothing changes. That
  needs a device-to-host flag copy and a synchronization after every
  sweep, which would stall the stream pipeline. My solution:
  1. *Self-terminating sweeps.* The host enqueues up to
     `--hysteresis_iters` sweeps without waiting. Sweep *k* writes
     `flag[k]` if it promoted anything. Sweep *k+1* returns immediately if
     `flag[k]` is 0. The flags are copied back with the result, so the log
     reports the actual number of sweeps and whether hysteresis converged.
  2. *In-tile propagation.* Each block loads a 16×16 tile plus halo into
     shared memory and repeats promotion there until the tile is stable.
     Separate read and write phases with barriers between them keep this
     free of data races. One sweep can therefore carry an edge across a
     whole tile, not just one pixel.
  3. *Persistent-style grid.* Instead of one block per tile, the kernel
     launches about 8 blocks per SM, which loop over tiles. Sweeps that
     return early then cost only a few microseconds. Tiles updated earlier
     in a sweep also speed up later tiles in the same sweep.
  On the synthetic test scenes hysteresis converged in 1–4 sweeps. The
  per-image counts are in `results.csv`.
* **Async copies need pinned memory.** With pageable buffers,
  `cudaMemcpyAsync` is effectively synchronous and streams do not overlap.
* **`cudaFree` and `cudaFreeHost` synchronize the device.** Growing buffers
  in the middle of a batch would serialize every stream. The program
  reads all headers first and allocates once to avoid this.
* **Order host work to hide latency.** Reading image *i+N* from disk
  *before* waiting on its slot lets disk I/O overlap with GPU compute.
  `summary.txt` splits host time into read, write and wait-for-GPU to show
  whether a run is I/O-bound or GPU-bound.
* **Event timing with several streams measures latency, not cost.** A
  stream's start-to-stop interval includes time spent queued behind other
  streams. I report per-image GPU latency separately from overall batch
  throughput, and I compare the CPU against end-to-end wall time
  (including I/O), which is the conservative choice.
* **Algorithm trade-offs that show up in the montages:**
  * *Relative thresholds* adapt to contrast, but on very flat, noisy
    images they let noise through.
  * *Luma conversion* hides edges between colors of equal brightness. Some
    shapes in the color set disappear for this reason.
  * A per-channel gradient (di Zenzo) would fix the luma problem; it is a
    possible extension.
* **Why not NPP?** NPP has a Canny primitive (`nppiFilterCannyBorder`), but
  it is a black box. Custom kernels let me fuse the max reduction into
  NMS and the edge count into Finalize. They also made bit-exact
  verification and the stream scheduling explicit, which is what the
  course is about.

## 9. Code style

The C++ follows the Google C++ Style Guide:

* Formatting is enforced with `clang-format` (`.clang-format`,
  `BasedOnStyle: Google`, `make format`). `make lint` runs cpplint when it
  is installed.
* Naming: `CamelCase` types and functions, `snake_case` variables,
  trailing `_` on members, `kConstant` constants.
* Every header has an include guard.
* No exceptions: CUDA errors go through `CUDA_CHECK`, which logs and exits.
* RAII and `std::unique_ptr` pimpl, so `canny_gpu.h` exposes no CUDA types
  and is included from plain C++.
* Non-copyable classes are marked with `= delete`.
* Every file carries a copyright header.

## License

MIT; see `LICENSE`. USC-SIPI images are subject to the SIPI database's own
terms. They are downloaded at run time and are not redistributed here.
