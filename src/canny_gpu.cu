// Copyright 2026 The cuda-batch-canny Authors.
//
// CUDA implementation of the batched Canny pipeline. Per image, on the
// slot's own stream:
//
//   H2D copy (pinned) -> ToGray -> GaussianRow -> GaussianCol -> Sobel
//   -> NonMaxSuppression (+ block max reduction) -> DoubleThreshold
//   -> Hysteresis x N (self-terminating) -> Finalize (+ edge count) -> D2H
//
// Nothing in that chain blocks the host, so several images are in flight at
// once on different streams while the host does file I/O.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "canny_gpu.h"
#include "canny_ops.h"
#include "cuda_check.h"

namespace canny {
namespace {

constexpr int kThreads1D = 256;
constexpr int kTile = 16;  // 2-D kernels use kTile x kTile thread blocks.
constexpr int kHaloTile = kTile + 2;
constexpr int kRowBlockX = 32;
constexpr int kRowBlockY = 8;
constexpr int kColBlockX = 32;
constexpr int kColBlockY = 16;
// Upper bound on in-shared-memory propagation rounds per tile per sweep.
constexpr int kLocalHysteresisIters = 64;

// Gaussian weights live in constant memory: every thread of a warp reads the
// same weight at the same time, which constant memory broadcasts in one
// transaction. One pipeline per process owns this table.
__constant__ float c_gaussian[2 * kMaxGaussianRadius + 1];

struct DeviceStats {
  unsigned int max_magnitude_bits;  // Bits of a non-negative float, so
                                    // integer atomicMax orders correctly.
  unsigned int padding;
  unsigned long long edge_count;  // NOLINT(runtime/int): atomicAdd overload.
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

struct MaxOp {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct SumOp {
  __device__ unsigned int operator()(unsigned int a, unsigned int b) const {
    return a + b;
  }
};

// Reduces `value` over the whole thread block with warp shuffles; the result
// is valid in thread 0. The block size must be a multiple of 32. Every
// thread of the block must call this (it contains __syncthreads).
template <typename T, typename Op>
__device__ T BlockReduce(T value, T identity, Op op) {
  __shared__ T warp_results[32];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (blockDim.x * blockDim.y) >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = op(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  if (lane == 0) warp_results[warp] = value;
  __syncthreads();
  T result = identity;
  if (warp == 0) {
    result = lane < num_warps ? warp_results[lane] : identity;
    for (int offset = 16; offset > 0; offset >>= 1) {
      result = op(result, __shfl_down_sync(0xffffffffu, result, offset));
    }
  }
  return result;
}

// Cooperatively copies the tile_w x tile_h window whose top-left pixel is
// (x0, y0) into shared memory, replicating edge pixels for out-of-range
// coordinates. Consecutive threads read consecutive addresses (coalesced).
template <typename T>
__device__ void LoadTileClamped(const T* src, int width, int height, int x0,
                                int y0, int tile_w, int tile_h, T* tile) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * blockDim.y;
  for (int i = tid; i < tile_w * tile_h; i += num_threads) {
    const int ty = i / tile_w;
    const int tx = i - ty * tile_w;
    const int gx = ClampInt(x0 + tx, 0, width - 1);
    const int gy = ClampInt(y0 + ty, 0, height - 1);
    tile[i] = src[static_cast<size_t>(gy) * width + gx];
  }
}

__device__ inline bool HasStrongNeighbor(const uint8_t* tile, int c,
                                         int stride) {
  return tile[c - stride - 1] == kLabelStrong ||
         tile[c - stride] == kLabelStrong ||
         tile[c - stride + 1] == kLabelStrong || tile[c - 1] == kLabelStrong ||
         tile[c + 1] == kLabelStrong || tile[c + stride - 1] == kLabelStrong ||
         tile[c + stride] == kLabelStrong ||
         tile[c + stride + 1] == kLabelStrong;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void ToGrayKernel(const uint8_t* __restrict__ input, int channels,
                             size_t n, float* __restrict__ gray) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    gray[i] = channels == 3 ? GrayFromRgb(input[3 * i], input[3 * i + 1],
                                          input[3 * i + 2])
                            : static_cast<float>(input[i]);
  }
}

// Horizontal pass of the separable Gaussian. Each block stages its rows plus
// a `radius`-wide apron on both sides in dynamic shared memory.
__global__ void GaussianRowKernel(const float* __restrict__ src, int width,
                                  int height, int radius,
                                  float* __restrict__ dst) {
  extern __shared__ float tile[];
  const int tile_w = blockDim.x + 2 * radius;
  const int bx = static_cast<int>(blockIdx.x * blockDim.x);
  const int by = static_cast<int>(blockIdx.y * blockDim.y);
  LoadTileClamped(src, width, height, bx - radius, by, tile_w,
                  static_cast<int>(blockDim.y), tile);
  __syncthreads();
  const int x = bx + threadIdx.x;
  const int y = by + threadIdx.y;
  if (x >= width || y >= height) return;
  const float* row = tile + threadIdx.y * tile_w + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k <= 2 * radius; ++k) acc += c_gaussian[k] * row[k];
  dst[static_cast<size_t>(y) * width + x] = acc;
}

// Vertical pass; the apron is above and below the block.
__global__ void GaussianColKernel(const float* __restrict__ src, int width,
                                  int height, int radius,
                                  float* __restrict__ dst) {
  extern __shared__ float tile[];
  const int tile_w = static_cast<int>(blockDim.x);
  const int tile_h = blockDim.y + 2 * radius;
  const int bx = static_cast<int>(blockIdx.x * blockDim.x);
  const int by = static_cast<int>(blockIdx.y * blockDim.y);
  LoadTileClamped(src, width, height, bx, by - radius, tile_w, tile_h, tile);
  __syncthreads();
  const int x = bx + threadIdx.x;
  const int y = by + threadIdx.y;
  if (x >= width || y >= height) return;
  const float* col = tile + threadIdx.y * tile_w + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k <= 2 * radius; ++k) acc += c_gaussian[k] * col[k * tile_w];
  dst[static_cast<size_t>(y) * width + x] = acc;
}

__global__ void SobelKernel(const float* __restrict__ src, int width,
                            int height, float* __restrict__ magnitude,
                            uint8_t* __restrict__ direction) {
  __shared__ float tile[kHaloTile * kHaloTile];
  const int bx = static_cast<int>(blockIdx.x) * kTile;
  const int by = static_cast<int>(blockIdx.y) * kTile;
  LoadTileClamped(src, width, height, bx - 1, by - 1, kHaloTile, kHaloTile,
                  tile);
  __syncthreads();
  const int x = bx + threadIdx.x;
  const int y = by + threadIdx.y;
  if (x >= width || y >= height) return;
  const int s = kHaloTile;
  const float* c = tile + (threadIdx.y + 1) * s + threadIdx.x + 1;
  float gx, gy;
  Sobel3x3(c[-s - 1], c[-s], c[-s + 1], c[-1], c[1], c[s - 1], c[s], c[s + 1],
           &gx, &gy);
  const size_t i = static_cast<size_t>(y) * width + x;
  magnitude[i] = GradientMagnitude(gx, gy);
  direction[i] = QuantizeDirection(gx, gy);
}

// Thins edges to one pixel and, in the same pass, reduces the per-image
// maximum magnitude (needed for the relative thresholds) with one atomic per
// block instead of a separate reduction kernel.
__global__ void NonMaxSuppressionKernel(const float* __restrict__ magnitude,
                                        const uint8_t* __restrict__ direction,
                                        int width, int height,
                                        float* __restrict__ nms,
                                        DeviceStats* stats) {
  __shared__ float tile[kHaloTile * kHaloTile];
  const int bx = static_cast<int>(blockIdx.x) * kTile;
  const int by = static_cast<int>(blockIdx.y) * kTile;
  LoadTileClamped(magnitude, width, height, bx - 1, by - 1, kHaloTile,
                  kHaloTile, tile);
  __syncthreads();
  const int x = bx + threadIdx.x;
  const int y = by + threadIdx.y;
  float value = 0.0f;
  if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
    const size_t i = static_cast<size_t>(y) * width + x;
    int dx, dy;
    DirectionOffset(direction[i], &dx, &dy);
    const int s = kHaloTile;
    const float* c = tile + (threadIdx.y + 1) * s + threadIdx.x + 1;
    value = SuppressNonMax(c[0], c[dy * s + dx], c[-dy * s - dx]);
  }
  if (x < width && y < height) nms[static_cast<size_t>(y) * width + x] = value;
  // All threads (including out-of-image ones) take part in the reduction.
  const float block_max = BlockReduce(value, 0.0f, MaxOp());
  if (threadIdx.x == 0 && threadIdx.y == 0 && block_max > 0.0f) {
    atomicMax(&stats->max_magnitude_bits, __float_as_uint(block_max));
  }
}

__global__ void DoubleThresholdKernel(const float* __restrict__ nms, size_t n,
                                      float low_ratio, float high_ratio,
                                      const DeviceStats* stats,
                                      uint8_t* __restrict__ labels) {
  const float max_m = __uint_as_float(stats->max_magnitude_bits);
  const float low = low_ratio * max_m;
  const float high = high_ratio * max_m;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    labels[i] = ClassifyPixel(nms[i], low, high);
  }
}

// One hysteresis sweep. Blocks loop over 16x16 tiles; inside a tile, weak
// pixels touching strong ones are promoted repeatedly in shared memory until
// the tile stops changing, so a single sweep can carry an edge across a
// whole tile. Blocks may read halo pixels that another block is updating in
// the same sweep and see the old value; labels only ever go weak -> strong,
// so a stale read can delay convergence but never break it: a sweep that
// promotes nothing wrote nothing, so everything it read was current.
// Sweeps are launched back-to-back without host involvement:
// sweep k exits immediately if sweep k-1 promoted nothing (converged), and
// records in iter_flags[k] whether it promoted anything itself.
__global__ void HysteresisKernel(uint8_t* labels, int width, int height,
                                 int tiles_x, int num_tiles, int iteration,
                                 int* iter_flags) {
  if (iteration > 0 && iter_flags[iteration - 1] == 0) return;
  __shared__ uint8_t tile[kHaloTile * kHaloTile];
  __shared__ int block_changed;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int c = (threadIdx.y + 1) * kHaloTile + threadIdx.x + 1;
  bool any_promoted = false;
  for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
    const int x0 = (t % tiles_x) * kTile;
    const int y0 = (t / tiles_x) * kTile;
    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;
    const bool inside = x < width && y < height;
    LoadTileClamped(labels, width, height, x0 - 1, y0 - 1, kHaloTile, kHaloTile,
                    tile);
    __syncthreads();
    bool promoted_here = false;
    for (int round = 0; round < kLocalHysteresisIters; ++round) {
      if (tid == 0) block_changed = 0;
      // Read phase and write phase are separated by a barrier so no thread
      // reads a neighbor while another thread is writing it.
      const bool promote = inside && tile[c] == kLabelWeak &&
                           HasStrongNeighbor(tile, c, kHaloTile);
      __syncthreads();
      if (promote) {
        tile[c] = kLabelStrong;
        block_changed = 1;
        promoted_here = true;
      }
      __syncthreads();
      const int changed = block_changed;
      __syncthreads();
      if (!changed) break;
    }
    if (promoted_here) {
      labels[static_cast<size_t>(y) * width + x] = kLabelStrong;
      any_promoted = true;
    }
    __syncthreads();  // The tile is reloaded on the next loop iteration.
  }
  if (any_promoted) iter_flags[iteration] = 1;
}

// Converts labels to the final 0/255 edge map in place, counts edge pixels
// and optionally renders the gradient magnitude for visualization.
__global__ void FinalizeKernel(uint8_t* __restrict__ labels,
                               const float* __restrict__ magnitude, size_t n,
                               uint8_t* __restrict__ gradient,
                               DeviceStats* stats) {
  const float max_m = __uint_as_float(stats->max_magnitude_bits);
  unsigned int count = 0;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const bool is_edge = labels[i] == kLabelStrong;
    labels[i] = is_edge ? kEdgeValue : 0;
    count += is_edge ? 1u : 0u;
    if (gradient != nullptr) gradient[i] = GradientToByte(magnitude[i], max_m);
  }
  const unsigned int block_count = BlockReduce(count, 0u, SumOp());
  if (threadIdx.x == 0 && block_count > 0) {
    atomicAdd(&stats->edge_count,
              static_cast<unsigned long long>(block_count));  // NOLINT
  }
}

// ---------------------------------------------------------------------------
// Host-side slot management
// ---------------------------------------------------------------------------

template <typename T>
void DeviceAlloc(T** ptr, size_t count) {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)));
}

template <typename T>
void HostAlloc(T** ptr, size_t count) {
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(ptr), count * sizeof(T)));
}

template <typename T>
void DeviceFree(T** ptr) {
  if (*ptr != nullptr) CUDA_CHECK(cudaFree(*ptr));
  *ptr = nullptr;
}

template <typename T>
void HostFree(T** ptr) {
  if (*ptr != nullptr) CUDA_CHECK(cudaFreeHost(*ptr));
  *ptr = nullptr;
}

}  // namespace

bool InitCudaDevice(int device, std::string* description, std::string* error) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    *error = std::string("no usable CUDA device: ") +
             (status != cudaSuccess ? cudaGetErrorString(status)
                                    : "device count is 0");
    return false;
  }
  if (device < 0 || device >= count) {
    *error =
        "--device out of range (found " + std::to_string(count) + " device(s))";
    return false;
  }
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  char buffer[512];
  std::snprintf(
      buffer, sizeof(buffer),
      "GPU %d: %s | compute capability %d.%d | %d SMs | "
      "%.1f GiB global memory | CUDA driver %d.%d, runtime %d.%d",
      device, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
      static_cast<double>(prop.totalGlobalMem) / (1 << 30), driver / 1000,
      (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);
  *description = buffer;
  return true;
}

struct GpuCannyPipeline::Impl {
  struct Slot {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    size_t capacity_pixels = 0;
    int capacity_channels = 0;
    bool busy = false;
    int width = 0;
    int height = 0;

    // Pinned host staging buffers (required for truly async copies).
    uint8_t* h_input = nullptr;
    uint8_t* h_edges = nullptr;
    uint8_t* h_gradient = nullptr;
    DeviceStats* h_stats = nullptr;
    int* h_iter_flags = nullptr;

    // Device buffers. Float buffers are reused across stages:
    //   f0: gray -> gradient magnitude, f1: row-blur tmp -> NMS, f2: blurred.
    uint8_t* d_input = nullptr;
    float* d_f0 = nullptr;
    float* d_f1 = nullptr;
    float* d_f2 = nullptr;
    uint8_t* d_direction = nullptr;
    uint8_t* d_labels = nullptr;
    uint8_t* d_gradient = nullptr;
    DeviceStats* d_stats = nullptr;
    int* d_iter_flags = nullptr;
  };

  CannyParams params;
  int radius = 0;
  bool keep_gradient = false;
  int sm_count = 1;
  std::vector<Slot> slots;

  int Grid1D(size_t n) const {
    const size_t blocks = (n + kThreads1D - 1) / kThreads1D;
    const size_t cap = static_cast<size_t>(sm_count) * 32;
    return static_cast<int>(std::max<size_t>(1, std::min(blocks, cap)));
  }

  void FreeImageBuffers(Slot* slot) {
    HostFree(&slot->h_input);
    HostFree(&slot->h_edges);
    HostFree(&slot->h_gradient);
    DeviceFree(&slot->d_input);
    DeviceFree(&slot->d_f0);
    DeviceFree(&slot->d_f1);
    DeviceFree(&slot->d_f2);
    DeviceFree(&slot->d_direction);
    DeviceFree(&slot->d_labels);
    DeviceFree(&slot->d_gradient);
    slot->capacity_pixels = 0;
    slot->capacity_channels = 0;
  }

  // Grows the slot's per-image buffers if needed. The slot must be idle.
  void EnsureCapacity(Slot* slot, size_t pixels, int channels) {
    if (pixels <= slot->capacity_pixels &&
        channels <= slot->capacity_channels) {
      return;
    }
    pixels = std::max(pixels, slot->capacity_pixels);
    channels = std::max(channels, slot->capacity_channels);
    FreeImageBuffers(slot);
    HostAlloc(&slot->h_input, pixels * channels);
    HostAlloc(&slot->h_edges, pixels);
    DeviceAlloc(&slot->d_input, pixels * channels);
    DeviceAlloc(&slot->d_f0, pixels);
    DeviceAlloc(&slot->d_f1, pixels);
    DeviceAlloc(&slot->d_f2, pixels);
    DeviceAlloc(&slot->d_direction, pixels);
    DeviceAlloc(&slot->d_labels, pixels);
    if (keep_gradient) {
      HostAlloc(&slot->h_gradient, pixels);
      DeviceAlloc(&slot->d_gradient, pixels);
    }
    slot->capacity_pixels = pixels;
    slot->capacity_channels = channels;
  }
};

GpuCannyPipeline::GpuCannyPipeline(const CannyParams& params, int num_slots,
                                   bool keep_gradient)
    : impl_(std::make_unique<Impl>()) {
  impl_->params = params;
  impl_->keep_gradient = keep_gradient;
  impl_->radius = GaussianRadius(params.sigma);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&impl_->sm_count,
                                    cudaDevAttrMultiProcessorCount, device));

  const std::vector<float> weights = BuildGaussianKernel(params.sigma);
  CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian, weights.data(),
                                weights.size() * sizeof(float)));

  impl_->slots.resize(std::max(1, num_slots));
  for (Impl::Slot& slot : impl_->slots) {
    // Non-blocking streams do not implicitly synchronize with stream 0.
    CUDA_CHECK(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&slot.start));
    CUDA_CHECK(cudaEventCreate(&slot.stop));
    HostAlloc(&slot.h_stats, 1);
    HostAlloc(&slot.h_iter_flags, params.max_hysteresis_iters);
    DeviceAlloc(&slot.d_stats, 1);
    DeviceAlloc(&slot.d_iter_flags, params.max_hysteresis_iters);
  }
}

GpuCannyPipeline::~GpuCannyPipeline() {
  for (Impl::Slot& slot : impl_->slots) {
    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
    impl_->FreeImageBuffers(&slot);
    HostFree(&slot.h_stats);
    HostFree(&slot.h_iter_flags);
    DeviceFree(&slot.d_stats);
    DeviceFree(&slot.d_iter_flags);
    CUDA_CHECK(cudaEventDestroy(slot.start));
    CUDA_CHECK(cudaEventDestroy(slot.stop));
    CUDA_CHECK(cudaStreamDestroy(slot.stream));
  }
}

int GpuCannyPipeline::num_slots() const {
  return static_cast<int>(impl_->slots.size());
}

void GpuCannyPipeline::Reserve(int max_width, int max_height,
                               int max_channels) {
  const size_t pixels = static_cast<size_t>(max_width) * max_height;
  for (Impl::Slot& slot : impl_->slots) {
    impl_->EnsureCapacity(&slot, pixels, max_channels);
  }
}

void GpuCannyPipeline::Submit(int slot_index, const Image& image) {
  Impl::Slot& slot = impl_->slots[slot_index];
  if (slot.busy) {
    std::fprintf(stderr, "internal error: Submit() on busy slot %d\n",
                 slot_index);
    std::exit(EXIT_FAILURE);
  }
  const int w = image.width;
  const int h = image.height;
  const size_t n = image.NumPixels();
  const int iters = impl_->params.max_hysteresis_iters;
  const int radius = impl_->radius;
  impl_->EnsureCapacity(&slot, n, image.channels);
  std::memcpy(slot.h_input, image.pixels.data(), image.SizeBytes());

  cudaStream_t s = slot.stream;
  CUDA_CHECK(cudaEventRecord(slot.start, s));
  CUDA_CHECK(cudaMemcpyAsync(slot.d_input, slot.h_input, image.SizeBytes(),
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemsetAsync(slot.d_stats, 0, sizeof(DeviceStats), s));
  CUDA_CHECK(cudaMemsetAsync(slot.d_iter_flags, 0, iters * sizeof(int), s));

  const int grid_1d = impl_->Grid1D(n);
  ToGrayKernel<<<grid_1d, kThreads1D, 0, s>>>(slot.d_input, image.channels, n,
                                              slot.d_f0);
  CUDA_CHECK_LAUNCH();

  const dim3 row_block(kRowBlockX, kRowBlockY);
  const dim3 row_grid((w + kRowBlockX - 1) / kRowBlockX,
                      (h + kRowBlockY - 1) / kRowBlockY);
  const size_t row_smem =
      (kRowBlockX + 2 * radius) * kRowBlockY * sizeof(float);
  GaussianRowKernel<<<row_grid, row_block, row_smem, s>>>(slot.d_f0, w, h,
                                                          radius, slot.d_f1);
  CUDA_CHECK_LAUNCH();

  const dim3 col_block(kColBlockX, kColBlockY);
  const dim3 col_grid((w + kColBlockX - 1) / kColBlockX,
                      (h + kColBlockY - 1) / kColBlockY);
  const size_t col_smem =
      kColBlockX * (kColBlockY + 2 * radius) * sizeof(float);
  GaussianColKernel<<<col_grid, col_block, col_smem, s>>>(slot.d_f1, w, h,
                                                          radius, slot.d_f2);
  CUDA_CHECK_LAUNCH();

  const dim3 tile_block(kTile, kTile);
  const int tiles_x = (w + kTile - 1) / kTile;
  const int tiles_y = (h + kTile - 1) / kTile;
  const dim3 tile_grid(tiles_x, tiles_y);
  SobelKernel<<<tile_grid, tile_block, 0, s>>>(slot.d_f2, w, h, slot.d_f0,
                                               slot.d_direction);
  CUDA_CHECK_LAUNCH();

  NonMaxSuppressionKernel<<<tile_grid, tile_block, 0, s>>>(
      slot.d_f0, slot.d_direction, w, h, slot.d_f1, slot.d_stats);
  CUDA_CHECK_LAUNCH();

  DoubleThresholdKernel<<<grid_1d, kThreads1D, 0, s>>>(
      slot.d_f1, n, impl_->params.low_ratio, impl_->params.high_ratio,
      slot.d_stats, slot.d_labels);
  CUDA_CHECK_LAUNCH();

  // A modest, persistent-style grid keeps the cost of the self-terminated
  // (no-op) sweeps after convergence to a few microseconds each.
  const int num_tiles = tiles_x * tiles_y;
  const int hyst_grid = std::max(1, std::min(num_tiles, impl_->sm_count * 8));
  for (int k = 0; k < iters; ++k) {
    HysteresisKernel<<<hyst_grid, tile_block, 0, s>>>(
        slot.d_labels, w, h, tiles_x, num_tiles, k, slot.d_iter_flags);
  }
  CUDA_CHECK_LAUNCH();

  FinalizeKernel<<<grid_1d, kThreads1D, 0, s>>>(
      slot.d_labels, slot.d_f0, n,
      impl_->keep_gradient ? slot.d_gradient : nullptr, slot.d_stats);
  CUDA_CHECK_LAUNCH();

  CUDA_CHECK(cudaMemcpyAsync(slot.h_edges, slot.d_labels, n,
                             cudaMemcpyDeviceToHost, s));
  if (impl_->keep_gradient) {
    CUDA_CHECK(cudaMemcpyAsync(slot.h_gradient, slot.d_gradient, n,
                               cudaMemcpyDeviceToHost, s));
  }
  CUDA_CHECK(cudaMemcpyAsync(slot.h_stats, slot.d_stats, sizeof(DeviceStats),
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(slot.h_iter_flags, slot.d_iter_flags,
                             iters * sizeof(int), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaEventRecord(slot.stop, s));

  slot.busy = true;
  slot.width = w;
  slot.height = h;
}

bool GpuCannyPipeline::WaitSlot(int slot_index, GpuCannyResult* result) {
  Impl::Slot& slot = impl_->slots[slot_index];
  if (!slot.busy) return false;
  CUDA_CHECK(cudaEventSynchronize(slot.stop));
  slot.busy = false;

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, slot.start, slot.stop));
  const int iters = impl_->params.max_hysteresis_iters;
  int leading = 0;
  while (leading < iters && slot.h_iter_flags[leading] != 0) ++leading;

  uint32_t max_bits = slot.h_stats->max_magnitude_bits;
  float max_magnitude;
  std::memcpy(&max_magnitude, &max_bits, sizeof(max_magnitude));

  result->width = slot.width;
  result->height = slot.height;
  result->edges = slot.h_edges;
  result->gradient = impl_->keep_gradient ? slot.h_gradient : nullptr;
  result->max_magnitude = max_magnitude;
  result->edge_count = slot.h_stats->edge_count;
  result->hysteresis_iterations = std::min(iters, leading + 1);
  result->hysteresis_converged = leading < iters;
  result->gpu_ms = ms;
  return true;
}

}  // namespace canny
