// Copyright 2026 The cuda-batch-canny Authors.
//
// Batched, multi-stream GPU Canny edge detector.
//
// The pipeline owns N "slots". Each slot has its own CUDA stream, pinned
// (page-locked) host staging buffers and device work buffers. The caller
// round-robins images over slots:
//
//   for each image i:
//     slot = i % N
//     WaitSlot(slot, &result)  // blocks only until *that* slot is done
//     ...consume result (write to disk)...
//     Submit(slot, image_i)    // returns immediately; work is queued
//
// so while the host reads/writes files for one slot, the GPU copies and
// computes on the others. This header exposes no CUDA types, so it can be
// included from plain C++ translation units compiled by the host compiler.

#ifndef CUDA_BATCH_CANNY_SRC_CANNY_GPU_H_
#define CUDA_BATCH_CANNY_SRC_CANNY_GPU_H_

#include <cstdint>
#include <memory>
#include <string>

#include "canny_params.h"
#include "image_io.h"

namespace canny {

// Selects `device` and returns a one-paragraph description of it (name,
// compute capability, SM count, memory). Returns false if no CUDA device is
// usable, with the reason in `error`.
bool InitCudaDevice(int device, std::string* description, std::string* error);

// Output for one image. Pointers reference pinned memory owned by the
// pipeline and stay valid until the same slot is submitted to again.
struct GpuCannyResult {
  int width = 0;
  int height = 0;
  const uint8_t* edges = nullptr;     // 0/255 per pixel.
  const uint8_t* gradient = nullptr;  // Only when keep_gradient was set.
  float max_magnitude = 0.0f;
  uint64_t edge_count = 0;
  int hysteresis_iterations = 0;
  bool hysteresis_converged = false;
  float gpu_ms = 0.0f;  // H2D copy -> kernels -> D2H copy, from CUDA events.
};

class GpuCannyPipeline {
 public:
  // `keep_gradient` additionally returns an 8-bit gradient-magnitude image.
  GpuCannyPipeline(const CannyParams& params, int num_slots,
                   bool keep_gradient);
  ~GpuCannyPipeline();

  GpuCannyPipeline(const GpuCannyPipeline&) = delete;
  GpuCannyPipeline& operator=(const GpuCannyPipeline&) = delete;

  int num_slots() const;

  // Pre-allocates every slot for images up to the given size. Optional, but
  // calling it once up front avoids re-allocations (cudaFree/cudaFreeHost
  // synchronize the whole device) in the middle of a batch.
  void Reserve(int max_width, int max_height, int max_channels);

  // Blocks until the work most recently submitted to `slot` has finished.
  // Returns false if the slot was idle; otherwise fills `result` and
  // returns true. After this call the slot is idle.
  bool WaitSlot(int slot, GpuCannyResult* result);

  // Queues `image` on `slot` and returns without waiting for the GPU. The
  // slot must be idle (call WaitSlot first). The image is copied into pinned
  // memory, so `image` may be destroyed as soon as this returns.
  void Submit(int slot, const Image& image);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace canny

#endif  // CUDA_BATCH_CANNY_SRC_CANNY_GPU_H_
