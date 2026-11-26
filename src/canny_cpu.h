// Copyright 2026 The cuda-batch-canny Authors.
//
// Single-threaded CPU reference implementation of the Canny pipeline. It is
// used only by --verify to check the GPU output and to measure speedup; the
// production path is always the GPU (canny_gpu.h).

#ifndef CUDA_BATCH_CANNY_SRC_CANNY_CPU_H_
#define CUDA_BATCH_CANNY_SRC_CANNY_CPU_H_

#include <cstdint>
#include <vector>

#include "canny_params.h"
#include "image_io.h"

namespace canny {

struct CpuCannyResult {
  std::vector<uint8_t> edges;  // 0 or 255 per pixel.
  float max_magnitude = 0.0f;  // Max gradient magnitude after NMS.
  uint64_t edge_count = 0;
};

void RunCannyCpu(const Image& image, const CannyParams& params,
                 CpuCannyResult* result);

}  // namespace canny

#endif  // CUDA_BATCH_CANNY_SRC_CANNY_CPU_H_
