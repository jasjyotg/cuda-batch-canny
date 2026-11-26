// Copyright 2026 The cuda-batch-canny Authors.
//
// Tunable parameters of the Canny edge detector, shared by the CPU reference
// and the GPU pipeline so both are configured identically.

#ifndef CUDA_BATCH_CANNY_SRC_CANNY_PARAMS_H_
#define CUDA_BATCH_CANNY_SRC_CANNY_PARAMS_H_

#include <string>
#include <vector>

namespace canny {

// Largest Gaussian radius supported; bounds the __constant__ weight table on
// the device. radius = ceil(3 * sigma), so sigma may be at most 5.
constexpr int kMaxGaussianRadius = 15;
constexpr float kMaxSigma = 5.0f;

struct CannyParams {
  // Standard deviation of the Gaussian smoothing filter, in pixels.
  float sigma = 1.4f;
  // Hysteresis thresholds expressed as fractions of the image's maximum
  // non-maximum-suppressed gradient magnitude (adapts to image contrast).
  float low_ratio = 0.05f;
  float high_ratio = 0.12f;
  // Upper bound on global hysteresis sweeps on the GPU. Each sweep already
  // propagates edges through a whole 16x16 tile in shared memory, so real
  // images typically converge in well under 20 sweeps.
  int max_hysteresis_iters = 100;
};

// Returns an empty string when `params` are valid, else a description of the
// first problem found.
std::string ValidateCannyParams(const CannyParams& params);

// Radius (half-width) of the Gaussian kernel used for `sigma`.
int GaussianRadius(float sigma);

// Normalized 1-D Gaussian weights of length 2 * GaussianRadius(sigma) + 1.
// Computed once on the host and uploaded to the GPU, so the CPU reference
// and the kernels use bit-identical weights.
std::vector<float> BuildGaussianKernel(float sigma);

}  // namespace canny

#endif  // CUDA_BATCH_CANNY_SRC_CANNY_PARAMS_H_
