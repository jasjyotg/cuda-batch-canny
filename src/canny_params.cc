// Copyright 2026 The cuda-batch-canny Authors.

#include "canny_params.h"

#include <algorithm>
#include <cmath>

namespace canny {

std::string ValidateCannyParams(const CannyParams& params) {
  if (!(params.sigma > 0.0f) || params.sigma > kMaxSigma) {
    return "--sigma must be in (0, 5]";
  }
  if (!(params.low_ratio > 0.0f) || !(params.high_ratio <= 1.0f) ||
      params.low_ratio > params.high_ratio) {
    return "thresholds must satisfy 0 < --low <= --high <= 1";
  }
  if (params.max_hysteresis_iters < 1 || params.max_hysteresis_iters > 10000) {
    return "--hysteresis_iters must be in [1, 10000]";
  }
  return "";
}

int GaussianRadius(float sigma) {
  const int radius = static_cast<int>(std::ceil(3.0f * sigma));
  return std::min(kMaxGaussianRadius, std::max(1, radius));
}

std::vector<float> BuildGaussianKernel(float sigma) {
  const int radius = GaussianRadius(sigma);
  std::vector<double> weights(2 * radius + 1);
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const double w = std::exp(-(i * i) / (2.0 * sigma * sigma));
    weights[i + radius] = w;
    sum += w;
  }
  std::vector<float> kernel(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    kernel[i] = static_cast<float>(weights[i] / sum);
  }
  return kernel;
}

}  // namespace canny
