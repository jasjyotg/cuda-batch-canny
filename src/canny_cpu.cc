// Copyright 2026 The cuda-batch-canny Authors.

#include "canny_cpu.h"

#include <algorithm>
#include <cstddef>

#include "canny_ops.h"

namespace canny {
namespace {

void ToGray(const Image& image, std::vector<float>* gray) {
  const size_t n = image.NumPixels();
  gray->resize(n);
  const uint8_t* src = image.pixels.data();
  for (size_t i = 0; i < n; ++i) {
    (*gray)[i] = image.channels == 3
                     ? GrayFromRgb(src[3 * i], src[3 * i + 1], src[3 * i + 2])
                     : static_cast<float>(src[i]);
  }
}

// Separable Gaussian blur with replicated borders. The accumulation order
// matches the GPU kernels exactly.
void GaussianBlur(const std::vector<float>& src, int width, int height,
                  const std::vector<float>& weights, std::vector<float>* dst) {
  const int radius = static_cast<int>(weights.size() / 2);
  std::vector<float> tmp(src.size());
  for (int y = 0; y < height; ++y) {
    const float* row = &src[static_cast<size_t>(y) * width];
    for (int x = 0; x < width; ++x) {
      float acc = 0.0f;
      for (int k = 0; k <= 2 * radius; ++k) {
        acc += weights[k] * row[ClampInt(x - radius + k, 0, width - 1)];
      }
      tmp[static_cast<size_t>(y) * width + x] = acc;
    }
  }
  dst->resize(src.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float acc = 0.0f;
      for (int k = 0; k <= 2 * radius; ++k) {
        const int yy = ClampInt(y - radius + k, 0, height - 1);
        acc += weights[k] * tmp[static_cast<size_t>(yy) * width + x];
      }
      (*dst)[static_cast<size_t>(y) * width + x] = acc;
    }
  }
}

void SobelAndDirection(const std::vector<float>& src, int width, int height,
                       std::vector<float>* magnitude,
                       std::vector<uint8_t>* direction) {
  magnitude->resize(src.size());
  direction->resize(src.size());
  auto at = [&](int x, int y) {
    return src[static_cast<size_t>(ClampInt(y, 0, height - 1)) * width +
               ClampInt(x, 0, width - 1)];
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float gx, gy;
      Sobel3x3(at(x - 1, y - 1), at(x, y - 1), at(x + 1, y - 1), at(x - 1, y),
               at(x + 1, y), at(x - 1, y + 1), at(x, y + 1), at(x + 1, y + 1),
               &gx, &gy);
      const size_t i = static_cast<size_t>(y) * width + x;
      (*magnitude)[i] = GradientMagnitude(gx, gy);
      (*direction)[i] = QuantizeDirection(gx, gy);
    }
  }
}

float NonMaxSuppression(const std::vector<float>& magnitude,
                        const std::vector<uint8_t>& direction, int width,
                        int height, std::vector<float>* nms) {
  nms->assign(magnitude.size(), 0.0f);
  float max_value = 0.0f;
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      const size_t i = static_cast<size_t>(y) * width + x;
      int dx, dy;
      DirectionOffset(direction[i], &dx, &dy);
      const float before =
          magnitude[i + static_cast<ptrdiff_t>(dy) * width + dx];
      const float after =
          magnitude[i - static_cast<ptrdiff_t>(dy) * width - dx];
      const float v = SuppressNonMax(magnitude[i], before, after);
      (*nms)[i] = v;
      max_value = std::max(max_value, v);
    }
  }
  return max_value;
}

// Classic stack-based hysteresis: flood-fill from strong pixels through
// 8-connected weak pixels. Produces the same fixed point as the GPU's
// iterative sweeps.
uint64_t Hysteresis(std::vector<uint8_t>* labels, int width, int height,
                    std::vector<uint8_t>* edges) {
  std::vector<size_t> stack;
  for (size_t i = 0; i < labels->size(); ++i) {
    if ((*labels)[i] == kLabelStrong) stack.push_back(i);
  }
  while (!stack.empty()) {
    const size_t i = stack.back();
    stack.pop_back();
    const int x = static_cast<int>(i % width);
    const int y = static_cast<int>(i / width);
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx;
        const int ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const size_t j = static_cast<size_t>(ny) * width + nx;
        if ((*labels)[j] == kLabelWeak) {
          (*labels)[j] = kLabelStrong;
          stack.push_back(j);
        }
      }
    }
  }
  edges->resize(labels->size());
  uint64_t count = 0;
  for (size_t i = 0; i < labels->size(); ++i) {
    const bool is_edge = (*labels)[i] == kLabelStrong;
    (*edges)[i] = is_edge ? kEdgeValue : 0;
    count += is_edge ? 1 : 0;
  }
  return count;
}

}  // namespace

void RunCannyCpu(const Image& image, const CannyParams& params,
                 CpuCannyResult* result) {
  const int w = image.width;
  const int h = image.height;
  std::vector<float> gray, blurred, magnitude, nms;
  std::vector<uint8_t> direction;

  ToGray(image, &gray);
  GaussianBlur(gray, w, h, BuildGaussianKernel(params.sigma), &blurred);
  SobelAndDirection(blurred, w, h, &magnitude, &direction);
  const float max_value = NonMaxSuppression(magnitude, direction, w, h, &nms);

  const float low = params.low_ratio * max_value;
  const float high = params.high_ratio * max_value;
  std::vector<uint8_t> labels(nms.size());
  for (size_t i = 0; i < nms.size(); ++i) {
    labels[i] = ClassifyPixel(nms[i], low, high);
  }
  result->max_magnitude = max_value;
  result->edge_count = Hysteresis(&labels, w, h, &result->edges);
}

}  // namespace canny
