// Copyright 2026 The cuda-batch-canny Authors.
//
// Per-pixel math shared verbatim by the CPU reference (compiled by the host
// compiler) and the CUDA kernels (compiled by nvcc). Sharing one definition,
// together with nvcc's --fmad=false, makes the GPU output bit-identical to
// the CPU output, which is what --verify checks.

#ifndef CUDA_BATCH_CANNY_SRC_CANNY_OPS_H_
#define CUDA_BATCH_CANNY_SRC_CANNY_OPS_H_

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define CANNY_HOST_DEVICE __host__ __device__
#else
#define CANNY_HOST_DEVICE
#endif

namespace canny {

// Labels produced by double thresholding and consumed by hysteresis.
constexpr uint8_t kLabelNone = 0;
constexpr uint8_t kLabelWeak = 1;
constexpr uint8_t kLabelStrong = 2;
constexpr uint8_t kEdgeValue = 255;

// Quantized gradient directions (image y axis points down).
constexpr uint8_t kDirHorizontal = 0;  // Gradient along x: compare W/E.
constexpr uint8_t kDirDiagDown = 1;    // Gradient along +x+y: NW/SE.
constexpr uint8_t kDirVertical = 2;    // Gradient along y: compare N/S.
constexpr uint8_t kDirDiagUp = 3;      // Gradient along +x-y: NE/SW.

CANNY_HOST_DEVICE inline int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ITU-R BT.601 luma. Evaluated left to right in single precision.
CANNY_HOST_DEVICE inline float GrayFromRgb(uint8_t r, uint8_t g, uint8_t b) {
  return 0.299f * static_cast<float>(r) + 0.587f * static_cast<float>(g) +
         0.114f * static_cast<float>(b);
}

// 3x3 Sobel operator on the neighborhood n[row][col] centered on n[1][1].
CANNY_HOST_DEVICE inline void Sobel3x3(float n00, float n01, float n02,
                                       float n10, float n12, float n20,
                                       float n21, float n22, float* gx,
                                       float* gy) {
  *gx = (n02 + 2.0f * n12 + n22) - (n00 + 2.0f * n10 + n20);
  *gy = (n20 + 2.0f * n21 + n22) - (n00 + 2.0f * n01 + n02);
}

CANNY_HOST_DEVICE inline float GradientMagnitude(float gx, float gy) {
  return sqrtf(gx * gx + gy * gy);
}

// Buckets the gradient angle into 4 directions using only multiplications
// and comparisons (no atan2), so host and device agree exactly.
CANNY_HOST_DEVICE inline uint8_t QuantizeDirection(float gx, float gy) {
  constexpr float kTan22_5 = 0.41421356f;
  constexpr float kTan67_5 = 2.41421356f;
  const float ax = fabsf(gx);
  const float ay = fabsf(gy);
  if (ay <= ax * kTan22_5) return kDirHorizontal;
  if (ay >= ax * kTan67_5) return kDirVertical;
  return (gx > 0.0f) == (gy > 0.0f) ? kDirDiagDown : kDirDiagUp;
}

// Pixel offsets (dx, dy) of the "before" neighbor along `dir`; the "after"
// neighbor is the negation.
CANNY_HOST_DEVICE inline void DirectionOffset(uint8_t dir, int* dx, int* dy) {
  switch (dir) {
    case kDirHorizontal:
      *dx = -1;
      *dy = 0;
      break;
    case kDirDiagDown:
      *dx = -1;
      *dy = -1;
      break;
    case kDirVertical:
      *dx = 0;
      *dy = -1;
      break;
    default:  // kDirDiagUp
      *dx = 1;
      *dy = -1;
      break;
  }
}

// Non-maximum suppression: a pixel survives only if it is a local maximum
// across the edge. The asymmetric (> / >=) test keeps exactly one pixel of a
// two-pixel plateau instead of zero or two.
CANNY_HOST_DEVICE inline float SuppressNonMax(float m, float before,
                                              float after) {
  return (m > before && m >= after) ? m : 0.0f;
}

CANNY_HOST_DEVICE inline uint8_t ClassifyPixel(float m, float low_threshold,
                                               float high_threshold) {
  if (!(m > 0.0f)) return kLabelNone;
  if (m >= high_threshold) return kLabelStrong;
  if (m >= low_threshold) return kLabelWeak;
  return kLabelNone;
}

// Maps a gradient magnitude to 0..255 for visualization only.
CANNY_HOST_DEVICE inline uint8_t GradientToByte(float m, float max_m) {
  if (!(max_m > 0.0f)) return 0;
  const float v = m * (255.0f / max_m) + 0.5f;
  return static_cast<uint8_t>(v > 255.0f ? 255.0f : v);
}

}  // namespace canny

#endif  // CUDA_BATCH_CANNY_SRC_CANNY_OPS_H_
