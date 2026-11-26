// Copyright 2026 The cuda-batch-canny Authors.
//
// Minimal, dependency-free reader/writer for binary Netpbm images
// (P5 = 8-bit grayscale PGM, P6 = 8-bit RGB PPM). Keeping image I/O free of
// third-party libraries means the project builds anywhere the CUDA toolkit
// is installed; scripts/fetch_sipi.sh converts TIFF/PNG sources to Netpbm.

#ifndef CUDA_BATCH_CANNY_SRC_IMAGE_IO_H_
#define CUDA_BATCH_CANNY_SRC_IMAGE_IO_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace canny {

// An interleaved 8-bit image stored row-major with no row padding.
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;  // 1 (grayscale) or 3 (RGB).
  std::vector<uint8_t> pixels;

  size_t NumPixels() const {
    return static_cast<size_t>(width) * static_cast<size_t>(height);
  }
  size_t SizeBytes() const { return NumPixels() * channels; }
};

// Dimensions parsed from a Netpbm header without reading the raster.
struct NetpbmHeader {
  int width = 0;
  int height = 0;
  int channels = 0;
};

// Parses only the header of a P5/P6 file. Used to pre-size GPU buffers for
// the largest image in a batch before any work is queued.
bool ReadNetpbmHeader(const std::string& path, NetpbmHeader* header,
                      std::string* error);

// Reads a binary PGM (P5) or PPM (P6) file with maxval 255. On failure
// returns false and writes a human-readable reason to `error`.
bool ReadNetpbm(const std::string& path, Image* image, std::string* error);

// Writes `image` as P5 (1 channel) or P6 (3 channels). Returns false and sets
// `error` on failure.
bool WriteNetpbm(const std::string& path, const Image& image,
                 std::string* error);

// Same as WriteNetpbm but for a raw 8-bit single-channel buffer, which lets
// callers write directly from pinned host memory without an extra copy.
bool WritePgm(const std::string& path, const uint8_t* data, int width,
              int height, std::string* error);

// Returns true when `path` ends in .pgm, .ppm or .pnm (case-insensitive).
bool HasNetpbmExtension(const std::string& path);

}  // namespace canny

#endif  // CUDA_BATCH_CANNY_SRC_IMAGE_IO_H_
