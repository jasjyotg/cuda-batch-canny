// Copyright 2026 The cuda-batch-canny Authors.

#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>

namespace canny {
namespace {

// Reads the next whitespace-delimited header token, skipping '#' comments as
// allowed by the Netpbm specification.
bool ReadHeaderToken(std::istream& in, std::string* token) {
  token->clear();
  int c = in.get();
  while (c != EOF) {
    if (c == '#') {
      while (c != EOF && c != '\n') c = in.get();
    } else if (std::isspace(c)) {
      c = in.get();
    } else {
      break;
    }
  }
  while (c != EOF && !std::isspace(c) && c != '#') {
    token->push_back(static_cast<char>(c));
    c = in.get();
  }
  // A comment may start immediately after a token; leave it for the next call.
  // Otherwise exactly one whitespace byte (just consumed) separates maxval
  // from the raster, so the stream is now positioned at the first pixel.
  if (c == '#') in.unget();
  return !token->empty();
}

bool ParsePositiveInt(const std::string& token, int* value) {
  if (token.empty() || token.size() > 9) return false;
  int result = 0;
  for (char ch : token) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
    result = result * 10 + (ch - '0');
  }
  if (result <= 0) return false;
  *value = result;
  return true;
}

bool WriteRaw(const std::string& path, const char* magic, const uint8_t* data,
              int width, int height, int channels, std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "cannot open for writing: " + path;
    return false;
  }
  out << magic << "\n" << width << " " << height << "\n255\n";
  const size_t bytes = static_cast<size_t>(width) * height * channels;
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(bytes));
  if (!out) {
    *error = "write failed: " + path;
    return false;
  }
  return true;
}

bool ParseNetpbmHeader(std::istream& in, const std::string& path,
                       NetpbmHeader* header, std::string* error) {
  std::string magic, w_tok, h_tok, max_tok;
  if (!ReadHeaderToken(in, &magic) || !ReadHeaderToken(in, &w_tok) ||
      !ReadHeaderToken(in, &h_tok) || !ReadHeaderToken(in, &max_tok)) {
    *error = "truncated Netpbm header: " + path;
    return false;
  }
  if (magic == "P5") {
    header->channels = 1;
  } else if (magic == "P6") {
    header->channels = 3;
  } else {
    *error = "unsupported Netpbm type '" + magic + "' (need P5/P6): " + path;
    return false;
  }
  int maxval = 0;
  if (!ParsePositiveInt(w_tok, &header->width) ||
      !ParsePositiveInt(h_tok, &header->height) ||
      !ParsePositiveInt(max_tok, &maxval)) {
    *error = "invalid Netpbm dimensions: " + path;
    return false;
  }
  if (maxval != 255) {
    *error = "only 8-bit Netpbm (maxval 255) is supported: " + path;
    return false;
  }
  constexpr int kMaxDim = 1 << 15;
  if (header->width > kMaxDim || header->height > kMaxDim) {
    *error = "image too large: " + path;
    return false;
  }
  return true;
}

}  // namespace

bool ReadNetpbmHeader(const std::string& path, NetpbmHeader* header,
                      std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open: " + path;
    return false;
  }
  return ParseNetpbmHeader(in, path, header, error);
}

bool ReadNetpbm(const std::string& path, Image* image, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open: " + path;
    return false;
  }
  NetpbmHeader header;
  if (!ParseNetpbmHeader(in, path, &header, error)) return false;
  image->width = header.width;
  image->height = header.height;
  image->channels = header.channels;
  image->pixels.resize(image->SizeBytes());
  in.read(reinterpret_cast<char*>(image->pixels.data()),
          static_cast<std::streamsize>(image->pixels.size()));
  if (static_cast<size_t>(in.gcount()) != image->pixels.size()) {
    *error = "truncated raster data: " + path;
    return false;
  }
  return true;
}

bool WriteNetpbm(const std::string& path, const Image& image,
                 std::string* error) {
  if (image.channels != 1 && image.channels != 3) {
    *error = "WriteNetpbm supports 1 or 3 channels";
    return false;
  }
  return WriteRaw(path, image.channels == 1 ? "P5" : "P6", image.pixels.data(),
                  image.width, image.height, image.channels, error);
}

bool WritePgm(const std::string& path, const uint8_t* data, int width,
              int height, std::string* error) {
  return WriteRaw(path, "P5", data, width, height, 1, error);
}

bool HasNetpbmExtension(const std::string& path) {
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) return false;
  std::string ext = path.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext == "pgm" || ext == "ppm" || ext == "pnm";
}

}  // namespace canny
