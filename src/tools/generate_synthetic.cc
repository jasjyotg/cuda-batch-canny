// Copyright 2026 The cuda-batch-canny Authors.
//
// generate_synthetic: writes a reproducible dataset of procedurally drawn
// scenes (gradient background + random circles, rings, rotated rectangles
// and triangles + sensor-like noise) as PGM or PPM files. This guarantees the
// project can be demonstrated on hundreds of small or tens of large images
// even in an environment without internet access.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "flag_parser.h"
#include "image_io.h"

namespace canny {
namespace {

struct Color {
  float c[3];
};

// Signed-area test used for triangle rasterization.
float Edge(float ax, float ay, float bx, float by, float px, float py) {
  return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

class SceneDrawer {
 public:
  SceneDrawer(int width, int height, int channels, uint32_t seed)
      : width_(width),
        height_(height),
        channels_(channels),
        rng_(seed),
        canvas_(static_cast<size_t>(width) * height * channels) {}

  Image Draw(int num_shapes, float noise_sigma) {
    DrawBackground();
    for (int i = 0; i < num_shapes; ++i) {
      switch (std::uniform_int_distribution<int>(0, 3)(rng_)) {
        case 0:
          DrawDisk(false);
          break;
        case 1:
          DrawDisk(true);
          break;
        case 2:
          DrawRotatedRect();
          break;
        default:
          DrawTriangle();
          break;
      }
    }
    return Quantize(noise_sigma);
  }

 private:
  float Uniform(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng_);
  }

  Color RandomColor() {
    Color color;
    const float gray = Uniform(0.0f, 255.0f);
    for (int k = 0; k < 3; ++k) {
      color.c[k] = channels_ == 3 ? Uniform(0.0f, 255.0f) : gray;
    }
    return color;
  }

  void Put(int x, int y, const Color& color) {
    float* p = &canvas_[(static_cast<size_t>(y) * width_ + x) * channels_];
    for (int k = 0; k < channels_; ++k) p[k] = color.c[k];
  }

  void DrawBackground() {
    const Color a = RandomColor();
    const Color b = RandomColor();
    const float angle = Uniform(0.0f, 6.2831853f);
    const float dx = std::cos(angle), dy = std::sin(angle);
    const float span = std::abs(dx) * width_ + std::abs(dy) * height_;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        float t =
            (dx * (x - width_ / 2.0f) + dy * (y - height_ / 2.0f)) / span +
            0.5f;
        t = std::min(1.0f, std::max(0.0f, t));
        Color c;
        for (int k = 0; k < 3; ++k) c.c[k] = a.c[k] + t * (b.c[k] - a.c[k]);
        Put(x, y, c);
      }
    }
  }

  float RandomSize() {
    const float scale = static_cast<float>(std::min(width_, height_));
    return Uniform(0.03f, 0.18f) * scale;
  }

  void DrawDisk(bool ring) {
    const float cx = Uniform(0.0f, width_), cy = Uniform(0.0f, height_);
    const float r = RandomSize();
    const float inner = ring ? r * Uniform(0.4f, 0.8f) : -1.0f;
    const Color color = RandomColor();
    const int x0 = std::max(0, static_cast<int>(cx - r));
    const int x1 = std::min(width_ - 1, static_cast<int>(cx + r));
    const int y0 = std::max(0, static_cast<int>(cy - r));
    const int y1 = std::min(height_ - 1, static_cast<int>(cy + r));
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const float d = std::hypot(x - cx, y - cy);
        if (d <= r && d >= inner) Put(x, y, color);
      }
    }
  }

  void DrawRotatedRect() {
    const float cx = Uniform(0.0f, width_), cy = Uniform(0.0f, height_);
    const float hw = RandomSize(), hh = RandomSize() * 0.6f;
    const float angle = Uniform(0.0f, 3.1415926f);
    const float ca = std::cos(angle), sa = std::sin(angle);
    const float extent = std::hypot(hw, hh);
    const Color color = RandomColor();
    const int x0 = std::max(0, static_cast<int>(cx - extent));
    const int x1 = std::min(width_ - 1, static_cast<int>(cx + extent));
    const int y0 = std::max(0, static_cast<int>(cy - extent));
    const int y1 = std::min(height_ - 1, static_cast<int>(cy + extent));
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const float u = (x - cx) * ca + (y - cy) * sa;
        const float v = -(x - cx) * sa + (y - cy) * ca;
        if (std::abs(u) <= hw && std::abs(v) <= hh) Put(x, y, color);
      }
    }
  }

  void DrawTriangle() {
    const float cx = Uniform(0.0f, width_), cy = Uniform(0.0f, height_);
    const float r = RandomSize() * 1.3f;
    float px[3], py[3];
    for (int i = 0; i < 3; ++i) {
      const float a = Uniform(0.0f, 6.2831853f);
      px[i] = cx + r * std::cos(a);
      py[i] = cy + r * std::sin(a);
    }
    const Color color = RandomColor();
    const int x0 = std::max(0, static_cast<int>(cx - r));
    const int x1 = std::min(width_ - 1, static_cast<int>(cx + r));
    const int y0 = std::max(0, static_cast<int>(cy - r));
    const int y1 = std::min(height_ - 1, static_cast<int>(cy + r));
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const float e0 = Edge(px[0], py[0], px[1], py[1], x, y);
        const float e1 = Edge(px[1], py[1], px[2], py[2], x, y);
        const float e2 = Edge(px[2], py[2], px[0], py[0], x, y);
        const bool inside =
            (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0);
        if (inside) Put(x, y, color);
      }
    }
  }

  Image Quantize(float noise_sigma) {
    std::normal_distribution<float> noise(0.0f, noise_sigma);
    Image image;
    image.width = width_;
    image.height = height_;
    image.channels = channels_;
    image.pixels.resize(canvas_.size());
    for (size_t i = 0; i < canvas_.size(); ++i) {
      const float v = canvas_[i] + (noise_sigma > 0 ? noise(rng_) : 0.0f);
      image.pixels[i] = static_cast<uint8_t>(
          std::lround(std::min(255.0f, std::max(0.0f, v))));
    }
    return image;
  }

  int width_;
  int height_;
  int channels_;
  std::mt19937 rng_;
  std::vector<float> canvas_;
};

int Run(int argc, char** argv) {
  std::string output_dir = "data/synthetic";
  std::string prefix = "scene";
  int count = 200;
  int width = 256;
  int height = 256;
  int shapes = 12;
  int seed = 2026;
  float noise = 6.0f;
  bool color = false;

  FlagParser parser(
      "generate_synthetic - write a reproducible dataset of synthetic scenes "
      "as PGM/PPM files.");
  parser.AddString("output_dir", &output_dir, "Destination directory");
  parser.AddString("prefix", &prefix, "File name prefix");
  parser.AddInt("count", &count, "Number of images to write");
  parser.AddInt("width", &width, "Image width in pixels");
  parser.AddInt("height", &height, "Image height in pixels");
  parser.AddInt("shapes", &shapes, "Random shapes per image");
  parser.AddInt("seed", &seed, "Base random seed (image i uses seed + i)");
  parser.AddFloat("noise", &noise, "Gaussian noise sigma in gray levels");
  parser.AddBool("color", &color, "Write RGB .ppm instead of gray .pgm");
  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }
  if (parser.help_requested()) {
    std::printf("%s", parser.Usage(argv[0]).c_str());
    return 0;
  }
  if (count < 1 || width < 8 || height < 8 || width > 16384 || height > 16384) {
    std::fprintf(stderr, "error: need count >= 1 and 8 <= size <= 16384\n");
    return 1;
  }
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  const char* ext = color ? "ppm" : "pgm";
  for (int i = 0; i < count; ++i) {
    SceneDrawer drawer(width, height, color ? 3 : 1,
                       static_cast<uint32_t>(seed + i));
    const Image image = drawer.Draw(shapes, noise);
    char name[64];
    std::snprintf(name, sizeof(name), "%s_%04d.%s", prefix.c_str(), i, ext);
    const std::string path = output_dir + "/" + name;
    if (!WriteNetpbm(path, image, &error)) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      return 1;
    }
  }
  std::printf("Wrote %d %dx%d %s image(s) to %s\n", count, width, height,
              color ? "RGB" : "gray", output_dir.c_str());
  return 0;
}

}  // namespace
}  // namespace canny

int main(int argc, char** argv) { return canny::Run(argc, argv); }
