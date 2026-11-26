// Copyright 2026 The cuda-batch-canny Authors.
//
// canny_batch: runs GPU Canny edge detection over every PGM/PPM image in a
// directory tree in a single execution, writes one edge map per input plus
// CSV/summary logs, and optionally verifies every result against a CPU
// reference implementation.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "canny_cpu.h"
#include "canny_gpu.h"
#include "canny_params.h"
#include "flag_parser.h"
#include "image_io.h"
#include "timer.h"

namespace canny {
namespace {

namespace fs = std::filesystem;

struct Options {
  std::string input_dir;
  std::string output_dir = "output";
  CannyParams params;
  int streams = 4;
  int device = 0;
  int max_images = 0;
  bool save_gradient = false;
  bool verify = false;
  float verify_tolerance = 0.0001f;
  bool quiet = false;
};

// Everything we learn about one input image over the run.
struct ImageRecord {
  std::string input_path;
  std::string relative_name;  // Path relative to --input_dir.
  std::string edges_path;
  bool ok = false;
  int width = 0;
  int height = 0;
  int channels = 0;
  float gpu_ms = 0.0f;
  float max_magnitude = 0.0f;
  uint64_t edge_count = 0;
  int hysteresis_iterations = 0;
  bool hysteresis_converged = false;
  // Filled by --verify.
  bool verified = false;
  double cpu_ms = 0.0;
  uint64_t mismatched_pixels = 0;
};

struct BatchTimings {
  double wall_ms = 0.0;
  double read_ms = 0.0;   // Host thread reading inputs from disk.
  double write_ms = 0.0;  // Host thread writing edge maps to disk.
  double wait_ms = 0.0;   // Host thread blocked waiting for the GPU.
};

bool ParseOptions(int argc, char** argv, Options* options) {
  FlagParser parser(
      "canny_batch - batched multi-stream CUDA Canny edge detection over a\n"
      "directory of 8-bit PGM/PPM images.");
  parser.AddString("input_dir", &options->input_dir,
                   "Directory searched recursively for .pgm/.ppm/.pnm files "
                   "(required)");
  parser.AddString("output_dir", &options->output_dir,
                   "Where edge maps, results.csv and summary.txt are written");
  parser.AddFloat("sigma", &options->params.sigma,
                  "Gaussian smoothing sigma in pixels, (0, 5]");
  parser.AddFloat("low", &options->params.low_ratio,
                  "Low hysteresis threshold as a fraction of the image's "
                  "max gradient");
  parser.AddFloat("high", &options->params.high_ratio,
                  "High hysteresis threshold as a fraction of the image's "
                  "max gradient");
  parser.AddInt("hysteresis_iters", &options->params.max_hysteresis_iters,
                "Max GPU hysteresis sweeps per image (self-terminating)");
  parser.AddInt("streams", &options->streams,
                "Number of CUDA streams / images in flight concurrently");
  parser.AddInt("device", &options->device, "CUDA device ordinal");
  parser.AddInt("max_images", &options->max_images,
                "Process at most this many images (0 = all)");
  parser.AddBool("save_gradient", &options->save_gradient,
                 "Also write gradient-magnitude images to <output_dir>/"
                 "gradient");
  parser.AddBool("verify", &options->verify,
                 "Re-run every image on the CPU reference and compare "
                 "pixel-by-pixel");
  parser.AddFloat("verify_tolerance", &options->verify_tolerance,
                  "Max fraction of mismatched pixels allowed by --verify");
  parser.AddBool("quiet", &options->quiet, "Suppress the per-image log lines");

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return false;
  }
  if (parser.help_requested()) {
    std::printf("%s", parser.Usage(argv[0]).c_str());
    std::exit(0);
  }
  if (options->input_dir.empty()) {
    std::fprintf(stderr, "error: --input_dir is required (see --help)\n");
    return false;
  }
  error = ValidateCannyParams(options->params);
  if (!error.empty()) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return false;
  }
  if (options->streams < 1 || options->streams > 32) {
    std::fprintf(stderr, "error: --streams must be in [1, 32]\n");
    return false;
  }
  return true;
}

// Returns the sorted list of Netpbm files under `root`.
std::vector<ImageRecord> FindImages(const Options& options) {
  std::vector<ImageRecord> records;
  std::error_code ec;
  const fs::path root(options.input_dir);
  if (!fs::is_directory(root, ec)) return records;
  for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (!it->is_regular_file(ec)) continue;
    const std::string path = it->path().string();
    if (!HasNetpbmExtension(path)) continue;
    ImageRecord record;
    record.input_path = path;
    record.relative_name = fs::relative(it->path(), root, ec).string();
    fs::path out =
        fs::path(options.output_dir) / "edges" / record.relative_name;
    out.replace_extension(".pgm");
    record.edges_path = out.string();
    records.push_back(record);
  }
  std::sort(records.begin(), records.end(),
            [](const ImageRecord& a, const ImageRecord& b) {
              return a.input_path < b.input_path;
            });
  if (options.max_images > 0 &&
      records.size() > static_cast<size_t>(options.max_images)) {
    records.resize(options.max_images);
  }
  return records;
}

bool WriteOutputPgm(const std::string& path, const uint8_t* data, int width,
                    int height) {
  std::error_code ec;
  fs::create_directories(fs::path(path).parent_path(), ec);
  std::string error;
  if (!WritePgm(path, data, width, height, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return false;
  }
  return true;
}

// Phase 1: stream every image through the GPU pipeline.
BatchTimings RunGpuBatch(const Options& options,
                         std::vector<ImageRecord>* records) {
  GpuCannyPipeline pipeline(options.params, options.streams,
                            options.save_gradient);

  // Size every slot for the largest image up front: growing buffers later
  // would call cudaFree, which synchronizes the whole device.
  int max_w = 1, max_h = 1, max_c = 1;
  for (const ImageRecord& r : *records) {
    NetpbmHeader header;
    std::string error;
    if (ReadNetpbmHeader(r.input_path, &header, &error)) {
      max_w = std::max(max_w, header.width);
      max_h = std::max(max_h, header.height);
      max_c = std::max(max_c, header.channels);
    }
  }
  std::printf("Largest image %dx%dx%d; allocating %d stream slot(s)\n", max_w,
              max_h, max_c, pipeline.num_slots());
  pipeline.Reserve(max_w, max_h, max_c);

  BatchTimings timings;
  const int num_slots = pipeline.num_slots();
  std::vector<int> slot_job(num_slots, -1);
  size_t finished = 0;
  const size_t total = records->size();

  auto harvest = [&](int slot) {
    GpuCannyResult result;
    Stopwatch wait_timer;
    if (!pipeline.WaitSlot(slot, &result)) return;
    timings.wait_ms += wait_timer.ElapsedMs();
    ImageRecord& r = (*records)[slot_job[slot]];
    Stopwatch write_timer;
    r.ok =
        WriteOutputPgm(r.edges_path, result.edges, result.width, result.height);
    if (r.ok && result.gradient != nullptr) {
      fs::path grad =
          fs::path(options.output_dir) / "gradient" / r.relative_name;
      grad.replace_extension(".pgm");
      WriteOutputPgm(grad.string(), result.gradient, result.width,
                     result.height);
    }
    timings.write_ms += write_timer.ElapsedMs();
    r.gpu_ms = result.gpu_ms;
    r.max_magnitude = result.max_magnitude;
    r.edge_count = result.edge_count;
    r.hysteresis_iterations = result.hysteresis_iterations;
    r.hysteresis_converged = result.hysteresis_converged;
    ++finished;
    if (!options.quiet) {
      const double density =
          100.0 * static_cast<double>(r.edge_count) /
          (static_cast<double>(r.width) * static_cast<double>(r.height));
      std::printf(
          "[%5zu/%zu] slot %d  %-40s %5dx%-5d c%d  gpu %8.3f ms  "
          "edges %6.2f%%  hysteresis %d sweep(s)%s\n",
          finished, total, slot, r.relative_name.c_str(), r.width, r.height,
          r.channels, r.gpu_ms, density, r.hysteresis_iterations,
          r.hysteresis_converged ? "" : " (NOT CONVERGED)");
    }
  };

  Stopwatch wall;
  for (size_t i = 0; i < total; ++i) {
    const int slot = static_cast<int>(i % num_slots);
    ImageRecord& r = (*records)[i];
    // Read the next input *before* waiting on the slot, so disk I/O overlaps
    // with the GPU work already queued on every slot.
    Image image;
    std::string error;
    Stopwatch read_timer;
    const bool read_ok = ReadNetpbm(r.input_path, &image, &error);
    timings.read_ms += read_timer.ElapsedMs();
    if (!read_ok) {
      std::fprintf(stderr, "skipping: %s\n", error.c_str());
      continue;
    }
    r.width = image.width;
    r.height = image.height;
    r.channels = image.channels;
    harvest(slot);
    pipeline.Submit(slot, image);
    slot_job[slot] = static_cast<int>(i);
  }
  for (int slot = 0; slot < num_slots; ++slot) harvest(slot);
  timings.wall_ms = wall.ElapsedMs();
  return timings;
}

// Phase 2: CPU reference on every image, compared against the written
// GPU output files.
void RunVerification(const Options& options,
                     std::vector<ImageRecord>* records) {
  std::printf("\nVerifying GPU output against the CPU reference...\n");
  for (ImageRecord& r : *records) {
    if (!r.ok) continue;
    Image input, gpu_edges;
    std::string error;
    if (!ReadNetpbm(r.input_path, &input, &error) ||
        !ReadNetpbm(r.edges_path, &gpu_edges, &error)) {
      std::fprintf(stderr, "verify: %s\n", error.c_str());
      continue;
    }
    if (gpu_edges.width != input.width || gpu_edges.height != input.height) {
      std::fprintf(stderr, "verify: size mismatch for %s\n",
                   r.edges_path.c_str());
      continue;
    }
    CpuCannyResult cpu;
    Stopwatch cpu_timer;
    RunCannyCpu(input, options.params, &cpu);
    r.cpu_ms = cpu_timer.ElapsedMs();
    uint64_t mismatched = 0;
    for (size_t i = 0; i < cpu.edges.size(); ++i) {
      mismatched += cpu.edges[i] != gpu_edges.pixels[i] ? 1 : 0;
    }
    r.mismatched_pixels = mismatched;
    r.verified = true;
    if (!options.quiet) {
      std::printf(
          "  %-40s cpu %9.2f ms  mismatched pixels %llu%s\n",
          r.relative_name.c_str(), r.cpu_ms,
          static_cast<unsigned long long>(mismatched),  // NOLINT
          cpu.edge_count == r.edge_count ? "" : "  (edge count differs)");
    }
  }
}

void WriteCsv(const std::string& path,
              const std::vector<ImageRecord>& records) {
  std::ofstream out(path);
  out << "file,width,height,channels,gpu_ms,edge_pixels,edge_density,"
         "max_gradient,hysteresis_sweeps,hysteresis_converged,cpu_ms,"
         "mismatched_pixels\n";
  for (const ImageRecord& r : records) {
    if (!r.ok) continue;
    const double pixels = static_cast<double>(r.width) * r.height;
    out << r.relative_name << "," << r.width << "," << r.height << ","
        << r.channels << "," << r.gpu_ms << "," << r.edge_count << ","
        << r.edge_count / pixels << "," << r.max_magnitude << ","
        << r.hysteresis_iterations << "," << (r.hysteresis_converged ? 1 : 0)
        << ",";
    if (r.verified) {
      out << r.cpu_ms << "," << r.mismatched_pixels;
    } else {
      out << ",";
    }
    out << "\n";
  }
}

// Builds the end-of-run report and returns the process exit code.
int Summarize(const Options& options, const std::string& device,
              const std::vector<ImageRecord>& records, const BatchTimings& t) {
  size_t ok = 0, not_converged = 0, verified = 0, exact = 0;
  double pixels = 0.0, gpu_ms_sum = 0.0, cpu_ms_sum = 0.0;
  double edges = 0.0, mismatched = 0.0, verified_pixels = 0.0;
  int max_sweeps = 0;
  for (const ImageRecord& r : records) {
    if (!r.ok) continue;
    ++ok;
    const double p = static_cast<double>(r.width) * r.height;
    pixels += p;
    gpu_ms_sum += r.gpu_ms;
    edges += static_cast<double>(r.edge_count);
    max_sweeps = std::max(max_sweeps, r.hysteresis_iterations);
    not_converged += r.hysteresis_converged ? 0 : 1;
    if (r.verified) {
      ++verified;
      verified_pixels += p;
      cpu_ms_sum += r.cpu_ms;
      mismatched += static_cast<double>(r.mismatched_pixels);
      exact += r.mismatched_pixels == 0 ? 1 : 0;
    }
  }
  std::ostringstream s;
  char line[256];
  s << "==================== canny_batch summary ====================\n";
  s << device << "\n";
  std::snprintf(line, sizeof(line),
                "params: sigma=%.2f (radius %d) low=%.3f high=%.3f "
                "streams=%d\n",
                options.params.sigma, GaussianRadius(options.params.sigma),
                options.params.low_ratio, options.params.high_ratio,
                options.streams);
  s << line;
  std::snprintf(line, sizeof(line),
                "images processed      : %zu of %zu (%.1f megapixels)\n", ok,
                records.size(), pixels / 1e6);
  s << line;
  std::snprintf(line, sizeof(line),
                "GPU batch wall time   : %.1f ms  (%.1f images/s, %.1f "
                "MPix/s, includes disk I/O)\n",
                t.wall_ms, ok / (t.wall_ms / 1e3),
                pixels / 1e6 / (t.wall_ms / 1e3));
  s << line;
  std::snprintf(line, sizeof(line),
                "  host read / write / wait-for-GPU: %.1f / %.1f / %.1f ms\n",
                t.read_ms, t.write_ms, t.wait_ms);
  s << line;
  std::snprintf(line, sizeof(line),
                "per-image GPU latency : mean %.3f ms (event-timed; overlaps "
                "across streams)\n",
                ok ? gpu_ms_sum / ok : 0.0);
  s << line;
  std::snprintf(line, sizeof(line),
                "edge density          : %.2f%% of pixels; hysteresis max "
                "%d sweep(s), %zu not converged\n",
                pixels > 0 ? 100.0 * edges / pixels : 0.0, max_sweeps,
                not_converged);
  s << line;
  int exit_code = ok == records.size() ? 0 : 1;
  if (options.verify) {
    const double fraction =
        verified_pixels > 0 ? mismatched / verified_pixels : 0.0;
    std::snprintf(line, sizeof(line),
                  "CPU reference         : %.1f ms total for %zu images -> "
                  "GPU batch is %.1fx faster end-to-end\n",
                  cpu_ms_sum, verified, cpu_ms_sum / t.wall_ms);
    s << line;
    std::snprintf(line, sizeof(line),
                  "verification          : %zu/%zu bit-exact, %.0f "
                  "mismatched pixels (%.5f%%) -> %s\n",
                  exact, verified, mismatched, 100.0 * fraction,
                  fraction <= options.verify_tolerance ? "PASS" : "FAIL");
    s << line;
    if (fraction > options.verify_tolerance || verified != ok) exit_code = 2;
  }
  s << "=============================================================\n";
  const std::string report = s.str();
  std::printf("\n%s", report.c_str());
  std::ofstream(fs::path(options.output_dir) / "summary.txt") << report;
  return exit_code;
}

int Run(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) return 1;

  std::string device, error;
  if (!InitCudaDevice(options.device, &device, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }
  std::printf("%s\n", device.c_str());

  std::vector<ImageRecord> records = FindImages(options);
  if (records.empty()) {
    std::fprintf(stderr, "error: no .pgm/.ppm images found under '%s'\n",
                 options.input_dir.c_str());
    return 1;
  }
  std::printf("Found %zu image(s) under %s\n", records.size(),
              options.input_dir.c_str());
  std::error_code ec;
  fs::create_directories(options.output_dir, ec);

  const BatchTimings timings = RunGpuBatch(options, &records);
  if (options.verify) RunVerification(options, &records);

  const std::string csv =
      (fs::path(options.output_dir) / "results.csv").string();
  WriteCsv(csv, records);
  std::printf("Wrote %s\n", csv.c_str());
  return Summarize(options, device, records, timings);
}

}  // namespace
}  // namespace canny

int main(int argc, char** argv) { return canny::Run(argc, argv); }
