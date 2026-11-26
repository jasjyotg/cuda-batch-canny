// Copyright 2026 The cuda-batch-canny Authors.

#ifndef CUDA_BATCH_CANNY_SRC_TIMER_H_
#define CUDA_BATCH_CANNY_SRC_TIMER_H_

#include <chrono>

namespace canny {

// Wall-clock stopwatch based on a monotonic clock.
class Stopwatch {
 public:
  Stopwatch() : start_(Clock::now()) {}

  void Reset() { start_ = Clock::now(); }

  double ElapsedMs() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_)
        .count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

}  // namespace canny

#endif  // CUDA_BATCH_CANNY_SRC_TIMER_H_
