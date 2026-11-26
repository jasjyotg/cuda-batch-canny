// Copyright 2026 The cuda-batch-canny Authors.
//
// Error-checking helpers for the CUDA runtime API. Following the Google C++
// style guide (no exceptions), a failed CUDA call is treated as a fatal,
// unrecoverable error: we log the failing expression and location and exit.

#ifndef CUDA_BATCH_CANNY_SRC_CUDA_CHECK_H_
#define CUDA_BATCH_CANNY_SRC_CUDA_CHECK_H_

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace canny {

inline void CheckCuda(cudaError_t status, const char* expr, const char* file,
                      int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n  in: %s\n",
                 static_cast<int>(status), cudaGetErrorString(status), file,
                 line, expr);
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace canny

#define CUDA_CHECK(expr) ::canny::CheckCuda((expr), #expr, __FILE__, __LINE__)

// Checks for errors from the most recent kernel launch (bad configuration,
// too much shared memory, ...). Execution errors surface at the next
// synchronizing call, which is also wrapped in CUDA_CHECK.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

#endif  // CUDA_BATCH_CANNY_SRC_CUDA_CHECK_H_
