// Steady-state launch overhead, measured after everything is warm.
// Compares <<<>>>, cudaLaunchKernel and cuLaunchKernel on an empty kernel:
//   enqueue  host time per launch when launches are queued back-to-back
//   sync     launch + wait round trip (median), i.e. launch latency

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "common.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                       \
              cudaGetErrorString(e_));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult r_ = (call);                                                      \
    if (r_ != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "%s:%d: CUresult %d\n", __FILE__, __LINE__, (int)r_);    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void empty_kernel(int) {}

constexpr int kEnqueueIters = 20000;
constexpr int kSyncIters = 2000;

template <typename Launch>
static void measure(const char *name, Launch launch) {
  for (int i = 0; i < 100; i++) launch();  // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());

  StageTimer t;
  for (int i = 0; i < kEnqueueIters; i++) launch();
  double enqueue_ms = t.lap();
  CUDA_CHECK(cudaDeviceSynchronize());
  t.lap();

  std::vector<double> sync_us(kSyncIters);
  for (int i = 0; i < kSyncIters; i++) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    sync_us[i] = t.lap() * 1000.0;
  }
  std::sort(sync_us.begin(), sync_us.end());

  printf("{\"api\":\"%s\",\"enqueue_us\":%.4f,\"sync_us_p50\":%.4f,"
         "\"sync_us_p90\":%.4f}\n",
         name, enqueue_ms * 1000.0 / kEnqueueIters, sync_us[kSyncIters / 2],
         sync_us[kSyncIters * 9 / 10]);
  fflush(stdout);
}

int main() {
  CUDA_CHECK(cudaFree(0));

  int arg = 0;
  void *args[] = {&arg};

  // Same kernel, reached through the driver API handle.
  cudaFunction_t rt_fn;
  CUDA_CHECK(cudaGetFuncBySymbol(&rt_fn, (const void *)empty_kernel));
  CUfunction cu_fn = (CUfunction)rt_fn;

  measure("runtime <<<>>>", [&] { empty_kernel<<<1, 1>>>(arg); });
  measure("cudaLaunchKernel", [&] {
    CUDA_CHECK(cudaLaunchKernel((const void *)empty_kernel, dim3(1), dim3(1),
                                args, 0, nullptr));
  });
  measure("cuLaunchKernel", [&] {
    CU_CHECK(cuLaunchKernel(cu_fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  });
  return EXIT_SUCCESS;
}
