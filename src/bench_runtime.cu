// Cold-start breakdown using the CUDA Runtime API, the way most apps use CUDA.
//
// The kernels are embedded in the executable by nvcc. Build variants decide
// what is embedded:
//   sass  -gencode arch=compute_86,code=sm_86       ready-to-run machine code
//   ptx   -gencode arch=compute_86,code=compute_86  PTX only -> driver JIT
// Module loading policy is chosen at run time with CUDA_MODULE_LOADING
// (LAZY is the default since CUDA 12.2; EAGER loads every kernel at init).

#include <cuda_runtime.h>

#include <vector>

#include "common.h"

#include KERNELS_FILE

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                       \
              cudaGetErrorString(e_));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main() {
  StageTimer t;

  // The first runtime call loads libcuda, runs cuInit and creates the primary
  // context; cudaFree(0) is the conventional way to trigger that explicitly.
  CUDA_CHECK(cudaFree(0));
  emit("runtime_init", t.lap());

  const size_t bytes = kN * sizeof(float);
  std::vector<float> host(kN, 1.0f);
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, host.data(), bytes, cudaMemcpyHostToDevice));
  emit("alloc_and_copy", t.lap());

  const int grid = (kN + kBlock - 1) / kBlock;
  k_000<<<grid, kBlock>>>(d_out, d_in, kN);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  emit("first_launch", t.lap());

  k_000<<<grid, kBlock>>>(d_out, d_in, kN);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  emit("second_launch", t.lap());

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
