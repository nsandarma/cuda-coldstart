// Holds a CUDA context open until stdin is closed. Used to emulate "another
// CUDA process is already running" (e.g. a Jupyter kernel or a model server),
// which keeps the GPU powered and the driver initialized.

#include <cuda.h>

#include <cstdio>

int main() {
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
    fprintf(stderr, "hold_gpu: CUDA init failed\n");
    return 1;
  }
  printf("ready\n");
  fflush(stdout);
  while (getchar() != EOF) {
  }
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}
