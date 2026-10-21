// Cold-start breakdown using the CUDA Driver API (+ NVRTC).
//
// libcuda and libnvrtc are dlopen()ed at runtime (like Triton/PyTorch do), so
// the cost of loading the driver library itself is measured as a stage
// instead of being hidden in process startup.
//
// Usage: bench_driver <mode> <file>
//   mode = cubin        load precompiled SASS (no JIT)
//          ptx          load PTX, driver JIT-compiles it (cache via env vars)
//          nvrtc-ptx    compile CUDA source with NVRTC to PTX, then driver JIT
//          nvrtc-cubin  compile CUDA source with NVRTC straight to SASS

#include <cuda.h>
#include <dlfcn.h>
#include <nvrtc.h>

#include <cstring>
#include <vector>

#include "common.h"

#define STR_(x) #x
#define STR(x) STR_(x)

// cuda.h maps e.g. cuMemAlloc -> cuMemAlloc_v2; decltype and STR both see the
// expanded name, so we resolve exactly the ABI the header was written for.
#define CU_FNS(X)                                                              \
  X(cuInit)                                                                    \
  X(cuGetErrorString)                                                          \
  X(cuDeviceGet)                                                               \
  X(cuDevicePrimaryCtxRetain)                                                  \
  X(cuDevicePrimaryCtxRelease)                                                 \
  X(cuCtxSetCurrent)                                                           \
  X(cuCtxSynchronize)                                                          \
  X(cuModuleLoadData)                                                          \
  X(cuModuleUnload)                                                            \
  X(cuModuleGetFunction)                                                       \
  X(cuMemAlloc)                                                                \
  X(cuMemFree)                                                                 \
  X(cuMemcpyHtoD)                                                              \
  X(cuLaunchKernel)

#define NVRTC_FNS(X)                                                           \
  X(nvrtcGetErrorString)                                                       \
  X(nvrtcCreateProgram)                                                        \
  X(nvrtcCompileProgram)                                                       \
  X(nvrtcDestroyProgram)                                                       \
  X(nvrtcGetProgramLogSize)                                                    \
  X(nvrtcGetProgramLog)                                                        \
  X(nvrtcGetPTXSize)                                                           \
  X(nvrtcGetPTX)                                                               \
  X(nvrtcGetCUBINSize)                                                         \
  X(nvrtcGetCUBIN)

#define DECLARE(fn) static decltype(&fn) p_##fn;
CU_FNS(DECLARE)
NVRTC_FNS(DECLARE)

static void *open_lib(const char *const *names) {
  for (const char *const *n = names; *n; n++) {
    if (void *h = dlopen(*n, RTLD_NOW | RTLD_LOCAL)) return h;
  }
  fprintf(stderr, "dlopen failed: %s\n", dlerror());
  exit(EXIT_FAILURE);
}

static void *resolve(void *lib, const char *name) {
  void *p = dlsym(lib, name);
  if (!p) {
    fprintf(stderr, "dlsym %s failed\n", name);
    exit(EXIT_FAILURE);
  }
  return p;
}

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult r_ = (call);                                                      \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char *m_ = "?";                                                    \
      p_cuGetErrorString(r_, &m_);                                             \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, m_);                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NVRTC_CHECK(call)                                                      \
  do {                                                                         \
    nvrtcResult r_ = (call);                                                   \
    if (r_ != NVRTC_SUCCESS) {                                                 \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                       \
              p_nvrtcGetErrorString(r_));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Returns PTX (to_sass=false) or a cubin (to_sass=true) for the given source.
static std::string nvrtc_compile(const std::string &src, bool to_sass) {
  nvrtcProgram prog;
  NVRTC_CHECK(p_nvrtcCreateProgram(&prog, src.c_str(), "kernels.cu", 0, nullptr,
                                   nullptr));
  const char *opts[] = {to_sass ? "--gpu-architecture=sm_86"
                                : "--gpu-architecture=compute_86"};
  nvrtcResult res = p_nvrtcCompileProgram(prog, 1, opts);
  if (res != NVRTC_SUCCESS) {
    size_t log_size;
    p_nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    p_nvrtcGetProgramLog(prog, log.data());
    fprintf(stderr, "NVRTC compile failed:\n%s\n", log.c_str());
    exit(EXIT_FAILURE);
  }
  size_t size;
  std::string out;
  if (to_sass) {
    NVRTC_CHECK(p_nvrtcGetCUBINSize(prog, &size));
    out.resize(size);
    NVRTC_CHECK(p_nvrtcGetCUBIN(prog, out.data()));
  } else {
    NVRTC_CHECK(p_nvrtcGetPTXSize(prog, &size));
    out.resize(size);
    NVRTC_CHECK(p_nvrtcGetPTX(prog, out.data()));
  }
  NVRTC_CHECK(p_nvrtcDestroyProgram(&prog));
  return out;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s cubin|ptx|nvrtc-ptx|nvrtc-cubin <file>\n",
            argv[0]);
    return EXIT_FAILURE;
  }
  const std::string mode = argv[1];
  const bool use_nvrtc = mode.rfind("nvrtc", 0) == 0;

  // File I/O is not a CUDA cost: read input before the clock starts.
  const std::string input = read_file(argv[2]);

  StageTimer t;

  static const char *const cuda_names[] = {"libcuda.so.1", "libcuda.so",
                                           nullptr};
  void *libcuda = open_lib(cuda_names);
#define RESOLVE_CU(fn) p_##fn = (decltype(&fn))resolve(libcuda, STR(fn));
  CU_FNS(RESOLVE_CU)
  emit("load_libcuda", t.lap());

  CU_CHECK(p_cuInit(0));
  emit("cuInit", t.lap());

  CUdevice dev;
  CUcontext ctx;
  CU_CHECK(p_cuDeviceGet(&dev, 0));
  CU_CHECK(p_cuDevicePrimaryCtxRetain(&ctx, dev));
  CU_CHECK(p_cuCtxSetCurrent(ctx));
  emit("context_create", t.lap());

  std::string image = input;
  if (use_nvrtc) {
    static const char *const nvrtc_names[] = {
        "libnvrtc.so.13", "libnvrtc.so", "/opt/cuda/lib64/libnvrtc.so", nullptr};
    void *libnvrtc = open_lib(nvrtc_names);
#define RESOLVE_NVRTC(fn) p_##fn = (decltype(&fn))resolve(libnvrtc, STR(fn));
    NVRTC_FNS(RESOLVE_NVRTC)
    emit("load_libnvrtc", t.lap());

    image = nvrtc_compile(input, mode == "nvrtc-cubin");
    emit("nvrtc_compile", t.lap());
  }

  CUmodule module;
  CU_CHECK(p_cuModuleLoadData(&module, image.c_str()));
  emit("module_load", t.lap());

  CUfunction fn;
  CU_CHECK(p_cuModuleGetFunction(&fn, module, "k_000"));
  emit("get_function", t.lap());

  const size_t bytes = kN * sizeof(float);
  std::vector<float> host(kN, 1.0f);
  CUdeviceptr d_in, d_out;
  CU_CHECK(p_cuMemAlloc(&d_in, bytes));
  CU_CHECK(p_cuMemAlloc(&d_out, bytes));
  CU_CHECK(p_cuMemcpyHtoD(d_in, host.data(), bytes));
  emit("alloc_and_copy", t.lap());

  int n = kN;
  void *args[] = {&d_out, &d_in, &n};
  const unsigned grid = (kN + kBlock - 1) / kBlock;
  CU_CHECK(p_cuLaunchKernel(fn, grid, 1, 1, kBlock, 1, 1, 0, nullptr, args,
                            nullptr));
  CU_CHECK(p_cuCtxSynchronize());
  emit("first_launch", t.lap());

  CU_CHECK(p_cuLaunchKernel(fn, grid, 1, 1, kBlock, 1, 1, 0, nullptr, args,
                            nullptr));
  CU_CHECK(p_cuCtxSynchronize());
  emit("second_launch", t.lap());

  CU_CHECK(p_cuMemFree(d_in));
  CU_CHECK(p_cuMemFree(d_out));
  CU_CHECK(p_cuModuleUnload(module));
  CU_CHECK(p_cuDevicePrimaryCtxRelease(dev));
  return EXIT_SUCCESS;
}
