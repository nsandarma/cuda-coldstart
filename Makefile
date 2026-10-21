CUDA_PATH ?= /opt/cuda
NVCC      = $(CUDA_PATH)/bin/nvcc
CXX       = clang++
SM        = 86

# Number of kernels per module: 1 = toy program, 256 = library-sized module
KERNEL_COUNTS = 1 32 256

CXXFLAGS  = -std=c++17 -O2 -Wall -Wextra -I$(CUDA_PATH)/include
NVCCFLAGS = -std=c++17 -O2 -Isrc -Ibuild

B = build
KERNEL_SRCS = $(KERNEL_COUNTS:%=$(B)/kernels_%.cu)
PTX         = $(KERNEL_COUNTS:%=$(B)/kernels_%.ptx)
CUBIN       = $(KERNEL_COUNTS:%=$(B)/kernels_%.cubin)
RT_SASS     = $(KERNEL_COUNTS:%=$(B)/rt_sass_%)
RT_PTX      = $(KERNEL_COUNTS:%=$(B)/rt_ptx_%)

all: $(B)/bench_driver $(B)/bench_launch $(B)/hold_gpu $(KERNEL_SRCS) $(PTX) $(CUBIN) $(RT_SASS) $(RT_PTX)

.SECONDARY: $(KERNEL_SRCS)

$(B):
	mkdir -p $@

$(B)/kernels_%.cu: gen_kernels.py | $(B)
	python3 gen_kernels.py $* $@

$(B)/kernels_%.ptx: $(B)/kernels_%.cu
	$(NVCC) -arch=compute_$(SM) -ptx -o $@ $<

$(B)/kernels_%.cubin: $(B)/kernels_%.cu
	$(NVCC) -arch=sm_$(SM) -cubin -o $@ $<

# Not linked against libcuda/libnvrtc: both are dlopen()ed and timed.
$(B)/bench_driver: src/bench_driver.cc src/common.h | $(B)
	$(CXX) $(CXXFLAGS) -o $@ $< -ldl

$(B)/rt_sass_%: src/bench_runtime.cu src/common.h $(B)/kernels_%.cu
	$(NVCC) $(NVCCFLAGS) -gencode arch=compute_$(SM),code=sm_$(SM) \
		-DKERNELS_FILE='"kernels_$*.cu"' -o $@ $<

$(B)/rt_ptx_%: src/bench_runtime.cu src/common.h $(B)/kernels_%.cu
	$(NVCC) $(NVCCFLAGS) -gencode arch=compute_$(SM),code=compute_$(SM) \
		-DKERNELS_FILE='"kernels_$*.cu"' -o $@ $<

$(B)/bench_launch: src/bench_launch.cu src/common.h | $(B)
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(SM) -o $@ $< -lcuda

$(B)/hold_gpu: src/hold_gpu.cc | $(B)
	$(CXX) $(CXXFLAGS) -o $@ $< -lcuda

bench: all
	python3 run_bench.py

.PHONY: all bench clean
clean:
	rm -rf $(B)
