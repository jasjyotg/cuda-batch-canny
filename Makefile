# cuda-batch-canny build file.
#
#   make               build bin/canny_batch and bin/generate_synthetic
#   make SM=75         build for a specific GPU (default: auto-detected)
#   make data          generate the synthetic datasets under data/
#   make run           build + run the full demo (./run.sh)
#   make format        clang-format all sources (Google style)
#   make lint          cpplint all sources (if cpplint is installed)
#   make clean

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc
CXX       ?= g++

# Target compute capability, e.g. 75 for a Tesla T4. Auto-detected from the
# first GPU; if detection fails nvcc's default arch is used and the embedded
# PTX is JIT-compiled for the actual GPU at load time.
SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.[:space:]')
ifneq ($(strip $(SM)),)
ARCH_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM) \
              -gencode arch=compute_$(SM),code=compute_$(SM)
endif

# --fmad=false stops nvcc fusing a*b+c into one FMA instruction. The CPU
# reference does not fuse, so this keeps GPU and CPU results bit-identical
# (checked by --verify) at a small cost in throughput.
CXXFLAGS  ?= -O3
CXXFLAGS  += -std=c++17 -Wall -Wextra -Isrc
NVCCFLAGS ?= -O3
NVCCFLAGS += -std=c++17 --fmad=false -lineinfo -Isrc $(ARCH_FLAGS) \
             -Xcompiler -Wall,-Wextra
# GCC 8 needs -lstdc++fs for std::filesystem: make LDLIBS=-lstdc++fs
LDLIBS    ?=

BIN_DIR := bin
OBJ_DIR := obj

CANNY_OBJS := $(OBJ_DIR)/main.o $(OBJ_DIR)/canny_gpu.o \
              $(OBJ_DIR)/canny_cpu.o $(OBJ_DIR)/canny_params.o \
              $(OBJ_DIR)/image_io.o $(OBJ_DIR)/flag_parser.o
GEN_OBJS   := $(OBJ_DIR)/tools/generate_synthetic.o \
              $(OBJ_DIR)/image_io.o $(OBJ_DIR)/flag_parser.o

SOURCES := $(wildcard src/*.cc src/*.h src/*.cu src/tools/*.cc)

.PHONY: all clean data run format lint help

all: $(BIN_DIR)/canny_batch $(BIN_DIR)/generate_synthetic

$(BIN_DIR)/canny_batch: $(CANNY_OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(ARCH_FLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/generate_synthetic: $(GEN_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LDLIBS)

$(OBJ_DIR)/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJ_DIR)/canny_gpu.o: src/canny_gpu.cu src/canny_gpu.h src/canny_ops.h \
                        src/canny_params.h src/cuda_check.h src/image_io.h
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

-include $(wildcard $(OBJ_DIR)/*.d $(OBJ_DIR)/tools/*.d)

# Synthetic datasets: hundreds of small images and tens of large ones.
data: $(BIN_DIR)/generate_synthetic
	./$(BIN_DIR)/generate_synthetic --output_dir=data/synthetic_small \
	    --prefix=small --count=500 --width=256 --height=256 --seed=1000
	./$(BIN_DIR)/generate_synthetic --output_dir=data/synthetic_color \
	    --prefix=color --count=100 --width=640 --height=480 --color --seed=5000
	./$(BIN_DIR)/generate_synthetic --output_dir=data/synthetic_large \
	    --prefix=large --count=12 --width=3840 --height=2160 --shapes=60 \
	    --seed=9000

run: all
	./run.sh

format:
	clang-format -i --style=file $(SOURCES)

lint:
	@command -v cpplint >/dev/null || { echo "pip install cpplint"; exit 1; }
	cpplint --root=.. --extensions=cc,h,cu --filter=-build/include_subdir $(SOURCES)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/canny_batch $(BIN_DIR)/generate_synthetic

help:
	@sed -n '3,10p' Makefile
