# directories
TOP_DIR := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)
BUILD_DIR := $(TOP_DIR)/build
NETWORK_DIR := $(TOP_DIR)/network
MODEL_DIR := $(TOP_DIR)/model
NGEN_DIR := $(TOP_DIR)/neural_gen

# C compiler
CXXFLAGS := -Wall -Wno-ignored-attributes -Werror -std=c++17
CLFLAGS := -framework OpenCL
CXX := g++-10 $(CXXFLAGS)

# NeuralGen
NGEN := python3 $(NGEN_DIR)

# checker
CHECKER := $(TOP_DIR)/utils/check.py

# other configurations
TEST_DIR := $(TOP_DIR)/debug/test
CL_PLAT_DEV := 0 2

# files
NGEN_SRCS := $(wildcard $(NGEN_DIR)/*.py)
NGEN_SRCS += $(wildcard $(NGEN_DIR)/**/**/*.cpp)
NGEN_SRCS += $(wildcard $(NGEN_DIR)/**/**/*.h)
NGEN_SRCS += $(wildcard $(NGEN_DIR)/**/**/*.cl)
NETWORKS := $(BUILD_DIR)/cpu $(BUILD_DIR)/cpu_o3 $(BUILD_DIR)/cpu_o3_omp
NETWORKS += $(BUILD_DIR)/cpu_o3_simd4 $(BUILD_DIR)/cpu_o3_simd8
NETWORKS += $(BUILD_DIR)/cpu_o3_omp_simd4 $(BUILD_DIR)/cpu_o3_omp_simd8
NETWORKS += $(BUILD_DIR)/cl $(BUILD_DIR)/cl_opt
NETWORK_SRCS := $(patsubst $(BUILD_DIR)/%, $(BUILD_DIR)/%.cpp, $(NETWORKS))


.PHONY: all clean test

all: $(BUILD_DIR) $(NETWORKS)

clean:
	-rm $(NETWORKS) $(NETWORK_SRCS)

test: $(BUILD_DIR) $(NETWORKS)
	-$(CHECKER) $(BUILD_DIR)/cpu $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cpu_o3 $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cpu_o3_omp $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cpu_o3_simd4 $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cpu_o3_simd8 $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cpu_o3_omp_simd4 $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cpu_o3_omp_simd8 $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cl $(CL_PLAT_DEV) $(MODEL_DIR)/lenet5.model $(TEST_DIR)
	-$(CHECKER) $(BUILD_DIR)/cl_opt $(CL_PLAT_DEV) $(MODEL_DIR)/lenet5.model $(TEST_DIR)

$(BUILD_DIR):
	-mkdir $@

$(BUILD_DIR)/cpu: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g cpp -o $@.cpp
	$(CXX) $@.cpp -o $@

$(BUILD_DIR)/cpu_o3: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g cpp -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3

$(BUILD_DIR)/cpu_o3_omp: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g cpp -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3 -fopenmp

$(BUILD_DIR)/cpu_o3_simd4: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g cpp -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3 -march=native -DSIMD_VEC_LEN=4

$(BUILD_DIR)/cpu_o3_simd8: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g cpp -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3 -march=native -DSIMD_VEC_LEN=8

$(BUILD_DIR)/cpu_o3_omp_simd4: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g cpp -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3 -fopenmp -march=native -DSIMD_VEC_LEN=4

$(BUILD_DIR)/cpu_o3_omp_simd8: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g cpp -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3 -fopenmp -march=native -DSIMD_VEC_LEN=8

$(BUILD_DIR)/cl: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g opencl -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3 $(CLFLAGS)

$(BUILD_DIR)/cl_opt: $(NGEN_SRCS)
	$(NGEN) $(NETWORK_DIR)/lenet5.json -g opencl-opt -o $@.cpp
	$(CXX) $@.cpp -o $@ -O3 $(CLFLAGS)
