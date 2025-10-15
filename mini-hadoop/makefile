# ============================
# MPI + OpenMP build Makefile
# ============================

# Inherit cluster settings
CXX ?= mpic++
MPICXX ?= $(CXX)
CXXFLAGS ?= -Wall -Wextra -Wpedantic -std=c++20 -O3 -fopenmp -DOMPI_SKIP_MPICXX
LDFLAGS ?= -fopenmp
INCLUDES = -Iinclude

SRC_DIR := src
BUILD_DIR := build

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/modes_static.cpp $(SRC_DIR)/modes_dynamic.cpp
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

BIN := $(BUILD_DIR)/mpi_text_hybrid

# Build target
$(BIN): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

# Object build rule
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp include/*.hpp
	@mkdir -p $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
