# ============================================
# MPI + OpenMP Makefile (cross-node safe)
# ============================================

# Always use system OpenMPI compiler wrapper
MPICXX := mpicxx

# Use portable flags (no -march=native!)
CXXFLAGS := -O3 -std=c++20 -fopenmp -Wall -Wextra -Wno-sign-compare
LDFLAGS  := -fopenmp
INCLUDES := -Iinclude

SRC_DIR   := src
BUILD_DIR := build
SRCS      := $(SRC_DIR)/main.cpp $(SRC_DIR)/modes_static.cpp $(SRC_DIR)/modes_dynamic.cpp
OBJS      := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
BIN       := $(BUILD_DIR)/mpi_text_hybrid

# Default build target
all: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp include/*.hpp
	@mkdir -p $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
