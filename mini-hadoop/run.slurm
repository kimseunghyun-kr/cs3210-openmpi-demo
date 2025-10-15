#!/bin/bash
#SBATCH -J hybrid_mpi_text
#SBATCH -o logs_%j.out
#SBATCH -e logs_%j.err
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -p i7-13700
#SBATCH --mem=16G

set -e  # stop on error

# --------------------------------------------------------------------
# 1. Return to directory where you ran sbatch
# --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
PROJECT_DIR=$(pwd)

# --------------------------------------------------------------------
# 2. OpenMP configuration
# --------------------------------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "[INFO] ====== Environment ======"
echo "Working directory: $PROJECT_DIR"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI ranks: $SLURM_NTASKS"
echo "OMP threads/rank: $OMP_NUM_THREADS"
echo "==============================="

# --------------------------------------------------------------------
# 3. Build the binary from *this* directory
# --------------------------------------------------------------------
if [ ! -f "$PROJECT_DIR/makefile" ]; then
    echo "[ERROR] No makefile found in $PROJECT_DIR"
    exit 1
fi

echo "[INFO] Building MPI binary..."
srun -N1 -n1 bash -c "cd '$PROJECT_DIR' && make clean && make all"

# --------------------------------------------------------------------
# 4. Run the binary
# --------------------------------------------------------------------
MODE=${1:-dynamic}
CORPUS=${2:-$PROJECT_DIR/oliver-twist.txt}
TOP=${3:-30}

BIN="$PROJECT_DIR/build/mpi_text_hybrid"

if [ ! -x "$BIN" ]; then
    echo "[ERROR] Binary not found: $BIN"
    exit 1
fi

echo "[INFO] ====== Running Job ======"
echo "Mode: $MODE"
echo "Corpus: $CORPUS"
echo "Top: $TOP"
echo "==============================="

srun --mpi=pmi2 -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS "$BIN" \
     "$MODE" "$CORPUS" \
     --top "$TOP" --chunk-lines 500 --bar-width 60
