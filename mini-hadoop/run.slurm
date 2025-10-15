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

set -e  # stop if anything fails

# --------------------------------------------------------------------
# 1. Move back to the directory where you ran `sbatch run.slurm`
# --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
PROJECT_DIR=$(pwd)
echo "[INFO] Project directory: $PROJECT_DIR"

# --------------------------------------------------------------------
# 2. Set OpenMP environment
# --------------------------------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "[INFO] ====== Environment ======"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI ranks: $SLURM_NTASKS"
echo "OMP threads/rank: $OMP_NUM_THREADS"
echo "==============================="

# --------------------------------------------------------------------
# 3. Build the program INSIDE the allocation, in the right folder
# --------------------------------------------------------------------
if [ ! -d "$PROJECT_DIR/mpi_text_hybrid" ]; then
    echo "[ERROR] Directory $PROJECT_DIR/mpi_text_hybrid not found"
    exit 1
fi

echo "[INFO] Building MPI binary..."
srun -N1 -n1 bash -c "cd '$PROJECT_DIR/mpi_text_hybrid' && make clean && make all"

# --------------------------------------------------------------------
# 4. Run the program
# --------------------------------------------------------------------
MODE=${1:-dynamic}
CORPUS=${2:-$PROJECT_DIR/oliver-twist.txt}
TOP=${3:-30}

BIN="$PROJECT_DIR/mpi_text_hybrid/build/mpi_text_hybrid"
if [ ! -x "$BIN" ]; then
    echo "[ERROR] Binary $BIN not found or not executable"
    exit 1
fi

echo "[INFO] ====== Running Job ======"
echo "Mode: $MODE"
echo "Corpus: $CORPUS"
echo "Top: $TOP"
echo "==============================="

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS "$BIN" \
     "$MODE" "$CORPUS" \
     --top "$TOP" --chunk-lines 500 --bar-width 60
