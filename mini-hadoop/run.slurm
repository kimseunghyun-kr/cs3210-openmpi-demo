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

set -e  # stop if any command fails

# ----------------------------
# FIX: go back to submission dir
# ----------------------------
cd "$SLURM_SUBMIT_DIR"
PROJECT_DIR=$(pwd)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "[INFO] ====== Environment ======"
echo "Working directory: $PROJECT_DIR"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI ranks: $SLURM_NTASKS"
echo "OMP threads/rank: $OMP_NUM_THREADS"
echo "==============================="

# ----------------------------
# BUILD STEP
# ----------------------------
echo "[INFO] Building MPI binary on compute node..."
srun -N 1 -n 1 make -C "$PROJECT_DIR/mpi_text_hybrid" clean all

# ----------------------------
# RUN STEP
# ----------------------------
MODE=${1:-dynamic}
CORPUS=${2:-$PROJECT_DIR/oliver-twist.txt}
TOP=${3:-30}

echo "[INFO] ====== Running Job ======"
echo "Mode: $MODE"
echo "Corpus: $CORPUS"
echo "Top: $TOP"
echo "==============================="

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS \
     "$PROJECT_DIR/mpi_text_hybrid/build/mpi_text_hybrid" \
     "$MODE" "$CORPUS" \
     --top "$TOP" --chunk-lines 500 --bar-width 60
