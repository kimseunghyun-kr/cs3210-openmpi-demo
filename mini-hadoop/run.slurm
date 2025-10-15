#!/bin/bash
#SBATCH -J hybrid_mpi_text
#SBATCH -o logs_%j.out
#SBATCH -e logs_%j.err
#SBATCH -N 2                   # number of physical nodes
#SBATCH -n 8                   # total number of MPI ranks
#SBATCH --ntasks-per-node=4    # ranks per node
#SBATCH --cpus-per-task=4      # OpenMP threads per rank
#SBATCH -t 00:15:00
#SBATCH -p i7-13700
#SBATCH --mem=16G

# ===========================
#  BUILD + RUN IN ONE SCRIPT
# ===========================

set -e  # stop on error

# --- environment ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "[INFO] ====== Environment ======"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI ranks: $SLURM_NTASKS"
echo "OMP threads/rank: $OMP_NUM_THREADS"
echo "==============================="

# --- build step (inside allocation, using system OpenMPI) ---
echo "[INFO] Building MPI binary on compute nodes..."
srun -N 1 -n 1 make -C mpi_text_hybrid clean all

# --- parameters ---
MODE=${1:-dynamic}          # default to dynamic mode
CORPUS=${2:-./oliver-twist.txt}
TOP=${3:-30}

echo "[INFO] ====== Running Job ======"
echo "Mode: $MODE"
echo "Corpus: $CORPUS"
echo "Top: $TOP"
echo "==============================="

# --- run step ---
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS \
     ./mpi_text_hybrid/build/mpi_text_hybrid \
     "$MODE" "$CORPUS" \
     --top "$TOP" --chunk-lines 500 --bar-width 60
