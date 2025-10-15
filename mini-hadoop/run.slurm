#!/bin/bash
#SBATCH -J hybrid_mpi_text
#SBATCH -p all
#SBATCH --constraint="[i7-13700*1&xs-4114*1]"
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Building MPI + OpenMP binary..."
make clean && make

echo "[INFO] ====== Runtime Environment ======"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI ranks: $SLURM_NTASKS"
echo "OMP threads/rank: $SLURM_CPUS_PER_TASK"
echo "======================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

MODE=${1:-dynamic}
CORPUS=${2:-./oliver-twist.txt}
TOP=${3:-30}

echo "[INFO] Launching job..."
srun ./build/mpi_text_hybrid "$MODE" "$CORPUS" \
     --top "$TOP" --chunk-lines 500 --bar-width 60
