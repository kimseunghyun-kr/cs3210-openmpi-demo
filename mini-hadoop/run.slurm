#!/bin/bash
#SBATCH -J hybrid_mpi_text
#SBATCH -o logs_%j.out
#SBATCH -e logs_%j.err
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -p i7-13700
#SBATCH -t 00:15:00
#SBATCH --mem=16G
# Optional: set this to enable OpenMP later
##SBATCH --cpus-per-task=4

set -e
cd "$SLURM_SUBMIT_DIR"

# ------------------------------
# OpenMP configuration
# ------------------------------
# Default to 1 thread if not explicitly set
if [ -z "$SLURM_CPUS_PER_TASK" ] || [ "$SLURM_CPUS_PER_TASK" -lt 1 ]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "[INFO] ====== Environment ======"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI ranks: $SLURM_NTASKS"
echo "OMP threads/rank: $OMP_NUM_THREADS"
echo "==============================="

# ------------------------------
# Build
# ------------------------------
echo "[INFO] Building binary..."
make clean && make all

# ------------------------------
# Run parameters
# ------------------------------
MODE=${1:-dynamic}
CORPUS=${2:-./oliver-twist.txt}
TOP=${3:-30}

echo "[INFO] ====== Running ======"
mpirun -np $SLURM_NTASKS \
    --map-by ppr:$((SLURM_NTASKS/SLURM_JOB_NUM_NODES)):node \
    ./build/mpi_text_hybrid "$MODE" "$CORPUS" \
    --top "$TOP" --chunk-lines 500 --bar-width 60
