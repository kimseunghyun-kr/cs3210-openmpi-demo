#!/bin/bash
#SBATCH -J hybrid_mpi_text
#SBATCH -o logs_%j.out
#SBATCH -e logs_%j.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -t 00:15:00
#SBATCH -p short
#SBATCH --mem=16G

# --- modules & env ---
module load gcc openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# --- parameters ---
MODE=${1:-dynamic}          # default to dynamic if not specified
CORPUS=${2:-/data/corpus.txt}
TOP=${3:-30}

echo "Hybrid parallelism: $SLURM_NTASKS MPI × $OMP_NUM_THREADS OMP threads"
echo "Mode: $MODE"

# --- run ---
srun ./build/mpi_text_hybrid $MODE $CORPUS --top $TOP --chunk-lines 500 --bar-width 60
