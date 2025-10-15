#!/bin/bash
#SBATCH -J hybrid_mpi_text
#SBATCH -o logs_%j.out
#SBATCH -e logs_%j.err
#SBATCH -N 2                   # number of physical nodes
#SBATCH -n 16                  # total number of MPI ranks
#SBATCH --ntasks-per-node=8    # ranks per node
#SBATCH --cpus-per-task=8      # OpenMP threads per rank
#SBATCH -t 00:15:00
#SBATCH -p i7-13700
#SBATCH --mem=16G

# --- modules & env ---
module load gcc openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# --- parameters ---
MODE=${1:-dynamic}          # default to dynamic mode
CORPUS=${2:-/data/corpus.txt}
TOP=${3:-30}

echo "Hybrid parallelism: $SLURM_NTASKS MPI × $OMP_NUM_THREADS OMP threads"
echo "Mode: $MODE"
echo "Nodes: $SLURM_JOB_NUM_NODES"

# --- run ---
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS \
     ./build/mpi_text_hybrid $MODE $CORPUS \
     --top $TOP --chunk-lines 500 --bar-width 60
