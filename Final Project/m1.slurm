#!/bin/bash
#SBATCH --job-name="Milestone1"
#SBATCH --output="Milestone1.out"
#SBATCH --error="Milestone1.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun ./m1 1000 > m1.out

# Generate running time information
{ time srun ./m1 1000 > m1.out; } 2> time.out

# Generate gprof output
srun ./m1 1000 && gprof -Q ./m1 gmon.out > outfile
