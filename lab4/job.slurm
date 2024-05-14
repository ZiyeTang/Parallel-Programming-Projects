#!/bin/bash
#SBATCH --job-name="lab4"
#SBATCH --output="lab4.out"
#SBATCH --error="lab4.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:01:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun ./lab4 -e data/0/output.dat -i data/0/input.dat,data/0/kernel.dat -o /tmp/myoutput.dat -t matrix 
srun ./lab4 -e data/1/output.dat -i data/1/input.dat,data/1/kernel.dat -o /tmp/myoutput.dat -t matrix 
srun ./lab4 -e data/2/output.dat -i data/2/input.dat,data/2/kernel.dat -o /tmp/myoutput.dat -t matrix 
srun ./lab4 -e data/3/output.dat -i data/3/input.dat,data/3/kernel.dat -o /tmp/myoutput.dat -t matrix 
srun ./lab4 -e data/4/output.dat -i data/4/input.dat,data/4/kernel.dat -o /tmp/myoutput.dat -t matrix 
srun ./lab4 -e data/5/output.dat -i data/5/input.dat,data/5/kernel.dat -o /tmp/myoutput.dat -t matrix 