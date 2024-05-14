#!/bin/bash
#SBATCH --job-name="lab6"
#SBATCH --output="lab6.out"
#SBATCH --error="lab6.err"
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

srun ./lab6 -e data/0/output.raw -i data/0/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/1/output.raw -i data/1/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/2/output.raw -i data/2/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/3/output.raw -i data/3/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/4/output.raw -i data/4/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/5/output.raw -i data/5/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/6/output.raw -i data/6/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/7/output.raw -i data/7/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/8/output.raw -i data/8/input.raw -o /tmp/myoutput.raw -t vector 
srun ./lab6 -e data/9/output.raw -i data/9/input.raw -o /tmp/myoutput.raw -t vector 
