#!/bin/bash
#SBATCH --job-name="lab7"
#SBATCH --output="lab7.out"
#SBATCH --error="lab7.err"
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

srun ./lab7 -e data/0/output.ppm -i data/0/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/1/output.ppm -i data/1/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/2/output.ppm -i data/2/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/3/output.ppm -i data/3/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/4/output.ppm -i data/4/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/5/output.ppm -i data/5/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/6/output.ppm -i data/6/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/7/output.ppm -i data/7/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/8/output.ppm -i data/8/input.ppm -o /tmp/myoutput.raw -t image
srun ./lab7 -e data/9/output.ppm -i data/9/input.ppm -o /tmp/myoutput.raw -t image
