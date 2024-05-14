#!/bin/bash
#SBATCH --job-name="lab8"
#SBATCH --output="lab8.out"
#SBATCH --error="lab8.err"
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

srun ./lab8 -e data/0/output.raw -i data/0/col.raw,data/0/row.raw,data/0/data.raw,data/0/vec.raw -o /tmp/myoutput.raw -t vector
srun ./lab8 -e data/1/output.raw -i data/1/col.raw,data/1/row.raw,data/1/data.raw,data/1/vec.raw -o /tmp/myoutput.raw -t vector
srun ./lab8 -e data/2/output.raw -i data/2/col.raw,data/2/row.raw,data/2/data.raw,data/2/vec.raw -o /tmp/myoutput.raw -t vector
srun ./lab8 -e data/3/output.raw -i data/3/col.raw,data/3/row.raw,data/3/data.raw,data/3/vec.raw -o /tmp/myoutput.raw -t vector
srun ./lab8 -e data/4/output.raw -i data/4/col.raw,data/4/row.raw,data/4/data.raw,data/4/vec.raw -o /tmp/myoutput.raw -t vector
srun ./lab8 -e data/5/output.raw -i data/5/col.raw,data/5/row.raw,data/5/data.raw,data/5/vec.raw -o /tmp/myoutput.raw -t vector
srun ./lab8 -e data/6/output.raw -i data/6/col.raw,data/6/row.raw,data/6/data.raw,data/6/vec.raw -o /tmp/myoutput.raw -t vector