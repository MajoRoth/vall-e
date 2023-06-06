#!/bin/bash
#SBATCH --job-name="tensorboard"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load tensorflow-all

conda activate vall-e

echo "Starting runs on port 16006."

tensorboard --logdir=runs --port=16006