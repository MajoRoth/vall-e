#!/bin/zsh
#SBATCH -c 8
#SBATCH --mem=20g
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:10g
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8
#SBATCH --array=0-19%5
#SBATCH --exclude=gsm-04
##SBATCH --killable
##SBATCH --requeue

dir=/cs/labs/daphna/avihu.dekel/FixMatch-pytorch
cd $dir

source /cs/usr/avihu.dekel/.zshrc
conda activate inari
module load torch
module load nvidia
python3 runner_fixmatch.py --id ${SLURM_ARRAY_TASK_ID}