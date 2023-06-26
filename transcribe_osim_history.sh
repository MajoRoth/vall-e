#!/bin/bash
#SBATCH --job-name=whisper
#SBATCH -c 8
#SBATCH --mem=16g
#SBATCH --time=3-0
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --exclude=gsm-04

dir=/cs/labs/adiyoss/amitroth/vall-e
cd $dir

source /cs/labs/adiyoss/amitroth/anaconda3/etc/profile.d/conda.sh
conda activate vall-e
conda info | egrep "conda version|active environment"
echo "prepare data"
python --version

module load cuda/11.7
module load cudnn

python prepare_datasets.py transcribe osim-history ${SLURM_ARRAY_TASK_ID} 30

