#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=32
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --exclude=gsm-04

dir=/cs/labs/adiyoss/amitroth/vall-e
cd $dir

conda activate vall-e
conda info | egrep "conda version|active environment"

module load cuda/11.7
module load cudnn

python -m vall_e.train yaml=config/saspeech/nar.yml

