#!/bin/bash
#SBATCH --job-name=train_ar
#SBATCH -c 16
#SBATCH --mem=32g
#SBATCH --time=3-0
#SBATCH --gres=gpu:1,vmem:24g
#SBATCH --exclude=gsm-04

dir=/cs/labs/adiyoss/amitroth/vall-e
cd $dir

source /cs/labs/adiyoss/amitroth/anaconda3/etc/profile.d/conda.sh
conda activate vall-e
conda info | egrep "conda version|active environment"
echo "ar"
python --version

module load cuda/11.7
module load cudnn

python -m vall_e.train yaml=config/hebrew/ar.yml

