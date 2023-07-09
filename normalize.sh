#!/bin/bash
#SBATCH --job-name=normalize_geekonomy
#SBATCH -c4
#SBATCH --mem=16g
#SBATCH --time=7-0
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --exclude=gsm-04
#SBATCH --array=1-50
#SBATCH --output=/cs/labs/adiyoss/amitroth/vall-e/slurm_outputs/%A_%a_%x.out

dir=/cs/labs/adiyoss/amitroth/vall-e
cd $dir

source /cs/labs/adiyoss/amitroth/anaconda3/etc/profile.d/conda.sh
conda activate vall-e
conda info | egrep "conda version|active environment"
echo "prepare data"
python --version

module load cuda/11.7
module load cudnn

python prepare_datasets.py normalize geekonomy ${SLURM_ARRAY_TASK_ID} 50

