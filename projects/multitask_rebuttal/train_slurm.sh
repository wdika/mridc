#!/bin/sh
#SBATCH --job-name=MTLRS
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --nice=10

# conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/dkarkalousos/envs/mridc/

mridc run -c /home/dkarkalousos/PycharmProjects/mridc/projects/multitask_rebuttal/model_zoo/conf/base_mtlrs_train-fold1.yaml &

# Wait is needed to make sure slurm doesn't quit the job when the lines with '&' immediately return
wait
