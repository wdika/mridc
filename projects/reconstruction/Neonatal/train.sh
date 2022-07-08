#!/bin/sh
#SBATCH --job-name=mridc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:v100:1

# conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/dkarkalousos/envs/mridc/

# job
srun python -m mridc.launch \
--config-path /home/dkarkalousos/PycharmProjects/mridc/projects/reconstruction/Neonatal/conf/ \
--config-name base_cirim_train.yaml

# Wait is needed to make sure slurm doesn't quit the job when the lines with '&' immediately return
wait
