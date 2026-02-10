#!/bin/bash
#SBATCH --job-name=test        # Job name
#SBATCH --output=log%j.log       # Both stdout and stderr go here
#SBATCH --error=log%j.log        # Optional, can be the same file
#SBATCH --time=00:10:00             # Max runtime
#SBATCH --partition=ENSTA-h100      # Partition
#SBATCH --nodelist=ensta-h10001.r2.enst.fr
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --cpus-per-task=4           # CPUs
#SBATCH --mem=16G                   # Memory

# Activate virtual environment
nvidia-smi
source $HOME/dl_env/bin/activate

cd $HOME/efficient_image_generation
# Run Python script and merge stderr into stdout
python test_loader.py 2>&1