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

# remove any .log files in the directory except the one for this job
# Use SLURM_JOB_ID if available, otherwise fall back to PID
LOGFILE="log${SLURM_JOB_ID:-$$}.log"
# Ensure the log file exists so it won't be deleted
touch "$LOGFILE"
# Avoid errors when there are no .log files
shopt -s nullglob
for f in *.log; do
  [ "$f" = "$LOGFILE" ] && continue
  rm -f -- "$f"
done
shopt -u nullglob

python test_loader.py 2>&1