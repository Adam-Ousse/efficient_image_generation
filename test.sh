#!/bin/bash
#SBATCH --job-name=test        # Job name
#SBATCH --output=log%j.log       # Both stdout and stderr go here
#SBATCH --error=log%j.log        # Optional, can be the same file
#SBATCH --time=01:00:00             # Max runtime
#SBATCH --partition=ENSTA-l40s       # Partition
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --cpus-per-task=6           # CPUs
#SBATCH --mem=32G                   # Memory
#SBATCH --nodelist=ensta-l40s02.r2.enst.fr
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

python benchmark_offload.py  2>&1