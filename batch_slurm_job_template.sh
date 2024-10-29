#!/bin/bash

# This script runs slurm array jobs.
# It requires a text file where each line specifies the arguments for either
# run_evaluation.py, evaluate_predictions.py or evaluate_attribution.py.
# Each array task will run the python script with the arguments from one line
# of the text file. The line number is defined by the SLURM_ARRAY_TASK_ID
# variable, which is automatically set by SLURM.

# Usage:
# 1. Change this to experiment file of your choice.
EXPERIMENT_FILE="/path/to/repo/experiments/templates/post-hoc_runs.txt"
# 2. Determine which python script should be run by commenting respective
# line 55, 56 or 57
# 3. Submit slurm job by running the following command (will run 10 array jobs
# with first 10 lines of EXPERIMENT_FILE
# sbatch --array=1-10 batch_slurm_job_template.sh

# SLURM arguments
#SBATCH --job-name=eval
#SBATCH --output=/path/to/terminal/outputs
#SBATCH --mail-user=your-email@domain.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --mem=92GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:48gb"

echo "Start Python Script!"

# This variable stores the arguments from a specific line from the given text file
EXPERIMENT=""
# This function gets the contents of a specific line from the given text file
# and stores them in EXPERIMENT.
# The line is defined by the SLURM_ARRAY_TASK_ID, which is automatically set
# by SLURM.
function get_experiment {
  FILE=$1
  i=0
  while read line; do
  i=$(( i + 1 ))
  test $i = $SLURM_ARRAY_TASK_ID && EXPERIMENT="$line";
  done <"$FILE"
}

source /path/to/your/miniconda3/bin/activate lab_env
module load cuda/11.8
export TOKENIZERS_PARALLELISM=false

# Set EXPERIMENT
get_experiment $EXPERIMENT_FILE

# Run python script.
python run_evaluation.py slurm_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" $EXPERIMENT
# python evaluate_predictions.py $EXPERIMENT
# python evaluate_attribution.py $EXPERIMENT