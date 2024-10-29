#!/bin/bash

# This script does multiple runs of run_evaluation.py, evaluate_predictions.py or
# evaluate_attribution.py in sequence.
# It requires an txt file where each line contains the arguments for a single run.
# It loops over the lines in the file and runs the python script with the arguments
# on each line.

# Usage:
# 1. Change to experiment file of your choice
EXPERIMENT_FILE="/path/to/repo/experiments/templates/post-hoc_runs.txt"
# 2. Choose to run evaluate_predictions.py or run_evaluation.py by commenting out
# line 39, 40 or 41.
# 3. Run this script in terminal:
# bash jobs_iter.sh

source /path/to/conda/bin/activate extraction_benchmark_vllm
module load cuda/11.8
export TOKENIZERS_PARALLELISM=false




# Initialize variable that will hold command line arguments
EXPERIMENT=""
# This function loops over the lines in EXPERIMENT_FILE argument and runs
# evaluate_predictions.py, run_evaluation.py or evaluate_attribution.py
# with the arguments in that line.
# Choose to run evaluate_predictions.py or run_evaluation.py by commenting out
# line 35, 36 or 37.
function run_experiments {
  # Get the file as the first command line argument
  FILE=$1
  i=0
  # Loop over lines in file
  while read line; do
  i=$(( i + 1 ))
  EXPERIMENT="$line";
  # python evaluate_predictions.py $EXPERIMENT
  python run_evaluation.py slurm_job_id="$SLURM_JOB_ID" $EXPERIMENT
  # python evalate_attribution.py $EXPERIMENT
  done <"$FILE"
}

# Run
run_experiments $EXPERIMENT_FILE
