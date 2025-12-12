#!/bin/bash

#SBATCH --job-name=proj_base_model
#SBATCH --account=eecs595f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=base_model.log


echo "Starting Inference on base model..."

# python score_essays.py --limit 10 --output-file test_results_1.jsonl --prompt-id 0
# python score_essays.py --limit 10 --output-file test_results_2.jsonl --prompt-id 1
# python score_essays.py --limit 10 --output-file test_results_3.jsonl --prompt-id 2

# Use these hyperparameters for your full SFT training
python -m src.generation.baseline_inference

echo "Execution on full dataset finished."
