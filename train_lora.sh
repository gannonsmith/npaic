#!/bin/bash

#SBATCH --job-name=proj_gen_summaries
#SBATCH --account=eecs595f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=gen_summary.log


echo "Starting Inference on base model..."

# Use these hyperparameters for your full SFT training
python -m src.preprocessing.generate_summaries

echo "Execution on full dataset finished."
