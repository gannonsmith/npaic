#!/bin/bash

#SBATCH --job-name=lora_training
#SBATCH --account=eecs595f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=gen_summary.log


echo "Starting lora training..."

# Use these hyperparameters for your full SFT training
python -m src.personality.train_lora

echo "Lora training complete."
