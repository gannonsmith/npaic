#!/bin/bash

#SBATCH --job-name=ft_inf
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=ft_inf.log


echo "Starting Inference on ft model..."

# Use these hyperparameters for your full SFT training
python -m src.generation.inference

echo "Execution on full dataset finished."
