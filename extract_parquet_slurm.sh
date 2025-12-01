#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=50g
#SBATCH -J "clash_royale_placement"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1

# Load CUDA modules
module load cuda12.6/toolkit
module load cuda12.6/blas
module load cuda12.6/fft
# Activate conda environment
conda activate clashroyale

# GPU debug info
gpu_debug

# Run detection
/home/ostikar/.conda/envs/clashroyale/bin/python extract_parquet_png.py --root ../hf_subset --arena arena_23