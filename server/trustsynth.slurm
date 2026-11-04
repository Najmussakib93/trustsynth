#!/usr/bin/env bash
#SBATCH --job-name=trustsynth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=results/private/slurm-%j.out
#SBATCH --error=results/private/slurm-%j.err

# Set your actual account/partition through sbatch flags. Submit from repo root.
# Create results/private before submission, so Slurm can open its log files.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
bash server/run.sh cuda
