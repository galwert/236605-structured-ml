#!/usr/bin/env bash
#SBATCH --job-name=hw2
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/wertheizer/structeredML/HW2/hw2_slurm.log
#SBATCH --error=/home/wertheizer/structeredML/HW2/hw2_slurm.err

set -euo pipefail
cd /home/wertheizer/structeredML/HW2
bash run_hw2.sh
