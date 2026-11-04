#!/usr/bin/env bash
#SBATCH --job-name=m2v_q9
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/home/wertheizer/structeredML/HW2/m2v_q9.log
#SBATCH --error=/home/wertheizer/structeredML/HW2/m2v_q9.err

set -euo pipefail
ENV=/home/wertheizer/miniforge3/envs/structml1
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
cd /home/wertheizer/structeredML/HW2

$ENV/bin/python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
stdbuf -oL -eL $ENV/bin/python -u retrain_m2v_and_q9.py
