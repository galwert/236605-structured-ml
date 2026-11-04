#!/usr/bin/env bash
# Execute hw2.ipynb end-to-end. Designed for srun --gres=gpu:l40:1
# Usage:
#   srun --gres=gpu:l40:1 --time=4:00:00 --mem=64G \
#        bash /home/wertheizer/structeredML/HW2/run_hw2.sh
set -euo pipefail

ENV=/home/wertheizer/miniforge3/envs/structml1
PY=$ENV/bin/python
PIP=$ENV/bin/pip
HW2=/home/wertheizer/structeredML/HW2

# Make sure conda's libstdc++ is loaded (optree needs GLIBCXX_3.4.31+).
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}

cd "$HW2"

# ----- 1. detect torch+cuda -----
$PY - <<'PYEOF'
import torch
print("torch =", torch.__version__, "| cuda =", torch.cuda.is_available(),
      "| device count =", torch.cuda.device_count(),
      "| build cuda =", torch.version.cuda)
PYEOF

# ----- 2. ensure the right torch-cluster wheel is installed -----
# if cuda is available, the cpu wheel will work too (since walks happen via
# torch_cluster's CPU ops) — Node2Vec parameter tensors still live on GPU.
# So no reinstall needed; CPU torch-cluster is fine here.

# ----- 3. SKIP py -> ipynb resync. Markdown answers (Q4, Q7, Q8, Q10-Q16) live
# only in hw2.ipynb; the .py is the stale scaffold. Just verify the notebook
# parses.
$PY - <<'PYEOF'
import json
with open('hw2.ipynb') as f:
    nb = json.load(f)
print(f"hw2.ipynb ready ({len(nb['cells'])} cells)")
PYEOF

# ----- 4. execute the notebook in-place -----
echo ">>> Executing hw2.ipynb (this is the long one)..."
# stdbuf -oL forces line-buffered stdout so nbconvert progress is visible in
# the slurm log live (otherwise it's block-buffered to 4 KB and we wait blind).
# --log-level=DEBUG emits a line per cell so we get a heartbeat.
stdbuf -oL -eL \
$PY -m jupyter nbconvert --to notebook --execute hw2.ipynb \
    --output hw2.ipynb \
    --log-level=INFO \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=structml1 2>&1

# ----- 5. also export to HTML (spec requirement) -----
$PY -m jupyter nbconvert --to html hw2.ipynb --output hw2.html

echo ">>> Done. Outputs:"
ls -la hw2.ipynb hw2.html embeddings.npz tsne_grid.png neighborhoods.png 2>/dev/null || true
