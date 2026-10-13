#!/bin/bash
#SBATCH --job-name=yolo-prep
#SBATCH --time=01:00:00
#SBATCH --output=prep-%j.out
#SBATCH --error=prep-%j.err
# CPU-only (no --gpus): data arranging / checking / CIFAR-10 conversion. Costs 0 SU.
# Submit from src/yolo:   sbatch slurm/prep_data.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
. slurm/cluster_env.sh
module load Miniforge3
source activate "$PROJECT_DIR/envs/yolo"
# Call the env's interpreter by full path: never trust whatever `python` resolves to after activation.
PY="$PROJECT_DIR/envs/yolo/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY not found; run slurm/setup_env.sh first" >&2; exit 1; }
"$PY" -c "import sys, PIL, ultralytics; print('python:', sys.executable, '| ultralytics', ultralytics.__version__)"

"$PY" prep_catsdogs.py
"$PY" prep_cifar10.py
