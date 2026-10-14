#!/bin/bash
#SBATCH --job-name=yolo-cpu
#SBATCH --time=00:30:00
#SBATCH --output=cpu-%j.out
#SBATCH --error=cpu-%j.err
# CPU-only (0 SU) wrapper for the light post-processing scripts, so torch never runs on the login node.
# Submit from src/yolo:   sbatch slurm/cpu_job.sbatch predict.py --run cd_s_ft
#                         sbatch slurm/cpu_job.sbatch aggregate.py
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
. slurm/cluster_env.sh
module load Miniforge3
source activate "$PROJECT_DIR/envs/yolo"
# Call the env's interpreter by full path: never trust whatever `python` resolves to after activation.
PY="$PROJECT_DIR/envs/yolo/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY not found; run slurm/setup_env.sh first" >&2; exit 1; }
"$PY" -c "import sys, PIL, ultralytics; print('python:', sys.executable, '| ultralytics', ultralytics.__version__)"
"$PY" "$@"
