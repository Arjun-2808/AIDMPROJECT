#!/bin/bash
#SBATCH --job-name=yolo-suite
#SBATCH --gpus=6000ada:1
#SBATCH --time=06:00:00
#SBATCH --output=yolo-%j.out
#SBATCH --error=yolo-%j.err
# Do NOT add --mem / --cpus-per-task: the cluster fixes 4 CPUs + a per-GPU RAM share and overrides them.
# Override anything at submit time, e.g.  sbatch --gpus=a40:1 --time=02:00:00 slurm/run_suite.sbatch --group models
#
# Submit from src/yolo:   sbatch slurm/run_suite.sbatch --group models
# Do not edit files under src/yolo while the job is pending: Slurm reads them when the job starts.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
. slurm/cluster_env.sh
module load Miniforge3
source activate "$PROJECT_DIR/envs/yolo"
# Call the env's interpreter by full path: never trust whatever `python` resolves to after activation.
PY="$PROJECT_DIR/envs/yolo/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY not found; run slurm/setup_env.sh first" >&2; exit 1; }
"$PY" -c "import sys, PIL, ultralytics; print('python:', sys.executable, '| ultralytics', ultralytics.__version__)"

echo "node: $(hostname)  job: $SLURM_JOB_ID"
nvidia-smi -L
# run_suite.py exits immediately (freeing the GPU) if CUDA is unusable or when the group is done
"$PY" run_suite.py "$@" --require-gpu
