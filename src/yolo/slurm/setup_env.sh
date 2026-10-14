#!/bin/bash
# One-time setup, run ON THE LOGIN NODE from src/yolo:   bash slurm/setup_env.sh /projects/<your_project>
# (creating a conda env + pip install is the "simple setup" the cluster docs allow on login nodes;
#  nothing here trains or preprocesses data.)
set -eo pipefail
PROJECT_DIR="$(cd "${1:?usage: bash slurm/setup_env.sh /projects/<your_project>}" && pwd)"
cd "$(dirname "$0")/.."

echo "export PROJECT_DIR=$PROJECT_DIR" > slurm/project.env
. slurm/cluster_env.sh
mkdir -p "$PROJECT_DIR"/{data,runs,weights,envs}

module load Miniforge3
ENV="$PROJECT_DIR/envs/yolo"
PY="$ENV/bin/python"
# check for the interpreter, not just the folder: an interrupted `conda create` leaves a half-built dir behind
if [ ! -x "$PY" ]; then
    rm -rf "$ENV"
    conda create -y -p "$ENV" python=3.12
fi
[ -x "$PY" ] || { echo "ERROR: $PY missing after conda create" >&2; exit 1; }
source activate "$ENV"

# Always go through the env's own interpreter so packages can never land in another environment.
echo "== installing with $PY ($("$PY" --version))"
"$PY" -m pip install --no-cache-dir -r requirements.txt
# Compute nodes have no libGL: use the headless OpenCV build. Both packages install into the same cv2/
# folder, so remove BOTH and reinstall headless fresh (removing only one leaves cv2 half-deleted and pip
# then reports headless as "already satisfied"). Also safe to re-run.
"$PY" -m pip uninstall -y opencv-python opencv-python-headless
"$PY" -m pip install --no-cache-dir opencv-python-headless
"$PY" -c "import torch, ultralytics, cv2; print('torch', torch.__version__, 'cuda-build', torch.version.cuda, '| ultralytics', ultralytics.__version__, '| cv2', cv2.__version__)"
du -sh "$ENV"
echo "SETUP OK. Next (with the env active):  python download_weights.py  and  sbatch slurm/prep_data.sbatch"
