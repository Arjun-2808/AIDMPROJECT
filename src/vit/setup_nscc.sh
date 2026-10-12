#!/bin/bash
# One-time setup on the ASPIRE 2A *login node* (it has internet; compute nodes may not).
#   bash setup_nscc.sh [path/to/datasets.zip]      (default: $WORK/datasets.zip)
# Everything (env, data, runs, caches) lives under WORK=~/scratch/AIDM, not in the 50 GB home.
# Writes jobs/paths.sh, which every PBS job sources.
set -eo pipefail
cd "$(dirname "$0")"
WORK=${WORK:-$HOME/scratch/AIDM}
ENV=${ENV:-$WORK/env}
ZIP=${1:-$WORK/datasets.zip}
mkdir -p "$WORK"/{data,runs,hf_cache,tmp} logs
export CONDA_PKGS_DIRS=$WORK/tmp/conda_pkgs PIP_CACHE_DIR=$WORK/tmp/pip_cache  # keep caches off home
echo ">> work dir: $WORK"

# ---- python env (NSCC miniforge3 module, otherwise a private Miniforge)
if [ ! -x "$ENV/bin/python" ]; then
  type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
  for m in miniforge3 anaconda3 miniconda3; do module load "$m" 2>/dev/null && command -v conda >/dev/null && break; done
  if ! command -v conda >/dev/null; then
    echo ">> no conda module found, installing Miniforge to $WORK/miniforge3"
    curl -fsSL -o "$WORK/tmp/miniforge.sh"       https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$WORK/tmp/miniforge.sh" -b -p "$WORK/miniforge3"
    export PATH=$WORK/miniforge3/bin:$PATH
  fi
  conda create -y -p "$ENV" -c conda-forge --override-channels python=3.11  # conda-forge: no Anaconda ToS prompt
fi
PY=$ENV/bin/python
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
"$PY" -m pip install -q timm==1.0.30 scikit-learn pandas matplotlib pillow
"$PY" -c "import torch, timm; print('>> torch', torch.__version__, 'cuda', torch.version.cuda, '| timm', timm.__version__)"

# ---- Dogs vs. Cats
DATA=$WORK/data/datasets
if [ ! -d "$DATA/train/cat" ]; then
  [ -f "$ZIP" ] || { echo "!! $ZIP not found - upload datasets.zip first (see README)"; exit 1; }
  echo ">> unzipping $ZIP"
  if command -v unzip >/dev/null; then
    unzip -q -o "$ZIP" -d "$WORK/data" -x "__MACOSX/*" "*.DS_Store"
  else
    "$PY" -m zipfile -e "$ZIP" "$WORK/data" && rm -rf "$WORK/data/__MACOSX"
  fi
fi
for s in train/cat train/dog val/cat val/dog test; do
  printf ">> %-10s %6d images\n" "$s" "$(ls "$DATA/$s" | wc -l)"
done

# ---- CIFAR-10 (python version)
CIFAR=$WORK/data/cifar10
if [ ! -f "$CIFAR/cifar-10-batches-py/test_batch" ]; then
  mkdir -p "$CIFAR"
  curl -fL -o "$CIFAR/cifar-10-python.tar.gz" https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  echo "c58f30108f718f92721af3b95e74349a  $CIFAR/cifar-10-python.tar.gz" | md5sum -c -
  tar xzf "$CIFAR/cifar-10-python.tar.gz" -C "$CIFAR" && rm "$CIFAR/cifar-10-python.tar.gz"
fi

# ---- pretrained weights into a cache the compute nodes can read offline
export HF_HOME=$WORK/hf_cache
"$PY" -c "
import timm
from models import TIMM_IDS
for name, tid in TIMM_IDS.items():
    timm.create_model(tid, pretrained=True)
    print('>> cached', name, tid)
"

cat > jobs/paths.sh <<EOF
# written by setup_nscc.sh
export PY=$PY
export IE4483_DATA=$DATA
export IE4483_CIFAR=$CIFAR
export IE4483_OUT=$WORK/runs
export HF_HOME=$WORK/hf_cache
export PROJECT=\${PROJECT:-}   # NSCC project id; set only if qsub asks for -P
EOF
echo ">> wrote jobs/paths.sh"
"$PY" sweep.py dvc --count >/dev/null && echo ">> sweep: $("$PY" sweep.py dvc --count) dvc configs, $("$PY" sweep.py cifar --count) cifar configs"
command -v myprojects >/dev/null && myprojects || true
echo ">> setup done. Next: bash jobs/submit.sh smoke"
