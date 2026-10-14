# Sourced by every job script / setup step. Implements the cluster rules from
# https://github.com/NTUEEECluster/docs (guideline.md sections 8 and 9):
#   - scratch, pip/conda/HF/triton caches live on project storage, never /tmp (4 GB) or $HOME (50 GB)
#   - Python packages come from our own conda env, not system packages
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$_here/slurm/project.env" ] && . "$_here/slurm/project.env"
if [ -z "${PROJECT_DIR:-}" ] || [ ! -d "$PROJECT_DIR" ]; then
    echo "PROJECT_DIR is unset or missing. Run: bash slurm/setup_env.sh /projects/<your_project>" >&2
    return 1 2>/dev/null || exit 1
fi

export SCRATCH="$PROJECT_DIR/.tmp"
export TMPDIR="$SCRATCH"
export PIP_CACHE_DIR="$SCRATCH/pip" PIP_TMPDIR="$SCRATCH/pip"
export CONDA_PKGS_DIRS="$SCRATCH/conda"
export HF_HOME="$SCRATCH/hf"
export TRITON_CACHE_DIR="$SCRATCH/triton"
export XDG_CACHE_HOME="$SCRATCH/cache"
export TORCH_HOME="$SCRATCH/torch" MPLCONFIGDIR="$SCRATCH/mpl"
export YOLO_CONFIG_DIR="$PROJECT_DIR/.config/Ultralytics"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$SCRATCH" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
