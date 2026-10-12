# Sourced by every PBS job: project dir, paths from setup_nscc.sh, offline model cache.
cd "${PBS_O_WORKDIR:?run via qsub from the project directory}"
source jobs/paths.sh
export HF_HUB_OFFLINE=1           # use the weights cached on the login node
export OMP_NUM_THREADS=${NCPUS:-8}
export PYTHONUNBUFFERED=1
echo "job $PBS_JOBID index ${PBS_ARRAY_INDEX:-none} on $(hostname), $(nproc) cpus, $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
