#!/bin/bash
# Submit jobs from the login node.
#   bash jobs/submit.sh smoke      quick end-to-end check (do this first)
#   bash jobs/submit.sh all        both sweeps + feature/SVM baseline + analysis after the dvc sweep
#   bash jobs/submit.sh analyze    re-run analysis + summary only
set -eo pipefail
cd "$(dirname "$0")/.."
source jobs/paths.sh
mkdir -p logs
Q="qsub${PROJECT:+ -P $PROJECT}"   # set PROJECT in jobs/paths.sh only if your account needs -P

case "${1:-all}" in
  smoke)
    $Q jobs/smoke.pbs ;;
  all)
    ND=$("$PY" sweep.py dvc --count)
    NC=$("$PY" sweep.py cifar --count)
    DVC=$($Q -N dvc_sweep -v TASK=dvc -J 0-$((ND - 1)) jobs/sweep.pbs)
    CIF=$($Q -N cifar_sweep -v TASK=cifar -J 0-$((NC - 1)) jobs/sweep.pbs)
    FEAT=$($Q jobs/features.pbs)
    ANA=$($Q -W depend=afterany:"$DVC":"$CIF":"$FEAT" jobs/analyze.pbs)
    echo "dvc sweep   $DVC ($ND configs x 3 seeds)"
    echo "cifar sweep $CIF ($NC configs x 3 seeds)"
    echo "features    $FEAT"
    echo "analysis    $ANA (starts when the others finish)" ;;
  analyze)
    $Q jobs/analyze.pbs ;;
  *)
    echo "usage: bash jobs/submit.sh [smoke|all|analyze]"; exit 1 ;;
esac
