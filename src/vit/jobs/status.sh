#!/bin/bash
# Sweep progress: finished runs with accuracy and run time, then the PBS queue (R = running, Q = queued, H = held).
#   bash jobs/status.sh
cd "$(dirname "$0")/.."
source jobs/paths.sh
"$PY" - <<'EOF'
import json, os
from pathlib import Path

root = Path(os.environ["IE4483_OUT"])
for task in ["dvc", "cifar"]:
    runs = sorted(p for p in (root / task).glob("*") if p.is_dir())
    done = [r for r in runs if (r / "metrics.json").exists()]
    print(f"{task}: {len(done)} finished, {len(runs) - len(done)} in progress")
    for r in done:
        m = json.loads((r / "metrics.json").read_text())
        acc = m["best_val_acc"] if task == "dvc" else m["test"]["acc"]
        label = "val" if task == "dvc" else "test"
        print(f"  {r.name:62s} {label} {100 * acc:6.2f}%  {m['train_time_s'] / 60:5.1f} min")
EOF
echo "PBS subjobs:"
qstat -t -u "$USER" 2>/dev/null | tail -n +6 | awk '{print $10}' | sort | uniq -c
