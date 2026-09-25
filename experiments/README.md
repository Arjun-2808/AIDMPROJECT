# Experiment logs

One directory per run, owned by one person:

```
experiments/<owner>/<timestamp>_<run_name>/
    config.json     exact config + git SHA + command line that produced this run
    metrics.csv     one row per epoch: lr, train_loss, train_acc, val_loss, val_acc
    summary.json    final/best numbers — what the leaderboard reads
    train.log       stdout copy
    best.pt         checkpoint — GITIGNORED, stays on your machine
```

Created automatically by `train.py`; you don't make these by hand.

## Rules

1. **Only write inside your own folder.** `arjun/`, `member2/`, `member3/` —
   rename these to real names once the split is settled. This is what keeps
   three people out of each other's merge conflicts.
2. **Commit your logs after every run that's worth keeping.** They're a few KB
   of text. The point of this repo is that the other two can see what you've
   already tried without asking.
3. **Never commit `.pt` / `.pth`.** Already gitignored. Note where the
   checkpoint lives in your run's `summary.json` if it matters.
4. **Use `--note`** to say what the run was testing. It ends up in the
   leaderboard and it's what makes the table readable a month from now:

   ```bash
   python train.py --config configs/vit_s16_adamw.yaml --owner member2 \
     --set optim.lr=1e-4 --note "lr sweep 3/5"
   ```

## Reading someone else's run

```bash
cat experiments/member2/2026*_vit_s16_adamw/summary.json
```

`config.json` holds the full config *after* `--set` overrides, plus the git SHA,
so any run in here can be reproduced exactly.
