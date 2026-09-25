# How three people share this repo

The short version: **each person owns their own files**. If you find yourself
editing a file with someone else's name on it, stop and ask first.

## Ownership map

| Area | Owner | Notes |
|---|---|---|
| `src/models/cnn.py` | member A | CNN track |
| `src/models/vit.py` | member B | ViT track |
| `src/models/baseline.py` | member C | third track — **topic TBD** |
| `configs/<yours>_*.yaml` | you | prefix your configs with your track |
| `experiments/<you>/` | you | your logs only |
| `src/data.py`, `src/engine.py`, `src/optim.py`, `train.py` | **shared** | discuss before changing |
| `results/leaderboard.md` | append-only | keep both sides on conflict |

Shared files are shared on purpose — they're what makes the three sets of
numbers comparable. Changing the augmentation defaults mid-project silently
invalidates everyone else's earlier runs, so raise it in the group chat first.

## Branching

```bash
git checkout -b vit/lr-sweep        # <track>/<what>
# ... work, commit ...
git push -u origin vit/lr-sweep
```

Open a PR into `main`, or just merge it yourself if it only touches your own
files. `main` should always be in a state where `python train.py --config
configs/smoke.yaml --owner <you>` runs.

## Before you push

```bash
git pull --rebase       # rebase, not merge — keeps history readable
```

If the rebase conflicts on `results/leaderboard.md`, keep **both** rows.

## Adding a model

Nothing outside your own file needs to change:

```python
# src/models/vit.py
from . import register

@register("my_model")
def _my_model(num_classes=2, pretrained=True, **kw):
    return SomeModule(...)
```

Then reference it from a config:

```yaml
model:
  name: my_model
  pretrained: true
```

Check it's visible: `python train.py --list-models`

## Adding an optimiser

`src/optim.py` already covers sgd, sgd_nesterov, adam, adamw, rmsprop, adagrad,
adadelta, nadam, and schedulers none/step/multistep/cosine/cosine_warm/plateau/
onecycle. If you need something outside that list, add it to the factory —
that file is shared, so mention it.

## Keep the tracks actually different

The comparison is only worth writing up if the three runs differ on a known
axis. Agree on what's fixed (image size, augmentation recipe, epochs, seed) and
vary one thing at a time. `config.json` in each run dir records everything, so
a disagreement about "what did you actually run" is always settleable.
