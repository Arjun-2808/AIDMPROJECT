# YOLO track: Dogs vs Cats (+ CIFAR-10)

Ultralytics YOLO26-cls (and one YOLO11-cls) fine-tuned / probed / trained from scratch, then reused for the
CIFAR-10 extension.

**Training runs on the NTU EEE GPU cluster; using and testing the trained models runs on your laptop.**

```
laptop ── rsync code + data ──▶ cluster: setup → prep data → smoke test → sbatch training runs
laptop ◀── rsync best.pt ─────── cluster
laptop: evaluate / predict (submission.csv) / error analysis / aggregate   (section 5, no cluster needed)
```

The cluster half is written against the cluster guidelines (<https://github.com/NTUEEECluster/docs>): batch
jobs only, no `--mem`/`--cpus-per-task`, GPU model always named, scratch and caches on project storage, conda
via Lmod, nothing heavy on the login node.

Below, `<user>`, `<login_ip>` and `<name>` are yours. **Replace the whole placeholder including the angle
brackets**: a stray `<` in a shell command is read as input redirection and fails with a confusing
`No such file or directory`. The login IP is in your account email, and you must be on NTU VPN / NTUSECURE.
`/projects/<name>` is the project directory you create with `storagemgr` (pick **SSD**); `ls /projects` on the
login node shows its exact spelling. It may resolve to `/projects/_ssd/<name>` in logs, which is the same folder.

## Layout on the cluster

```
/projects/<name>/
  code/yolo/          <- this folder (submit every job from here)
  envs/yolo/          conda env (on SSD: thousands of small files)
  data/               catsdogs (train/val/test), cifar10, cifar10_imb, cifar10_imb_os
  weights/            pretrained *.pt
  runs/<experiment>/  weights/best.pt, results.csv, result.json, *_predictions.csv
  .tmp/ .config/      scratch + caches (guideline section 9)
```

## 0. Check your limits first (login node, cheap)

```bash
storagemgr                                                    # make sure the project dir has room: ~10 GB env + ~2 GB data + runs
sacctmgr show assoc where user=$USER format=Account,QOS,DefaultQOS
sacctmgr show qos -P format=Name,MaxTRESPerUser               # concurrent-GPU caps per QoS
```

Default QoS allows **one job at a time** (batch or interactive). That is why sweeps run *inside* one job
(`run_suite.py`) and every experiment is resumable.

## 1. Upload code and data (from your laptop, in the repo root)

```bash
# rsync only creates the last folder of a path, so create the parents first
ssh <user>@<login_ip> 'mkdir -p /projects/<name>/code/yolo /projects/<name>/data'
# code
rsync -av --exclude __pycache__ src/yolo/ <user>@<login_ip>:/projects/<name>/code/yolo/
# Dogs-vs-Cats data: the extracted folder from the brief (train/ val/ test/), ~1.1 GB, 25.5k files
rsync -a --progress datasets/ <user>@<login_ip>:/projects/<name>/data/catsdogs_raw/
```

- macOS's built-in rsync is old and rejects newer flags such as `--info=progress2`; `--progress` works.
- **Make sure the data upload finished.** If it is interrupted (VPN drop, sleeping laptop) the folders exist but
  are only partly filled, and everything downstream looks fine until training. Just re-run the same rsync line:
  it resumes and sends only what is missing. Keep the laptop awake until it returns to the prompt.
- Once step 3 has run, the data has been *moved* out of `catsdogs_raw`, which prep then ignores. To add missing
  files after that, upload straight into the final folders, e.g.
  `rsync -a --progress datasets/val/ <user>@<login_ip>:/projects/<name>/data/catsdogs/val/`.

## 2. One-time setup (login node)

```bash
cd /projects/<name>/code/yolo
bash slurm/setup_env.sh /projects/<name> 2>&1 | tee setup.log   # conda env + pip installs, writes slurm/project.env
tail -3 setup.log                                                # must end with "SETUP OK" and a torch/ultralytics/cv2 line
module load Miniforge3 && source activate /projects/<name>/envs/yolo
python download_weights.py                          # ~130 MB of weights (compute nodes may lack internet)
python prep_cifar10.py --download-only              # ~145 MB CIFAR-10 archive (only needed for the CIFAR groups)
```

`setup_env.sh` installs through the env's own interpreter and can be re-run safely. Every job script likewise
calls `/projects/<name>/envs/yolo/bin/python` by full path and prints `python: <path> | ultralytics <version>` as
the first line of its log, so you can see which interpreter ran.

## 3. Arrange and check the data (CPU-only job, 0 SU)

```bash
sbatch slurm/prep_data.sbatch
squeue --me                                          # wait for it to finish, then: cat prep-*.out
```

`prep-*.out` must show `images found: {'train/cat': 10000, 'train/dog': 10000, 'val/cat': 2500, 'val/dog': 2500,
'test': 500}`. Prep stops with an error if any split is empty, and warns if the counts differ from these (fine
if you deliberately use a subset). It is safe to re-run.

## 4. Smoke test, then the experiments

```bash
# about 30 s of training: 1 epoch on 2% of the data. Confirms GPU, AMP, dataloader, scoring.
sbatch --time=00:20:00 slurm/run_suite.sbatch --group smoke
```

A good smoke test ends with `[done] cd_smoke val_acc=...` in `yolo-<jobid>.out` (the accuracy itself means
nothing at this size). These messages are harmless: the Polars "Missing required CPU features" warning in the
`.err` file (a dependency that our scripts never use), "Slow image access" (data is on network storage) and the
"new ultralytics version available" notice (the version is pinned on purpose).

```bash
# real runs (one job at a time; submit the next when the previous finishes)
sbatch slurm/run_suite.sbatch --group models        # yolo26 n/s/m/l/x + yolo11s, fine-tuned (6 trainings, one after another)
sbatch slurm/run_suite.sbatch --group methods       # zero-shot / linear probe / fine-tune / from scratch
sbatch slurm/run_suite.sbatch --group lr
sbatch slurm/run_suite.sbatch --group aug
sbatch slurm/run_suite.sbatch --group imgsz
sbatch slurm/run_suite.sbatch --group data
sbatch slurm/run_suite.sbatch --group cifar         # question (g)
sbatch slurm/run_suite.sbatch --group cifar_imb     # question (h)
# or everything in one go (raise --time; see note below):
sbatch --time=24:00:00 slurm/run_suite.sbatch --group all

# a single experiment instead of a group, e.g. only the yolo26 "l" model:
sbatch --time=02:00:00 slurm/run_suite.sbatch --only cd_l_ft

# once you know the best config: extra seeds -> mean +/- std over runs (question b)
sbatch slurm/run_suite.sbatch --seeds-of cd_s_ft
```

Slurm options (`--time`, `--gpus`) go **before** the script name; everything after it is passed to
`run_suite.py`. `--group`, `--only` and `--seeds-of` are alternatives: give exactly one.
`python run_suite.py --list` shows every group and experiment name. Groups overlap on purpose (e.g. `cd_s_ft`
is in `models`, `methods`, `lr`, ...); a finished experiment is never re-run.

**If a job hits its time limit or is preempted, submit the same command again**: finished experiments are
skipped and the interrupted one resumes from its last epoch. Shorter `--time` queues faster. Nothing has been
timed for the bigger models yet, so after the first job finishes check
`sacct -j <jobid> --format=JobID,Elapsed,State` and size later requests from it.

Change GPU with `--gpus=<model>:1` on the command line (`6000ada`, `a40`, `a6000`, `l40`, `pro6000`). The
default is `6000ada`, the GPU that the docs list for EEE users (verified: jobs land on `gpu-6000ada-*`);
`pro6000` costs 480 SU/GPU-hour vs 240 for `6000ada` and 180 for `a40`/`a6000`, and these small models do not
need it.

## 5. Use the trained model on your laptop (no cluster needed)

Nothing here touches the cluster, and inference is light. (Training locally is possible for the small `n`/`s`
models but is slow and loads the machine heavily; leave it to the cluster.)

**5a. Pull the trained run(s)** from the cluster. `best.pt` is the weights, the small JSON/CSV files carry the
config and metrics; the bigger `last.pt` (only used for resuming) is skipped:

```bash
mkdir -p ~/yolo_project/runs
rsync -av --exclude='last.pt' <user>@<login_ip>:/projects/<name>/runs/cd_l_ft ~/yolo_project/runs/   # one run
rsync -av --exclude='last.pt' <user>@<login_ip>:/projects/<name>/runs/ ~/yolo_project/runs/          # or all of them
```

**5b. One-time local setup.** Keep the work folder *outside* iCloud Drive (which would try to sync the data):

```bash
cd src/yolo
python3.12 -m venv ~/venvs/yolo && source ~/venvs/yolo/bin/activate     # python3.12: recent macOS pythons may lack torch wheels
pip install -r requirements.txt
export PROJECT_DIR=~/yolo_project                                       # add this line to ~/.zshrc

# Dogs-vs-Cats: only val (to score) and test (to predict) are needed, not train. -c = instant APFS clone
mkdir -p $PROJECT_DIR/data/catsdogs
cp -cR ../../datasets/val  $PROJECT_DIR/data/catsdogs/val
cp -cR ../../datasets/test $PROJECT_DIR/data/catsdogs/test
# CIFAR-10 runs only: downloads ~146 MB from GitHub and builds the folders (also builds the imbalanced variants, unneeded but harmless)
python prep_cifar10.py
```

**5c. Use it.** These read the config (image size, class names) stored next to each run's weights:

```bash
python evaluate.py --run cd_l_ft --split val           # accuracy, per-class recall/precision, macro-F1 on the 5,000 val images
python predict.py --run cd_l_ft                        # $PROJECT_DIR/submission.csv  (id,label; 1=dog 0=cat) + submission_probs.csv
python predict.py --run cd_l_ft --run cd_m_ft --out $PROJECT_DIR/submission_ens.csv   # optional: average several models
python analysis.py errors --run cd_l_ft                # most confident val mistakes -> runs/cd_l_ft/errors_val.png (question e)
python analysis.py test --probs $PROJECT_DIR/submission_probs.csv       # least/most confident test images -> PNGs
python aggregate.py                                    # runs/summary.csv, runs/curves/*.png, table of every pulled run
python evaluate.py --run c10_s_ft --split test         # CIFAR-10: balanced official test split
```

**`evaluate.py --split test` only works for datasets whose test split has labels** (CIFAR-10). Dogs-vs-Cats'
`data/catsdogs/test/` is the 500 unlabelled images from the brief, laid out flat with no class subfolders
(that's what `predict.py` reads to write `submission.csv`) — `evaluate.py --run cd_l_ft --split test` refuses
with a clear message pointing at `predict.py` instead of guessing at accuracy against non-existent labels.
`DATASET_EVAL` in `experiments.py` is what decides this per dataset.

`evaluate.py` and `predict.py` need only `weights/best.pt`: if `result.json` did not come across they recover
the image size and dataset from the training args stored inside the checkpoint and print a `NOTE` saying so.
`aggregate.py` does need `result.json` (see Troubleshooting if it is missing).

Sanity check: the accuracy `evaluate.py` prints should match the one recorded on the cluster (last digit may
differ across GPU/CPU):

```bash
python -c "import json,os; print(json.load(open(os.path.expandvars('$PROJECT_DIR/runs/cd_l_ft/result.json')))['val_accuracy'])"
```

Inference uses the CPU unless CUDA exists. To use the Apple GPU, prefix a command with `YOLO_DEVICE=mps` (works;
opt-in only). Scoring the 5,000 val images takes longer than predicting the 500 test images, more so for the
bigger models.

**Memory.** `evaluate.py` and `predict.py` take `--batch N` (default 32; lower = less memory, predictions are
identical). Peak memory of the Python process for CPU inference at 224 px, measured on a 16 GB MacBook Air
(roughly +/- 0.2 GB between runs, and it does not grow with the number of images):

| model | `--batch 32` (default) | `--batch 128` |
|---|---|---|
| `yolo26n-cls` | 0.6 GB | 1.3 GB |
| `yolo26s-cls` | 0.8 GB | 2.0 GB |
| `yolo26m-cls` | 1.2 GB | 3.4 GB |
| `yolo26l-cls` | 1.2 GB | over 3.5 GB (not measured further) |
| `yolo26x-cls` | 1.7 GB | over 3.5 GB (not measured further) |

Importing Ultralytics costs about 0.2 GB of that. Use `--batch 8` if the machine is short on memory. These are CPU
numbers; memory with `YOLO_DEVICE=mps` was not measured.

**5d. Copy the final numbers into the repo** (from `src/yolo`; `*.pt` files are gitignored on purpose):

```bash
mkdir -p ../../results/yolo && cp $PROJECT_DIR/submission.csv $PROJECT_DIR/runs/summary.csv ../../results/yolo/
```

## What each script does

| file | purpose |
|---|---|
| `experiments.py` | registry of all experiments and groups (edit here to add/change runs) |
| `train.py` | run / resume / skip one experiment, then score it (`result.json`) |
| `run_suite.py` | run a group sequentially inside one Slurm job |
| `yolo_ext.py` | true no-augmentation, rotation, class-weighted and focal loss on top of Ultralytics |
| `zero_shot.py` | untrained ImageNet-1k model, dog/cat classes summed |
| `evaluate.py` | metrics + per-image predictions with the same inference path as the test set |
| `predict.py` | `submission.csv` (+ per-image probabilities) |
| `analysis.py` | contact sheets of errors / least-confident test images |
| `aggregate.py` | one table + validation curves over all runs |
| `prep_catsdogs.py`, `prep_cifar10.py` | data arranging and checks; CIFAR-10 val split and imbalanced variants |
| `download_weights.py` | pretrained weights into `$PROJECT_DIR/weights` |
| `slurm/setup_env.sh` | one-time conda env + pip installs (login node) |
| `slurm/cluster_env.sh` | scratch/cache paths on project storage, sourced by every job |
| `slurm/prep_data.sbatch`, `slurm/run_suite.sbatch` | CPU-only data prep job; GPU training job |
| `slurm/cpu_job.sbatch` | optional CPU-only wrapper to run a post-processing script on the cluster instead of locally |

## Experiment design (report mapping)

- **Recipe** (all fine-tunes): AdamW, lr 1e-3, cosine schedule to 1% , 1 warm-up epoch, batch 64, 15 epochs,
  224 px, cross-entropy, Ultralytics default augmentation (random resized crop, flip, RandAugment, random
  erasing), seed 0, `deterministic=True`. `best.pt` is the epoch with best *validation* accuracy; the val set
  is used for nothing else and the optimiser only ever sees `train/`.
- **(a)** `aug` group (none / basic / strong / strong + rotation) and `data` group (1k-20k training images).
- **(b)** architecture printed at the start of each run's log; `--seeds-of` gives mean +/- std.
- **(c)** `lr` and `imgsz` groups.
- **(d)** `result.json` -> `val_accuracy`; `predict.py` -> `submission.csv`.
- **(e)** `analysis.py`.
- **(f)** `models`, `methods`, `aug`, `imgsz`, `data` groups.
- **(g)** `cifar`: 45k train / 5k val (carved from train) / official 10k test; images upscaled from 32 px to
  64/128 px. Test accuracy is in `result.json` -> `splits.test`.
- **(h)** `cifar_imb`: `cat`, `deer`, `truck` cut to 5% (225 vs 4500 images). Baseline vs oversampling vs
  class-weighted CE (inverse frequency, effective number) vs focal loss; compare per-class recall and macro-F1
  on the balanced test split. Change the imbalance with `prep_cifar10.py --minority ... --keep ...`.

## Troubleshooting

- **`No module named 'PIL'` / `'ultralytics'` / `'cv2'`**: the Python that ran is not the env, or the env is
  incomplete. Check with
  `/projects/<name>/envs/yolo/bin/python -c "import PIL, cv2, ultralytics; print('ok')"`. If that fails, re-run
  `bash slurm/setup_env.sh /projects/<name> 2>&1 | tee setup.log` (safe to repeat) and look for `SETUP OK`.
  `opencv-python` and `opencv-python-headless` share one `cv2/` folder, so the script removes both and reinstalls
  headless; removing only one leaves `cv2` half-deleted.
- **`No images found in 'val' split`, or `found N images in 1 classes (requires 2 classes)`** (in `yolo-*.out`):
  the data upload was incomplete. Check
  `for d in train/cat train/dog val/cat val/dog; do echo $d $(ls /projects/<name>/data/catsdogs/$d | wc -l); done`
  (expect 10000, 10000, 2500, 2500; 500 in `catsdogs/test`), finish the upload as described in step 1, then
  `rm /projects/<name>/data/catsdogs/*.cache` and re-run `sbatch slurm/prep_data.sbatch`.
- **A pulled run has `results.csv` and `weights/best.pt` but no `result.json`** (also no `val_predictions.csv`):
  training finished but the scoring step at the end of the job did not complete, so the run is missing from
  `aggregate.py`. On the cluster look at the end of the job's logs (`tail -40 yolo-<jobid>.err yolo-<jobid>.out`;
  `run_suite.py` prints a traceback and `FAILED: <name>`). Two causes seen so far, both in the scoring step, not
  training: `oom_kill` (host RAM) or `torch.OutOfMemoryError: CUDA out of memory` (GPU RAM, e.g. failing inside
  `model.warmup`) — earlier versions of `evaluate.py` made one `predict()` call over the *entire* image list,
  whose memory grew with the number of images processed within that call (measured on CPU: over 3.5 GB for 256
  images with the `l` model, growing further as more images were processed) instead of staying flat; it now
  predicts one batch at a time. `train.py`'s scoring step also now clears the CUDA cache and scores at the same
  batch size training itself already ran at, instead of a fixed default. **Re-sync the code first** (step 1),
  then submit the **same command** again, e.g. `sbatch --time=00:30:00 slurm/run_suite.sbatch --only cd_l_ft`. It
  sees that training is complete and only re-scores (about a minute, no retraining), then pull the run again.
  Locally, `evaluate.py` and `predict.py` work without `result.json`.
- **`mkdir ... failed: No such file or directory` from rsync**: the parent folder is missing (`ssh ... mkdir -p`
  first) or the project name is misspelled; also see the note about angle brackets at the top.
- **`torch.cuda.is_available()` is False in a GPU job** (`run_suite.py --require-gpu` aborts immediately): the
  pip wheel's CUDA build is newer than the node driver. On a GPU node check `nvidia-smi` for the driver's CUDA
  version and reinstall a matching build, e.g.
  `pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126`.
  (On the 6000ada nodes the default `cu130` wheel worked.)
- **`Disk quota exceeded`**: check `du -sh /tmp $HOME /projects/<name>/*`; scratch should already be redirected.
- **CIFAR-10 download is slow**: `prep_cifar10.py` fetches from GitHub by default. The official host was
  throttled to ~1 KB/s when this was written; `--source torchvision` uses it, or place
  `cifar-10-python.tar.gz` in `data/cifar10_raw/` yourself first.
- **Laptop gets sluggish** when running something on the Apple GPU: stop it and use the CPU (unset
  `YOLO_DEVICE`); the cluster is the place for training.
- **Need more than one job at a time**: `--qos=override-limits-but-killable` (idle GPUs only, may be requeued,
  free against your monthly cap, not available to every account). Resumable runs make this safe.
- Never leave a shell open on a GPU node and never edit files while a job is pending.
