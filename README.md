# EE5438 VisDrone small object detection experiment

This project compares a YOLO11n baseline on VisDrone2019-DET with a P2/P3/P4 prediction variant. CBAM remains a possible later ablation, not part of the P2 experiment. The official Ultralytics source is cloned under `external/ultralytics`. Project-specific model YAML files and scripts live under `configs/` and `src/`.

## Get started on a teammate's computer

Install Git and Python 3.11 first. On Windows PowerShell, clone this project and run these commands from its root directory. Use a CUDA 12.8 compatible NVIDIA driver; on a CPU-only computer, install the CPU PyTorch wheels and change `device=0` in the scripts to `device="cpu"`.

```powershell
git clone https://github.com/LeoKaslana/ee5438-visdrone-yolo.git
cd ee5438-visdrone-yolo
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/ultralytics/ultralytics.git external/ultralytics
git -C external/ultralytics checkout --detach 25cda524aa10298b3ba8b9b287677b49fd236b7d
& .\.venv\Scripts\python.exe -m pip install -e .\external\ultralytics
& .\.venv\Scripts\python.exe -m pip install pycocotools==2.0.11
```

The official Ultralytics source is fixed to commit `25cda524aa10298b3ba8b9b287677b49fd236b7d` (2026-09-23), which reports package version 8.4.161. The project code, model configuration, and documentation are tracked here. Datasets, downloaded source dependencies, Python environments, checkpoints, and runs are ignored because they are large or machine specific.

The first setup machine used Windows, Python 3.11.16, PyTorch 2.11.0+cu128, and an RTX 5070 Laptop GPU (8 GB).

## Recorded experiments and strict comparison

Use `src/run_recorded_experiment.py` for new training runs. It refuses an existing run ID and writes a separate `runs/records/<comparison-id>-<variant>/manifest.json` and `train.log` before/during training. The manifest records the exact command, Git revision and dirty state, software/GPU versions, input hashes, training settings, exit status, per-epoch metrics, checkpoint hashes, and AP-small evaluation result. Ultralytics saves `args.yaml`, `results.csv`, plots, `best.pt`, `last.pt`, and a periodic checkpoint every 10 epochs in the corresponding `runs/baseline/` or `runs/p2/` directory. Evaluation JSON and `evaluation.log` are saved separately. A failed run keeps its log and a failure status; it does not silently turn into a successful run.

For the matched 50-epoch baseline/P2 comparison on the first machine, launch the visible PowerShell window with:

```powershell
& .\scripts\train_strict_pair_visible.ps1
```

The script runs baseline first, then P2, sequentially on one GPU. Both use full VisDrone train/val, 640 pixels, physical batch 8, `workers=0`, seed 0, AMP, the same COCO `yolo11n.pt` initialization, and the same AP-small evaluator. Each gets a unique comparison ID and will not overwrite earlier runs. Keep the terminal open and prevent Windows from sleeping. If baseline fails, P2 is not started. A batch change or manual resume must be documented as a new experimental condition; do not present mixed-batch results as a strict comparison. Earlier experiments are summarized in `EXPERIMENTS.md`.

`runs/` remains ignored by Git because logs, data-derived JSON, and checkpoints can be large. Copy or back up `runs/records/`, `runs/evaluation/`, and the corresponding training directories if results must survive a disk failure or be shared with teammates; GitHub carries only the code and textual summaries.

## First checks

Run from PowerShell in the cloned project directory:

```powershell
& .\.venv\Scripts\python.exe .\src\smoke_test.py
```

The smoke test verifies CUDA tensor operations and trains YOLO11n for one epoch on the official `coco8` miniature dataset. It writes results to `runs/smoke/`.
The official `coco8` smoke test may create the small sample under `external/datasets/` because the upstream YAML uses a relative dataset path.

## VisDrone

The preparation script executes the download and conversion block from the official Ultralytics `VisDrone.yaml`, covering train (6,471 images), val (548), and test-dev (1,610). It generates `data/VisDrone/VisDrone-local.yaml` with the correct local path for each computer. First use downloads about 2 GB and needs more space during extraction.

```powershell
& .\.venv\Scripts\python.exe .\src\prepare_visdrone.py
```

The preparation script checks image and label counts and validates a sample of YOLO labels. The full baseline should be started only after the preparation script succeeds.

## Baseline training

Start with a short pilot and inspect GPU memory and time per epoch:

```powershell
& .\.venv\Scripts\python.exe .\src\train_baseline.py --epochs 10 --batch 4
```

For a matched baseline, use `--epochs 50 --batch 16 --workers 2 --seed 0`; keep these settings aligned with the P2 run where hardware permits. Record the exact command, Git commit, package versions, seed, and hardware with each run.

The completed formal baseline used 50 epochs, 640-pixel input, batch 16, two workers, and seed 0. Its run is `runs/baseline/formal_yolo11n_640_b16_seed0_live/` (checkpoints and images are intentionally not tracked in Git).

## P2/P3/P4 variant

`configs/yolo11n-p2p3p4.yaml` keeps the YOLO11n backbone but replaces the P5/32 detection output with P2/4, yielding detection strides 4/8/16. It has about 1.94 million parameters and 9.9 GFLOPs, versus about 2.59 million parameters and 6.5 GFLOPs for the 10-class baseline. The P2 feature map increases activation memory despite the lower parameter count. `src/train_p2.py` initializes matching layers from the same `yolo11n.pt` COCO checkpoint used by the baseline; newly shaped layers train from scratch.

On the first Windows machine, `scripts/train_p2_visible.ps1` runs a one-epoch 1%-data smoke test, then starts the 50-epoch full-data run in the same visible PowerShell window if the smoke test passes. The formal run is configured at 640 pixels, batch 16, two workers, and seed 0. A smoke run only checks the pipeline and memory; its accuracy is not meaningful. GPU memory approached the 8 GB hardware limit in the smoke test, so inspect the full run for CUDA out-of-memory errors.

```powershell
& .\scripts\train_p2_visible.ps1
```

The exact architecture and detection strides can be checked with `python -m unittest discover -s tests -v`. Once the full run finishes, evaluate its `best.pt` using the same `src/evaluate_small.py` settings as the baseline. Do not compare the 1%-data smoke score against the formal baseline.

If the full run is interrupted, `scripts/resume_p2_visible.ps1` resumes from that run's `last.pt` in a visible PowerShell window, restoring the checkpoint's original training settings. The checkpoint is not uploaded to GitHub; each machine resumes its own local run.

On the first machine, the same Windows DataLoader worker failure recurred during epoch 11 after a same-settings resume. `scripts/resume_p2_safe_visible.ps1` therefore resumes from the completed epoch 10 checkpoint with batch 8 and `workers=0`; all other checkpoint settings remain unchanged. The existing run directory still contains `b16` in its name because it began with batch 16. **This run mixes batch sizes and should be treated as a recovery/exploratory run, not a strictly matched architecture ablation.** For a final controlled comparison, train baseline and P2 from the same COCO checkpoint with one fixed, memory-safe batch configuration throughout, or explicitly report this limitation.

## Small-object evaluation

Ultralytics' ordinary validation output does not report the project's primary metric, COCO AP-small. After training, run:

```powershell
& .\.venv\Scripts\python.exe .\src\evaluate_small.py --weights .\runs\baseline\yolo11n_640_seed0\weights\best.pt --split val
```

The script converts the existing YOLO labels to COCO ground truth in **original image pixels**, runs predictions on the validation split, and writes `summary.json`, `ground_truth_coco.json`, and `predictions_coco.json` under `runs/evaluation/`. It reports AP-small (area below 32 x 32 pixels), AP-medium, AP-large, overall AP, AP50, AP75, and per-class AP. It allows up to 902 detections per image because VisDrone is dense. These are metrics against the Ultralytics-converted labels, not an official VisDrone challenge score. Keep test-dev unused until the model and settings are selected.

Replace the weights path if Ultralytics gave your run a suffix such as `-2`. The first setup machine's completed 10-epoch pilot is under `yolo11n_640_seed0-4`.

To recompute COCO metrics from the saved predictions without rerunning inference, add `--reuse-predictions`. The COCO-style AP numbers can differ from Ultralytics' built-in mAP because the matching and aggregation implementations differ; compare all variants with this same script.

For a short VisDrone pipeline test, use:

```powershell
& .\.venv\Scripts\python.exe .\src\train_baseline.py --epochs 1 --batch 2 --fraction 0.01 --workers 0
```

## Sources

See `references/SOURCES.md` for the official code, dataset, and research references.
