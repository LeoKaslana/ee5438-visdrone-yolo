# EE5438 VisDrone small object detection experiment

This project starts with a YOLO11n baseline on VisDrone2019-DET. Planned experiments compare the original P3/P4/P5 prediction hierarchy with a P2/P3/P4 variant and a localized CBAM variant; those improvements are not implemented yet. The official Ultralytics source is cloned under `external/ultralytics`. Keep project-specific model YAML files and modules under `configs/` and `src/`.

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
```

The official Ultralytics source is fixed to commit `25cda524aa10298b3ba8b9b287677b49fd236b7d` (2026-09-23), which reports package version 8.4.161. The project code, model configuration, and documentation are tracked here. Datasets, downloaded source dependencies, Python environments, checkpoints, and runs are ignored because they are large or machine specific.

The first setup machine used Windows, Python 3.11.16, PyTorch 2.11.0+cu128, and an RTX 5070 Laptop GPU (8 GB).

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

For a full baseline, use `--epochs 100`. Record the exact command, Git commit, package versions, seed, and hardware with each run.

For a short VisDrone pipeline test, use:

```powershell
& .\.venv\Scripts\python.exe .\src\train_baseline.py --epochs 1 --batch 2 --fraction 0.01 --workers 0
```

## Sources

See `references/SOURCES.md` for the official code, dataset, and research references.
