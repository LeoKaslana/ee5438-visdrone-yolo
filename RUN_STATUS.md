# Setup and smoke test status

Checked on 2026-09-24, Asia/Shanghai.

## Environment

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8,151 MiB reported by PyTorch
- NVIDIA driver: 573.01, maximum CUDA version reported by `nvidia-smi`: 12.8
- Python: 3.11.16, in `.venv/`
- PyTorch: 2.11.0+cu128
- Torchvision: 0.26.0+cu128
- Ultralytics: 8.4.161 from official GitHub clone at commit `25cda524aa10298b3ba8b9b287677b49fd236b7d`

## Checks completed

1. `torch.cuda.is_available()` returned `True` and identified the RTX 5070 Laptop GPU.
2. A CUDA tensor forward and backward operation succeeded.
3. YOLO11n completed one GPU training epoch and validation on official COCO8. Output: `runs/smoke/yolo11n_coco8/`.
4. Official Ultralytics VisDrone download and conversion completed. Train, val, and test-dev each have matching image and label counts: 6,471 / 548 / 1,610.
5. YOLO11n completed a VisDrone pilot at 640-pixel input, batch 2, one epoch, and `fraction=0.01` (65 training images) with full 548-image validation. Output: `runs/baseline/yolo11n_640_seed0/`.
6. The pilot wrote `best.pt` and `results.csv`. The training log reported about 0.633 GB peak GPU memory and 20.99 seconds for the run's epoch time. These values are not a full-dataset runtime estimate.

The pilot's mAP is not meaningful for model comparison because it uses only 1% of the training images and one epoch. A full baseline has not yet been trained.

## Next experiment

Run the 10-epoch full-data pilot first:

```powershell
& .\.venv\Scripts\python.exe .\src\train_baseline.py --epochs 10 --batch 4
```

Inspect memory, per-epoch time, and loss curves. Then choose final training epochs and batch size before starting the full baseline and model ablations.

