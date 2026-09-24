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

The 1%-data pilot's mAP is not meaningful for model comparison because it uses only one epoch. A formal long baseline has not yet been trained.

## Full-data 10-epoch pilot

The full-data 10-epoch YOLO11n pilot completed on 2026-09-24 at 640-pixel input, batch 4, and seed 0. Output: `runs/baseline/yolo11n_640_seed0-4/`. Training was interrupted after epoch 8 by a Windows DataLoader worker failure, then resumed from `last.pt` with `workers=0`. The final validation output on 548 images and 38,759 boxes was mAP50 **0.2353**, mAP50-95 **0.1291**, precision **0.3464**, and recall **0.2660**. Both training and validation losses were still decreasing at epoch 10. This is a pipeline pilot, not a final fully trained baseline.

The independent COCO-style size evaluation of `best.pt` is in `runs/evaluation/yolo11n_640_seed0-4_val/summary.json`. It reconstructs box areas in original image pixels from the converted labels and uses up to 902 detections per image. Of 38,759 validation boxes, 26,586 (68.6%) are small under the COCO <32 x 32 pixel definition. AP-small is **0.0454**, AP-medium **0.1852**, and AP-large **0.3692**. Its overall COCO AP is **0.1138** and AP50 **0.2015**; these need not equal the built-in Ultralytics mAP because the metric implementations differ. The figures strongly motivate testing a small-object-specific modification, but they do not prove P2 will help.

## Next experiment

Benchmark stable batch size and DataLoader worker settings, then freeze the configuration and train a formal longer baseline. The `workers=0` resume was stable but much slower than the first eight epochs with workers. AP-small and speed must be measured with the same protocol for every ablation. The initial 10-epoch pilot command was:

```powershell
& .\.venv\Scripts\python.exe .\src\train_baseline.py --epochs 10 --batch 4
```

Do not use the 10-epoch pilot as the final comparison against P2 or CBAM.
