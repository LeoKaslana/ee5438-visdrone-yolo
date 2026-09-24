# Experiment record

The source code and this index are tracked in Git. Raw checkpoints, plots, predictions, and logs are local under `runs/` and are not uploaded to GitHub. Do not treat this index as a substitute for the raw run directories.

| Run | Train setting | Built-in mAP50-95 | COCO-style AP-small | Status and caveat |
| --- | --- | ---: | ---: | --- |
| `runs/baseline/yolo11n_640_seed0-4/` | 10 epochs, batch 4, seed 0; resumed after worker failure | 0.1291 | 0.0454 | Pipeline pilot only |
| `runs/baseline/formal_yolo11n_640_b16_seed0_live/` | 50 epochs, 640 px, batch 16, workers 2, seed 0 | 0.1673 | 0.0709 | Completed baseline |
| `runs/p2/formal_yolo11n_p2p3p4_640_b16_seed0/` | 50 epochs, 640 px, seed 0; batch 16/workers 2 for epochs 1-10, batch 8/workers 0 for epochs 11-50 | 0.1647 | 0.0793 | Completed recovery run; **not** a fixed-setting comparison |

The COCO-style AP-small values use `src/evaluate_small.py` on the 548-image VisDrone val split with areas in original image pixels and up to 902 detections per image. They are not official VisDrone challenge scores. The P2 recovery run was interrupted twice by Windows DataLoader worker exits, then completed from its epoch-10 checkpoint with safer settings. Its `args.yaml` now shows the final batch 8/worker 0 settings, not the full history; `RUN_STATUS.md` preserves the change.

## New recorded runs

For new work, use `scripts/train_strict_pair_visible.ps1` or `src/run_experiment.py`. Each run receives a unique ID, one concise result JSON under `runs/records/`, normal Ultralytics artifacts, and an AP-small summary. The terminal keeps its native progress bar; verbose console output is not saved. Add the final strict pair and its comparison here once both runs complete.
