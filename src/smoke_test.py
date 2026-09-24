"""Verify CUDA and one real YOLO11n training epoch on the official coco8 sample."""

from pathlib import Path
import sys

import torch
from ultralytics import YOLO, __version__ as ultralytics_version, settings


ROOT = Path(__file__).resolve().parents[1]
settings.update({"datasets_dir": str(ROOT / "data"), "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs")})

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}; CUDA wheel: {torch.version.cuda}")
print(f"Ultralytics: {ultralytics_version}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch does not detect the NVIDIA GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda:0")
x = torch.randn(512, 512, device=device, requires_grad=True)
loss = x.square().mean()
loss.backward()
torch.cuda.synchronize()
print(f"CUDA forward/backward: OK; loss={loss.item():.4f}")

model = YOLO("yolo11n.pt")
model.train(
    data="coco8.yaml",
    epochs=1,
    imgsz=320,
    batch=2,
    device=0,
    workers=0,
    amp=True,
    cache=False,
    project=str(ROOT / "runs" / "smoke"),
    name="yolo11n_coco8",
    exist_ok=True,
)
print("YOLO11n GPU training smoke test: OK")

