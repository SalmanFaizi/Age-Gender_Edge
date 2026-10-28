import functools
import torch
from ultralytics import YOLO

# Override the security default for trusted local weights execution
torch.load = functools.partial(torch.load, weights_only=False)

# The rest of your script executes smoothly
model = YOLO("yolo11n.pt")
export_path = model.export(format="imx", data="coco8.yaml")  
print(f"Model exported successfully: {export_path}")
