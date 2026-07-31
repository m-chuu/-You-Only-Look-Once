import os
from ultralytics import YOLO

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(script_dir, "best.pt"))   # same weights that produced detection_result-3
results = model(os.path.join(script_dir, "DetectTheseImage"),   # input image directory
                imgsz=1280,   # birds here are small; 416 loses them when the image is downscaled
                conf=0.25,    # Ultralytics default, matching detection_result-3
                save=True, project=os.path.join(script_dir, "runs/detect/Inference"),
                name="DetectTheseImage_result")   # output image directory

print("Inference completed")
