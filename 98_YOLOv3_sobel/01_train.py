from ultralytics import YOLO

# -------------------------------------------------
# Load YOLO model architecture
# choose to run model 1 : YOLOv3-tinyu or model 2: YOLOv8 Nano 


# -----------------Model 1--------------------------------
model = YOLO("01_yolov3-tinyu.pt") # Model 1: YOLOv3-tinyu (lightweight, fast model)


# -------------------------------------------------
# Train the model
# -------------------------------------------------
model.train(
    data="01_data.yaml",     # Path to dataset configuration file
    epochs=100,            # Number of training epochs
    imgsz=416,            # Input image size (640x640)
    batch=8,             # Batch size per iteration
    conf=0.01,
    name="result_yolov3-tinyu"   # Folder name for saving results - change to own name
)
