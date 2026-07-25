import subprocess
import glob
import os
import shutil

# ----------------------------
# 🔹 USER SETTINGS
# ----------------------------

# Absolute path to darknet folder
DARKNET_DIR = "/home/patih-fauzi/Kerja/YOLO_AI/97_YOLOv2/darknet_train"

# Image input folder (CHANGE THIS)
INPUT_DIR = "/home/patih-fauzi/Kerja/YOLO_AI/97_YOLOv2/darknet_train/images"

# Output folder (CHANGE THIS)
OUTPUT_DIR = "/home/patih-fauzi/Kerja/YOLO_AI/97_YOLOv2/darknet_train/outputs_images"

# Model files
CFG = os.path.join(DARKNET_DIR, "cfg/yolov2-tiny.cfg")
WEIGHTS = os.path.join(DARKNET_DIR, "yolov2-tiny.weights")
DATA = os.path.join(DARKNET_DIR, "cfg/coco.data")
DARKNET = os.path.join(DARKNET_DIR, "darknet")

# ----------------------------
# Setup
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

images = glob.glob(os.path.join(INPUT_DIR, "*.*"))

print(f"Found {len(images)} images")

# ----------------------------
# Run Detection
# ----------------------------

for img in images:
    print(f"Processing: {img}")

    cmd = [
        DARKNET,
        "detector", "test",
        DATA,
        CFG,
        WEIGHTS,
        img,
        "-dont_show"
    ]

    # IMPORTANT: run inside darknet folder
    subprocess.run(cmd, cwd=DARKNET_DIR)

    pred_path = os.path.join(DARKNET_DIR, "predictions.jpg")

    if os.path.exists(pred_path):
        output_path = os.path.join(
            OUTPUT_DIR,
            os.path.basename(img)
        )

        shutil.move(pred_path, output_path)
        print(f"✅ Saved to {output_path}")
    else:
        print("❌ Detection failed")

print("🎉 All images processed!")
