import os
from ultralytics import YOLO

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO("best.pt")   # get the best.pt from your  train result 
results = model("./01_Train_images/test/images/", #image file directory #input image
                save=True, project=os.path.join(script_dir, "runs/detect/Inference"), name="detection_result") #output image

print("Inference completed")


