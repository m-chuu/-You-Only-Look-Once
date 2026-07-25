from ultralytics import YOLO

model = YOLO("best.pt")   # get the best.pt from your  train result 
results = model("./01_Train_images/test/images/", #image file directory #input image
                save=True, project="Inference", name="detection_result") #output image

print("Inference completed")


