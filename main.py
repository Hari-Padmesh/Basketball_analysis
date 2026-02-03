from ultralytics import YOLO
import os

model = YOLO("yolov8m.pt")  
result=model.predict("input_videos/video_1.mp4",save=True, project="output_videos",name="video_1_results") 
print(result)
print("============")
for box in result[0].boxes:
    print(box)
