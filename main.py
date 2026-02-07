from ultralytics import YOLO
import os



model = YOLO("models/ball_detector_model.pt")
source_path = "input_videos/video_1.mp4"
output_root = r"D:\Basketball_analysis\runs"
result = model.predict(
    source=source_path,
    save=True,
    project=output_root,
    name="basketball_output",
    exist_ok=True,
)

print(result)
print("============")
# for box in result[0].boxes:
#     print(box)
save_dir = result[0].save_dir
print(save_dir)

# Rename the output to a fixed filename inside the runs folder.
original_name = os.path.basename(source_path)
src_video = os.path.join(save_dir, original_name)
target_video = os.path.join(save_dir, "annotated_video.mp4")
if os.path.exists(src_video):
    if os.path.exists(target_video):
        os.remove(target_video)
    os.replace(src_video, target_video)
    print(f"Annotated video saved to: {target_video}")
else:
    print(f"Could not find output video at: {src_video}")
