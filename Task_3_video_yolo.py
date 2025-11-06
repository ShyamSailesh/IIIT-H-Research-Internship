from ultralytics import YOLO
import os

# CONFIGURATION
video_path = r"C:\Users\shyam\yoloproject\Task_3\Task_3_video_2.mp4"
output_dir = r"C:\Users\shyam\yoloproject\Task_3"

# 1️⃣ Object Detection on Video
print("🚀 Running YOLOv8 Object Detection on video...")

model_det = YOLO("yolov8n.pt")

results_det = model_det.predict(
    source=video_path,
    show=False,          
    save=True,
    project=output_dir,
    name="Video_Detection"
)

print("✅ Object Detection complete! Check:", os.path.join(output_dir, "Video_Detection"))

# 2️⃣ Person Segmentation on Video

print("🚀 Running YOLOv8 Segmentation (Persons only)...")

model_seg = YOLO("yolov8n-seg.pt")

results_seg = model_seg.predict(
    source=video_path,
    show=False,
    save=True,
    classes=[0],         # class 0 = 'person' (COCO dataset)
    project=output_dir,
    name="Video_Segmentation"
)

print("✅ Segmentation complete! Check:", os.path.join(output_dir, "Video_Segmentation"))

print("\n🎉 All tasks done! Your processed videos are saved inside YOLO_Outputs.")
