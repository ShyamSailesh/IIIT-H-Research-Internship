from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image_path = r"C:\Users\shyam\yoloproject\zidane.jpg"
results = model.predict(source=image_path, show=True, save=True,
                        project=r"C:\Users\shyam\yoloproject\YOLO_Outputs",
                        name="Zidane_sel")


# Load the segmentation model
model = YOLO("yolov8n-seg.pt")  # segmentation version of YOLOv8

# Path to your local image
image_path = r"C:\Users\shyam\yoloproject\zidane.jpg"

# Run segmentation
# - classes=[0] → only 'person' class (COCO dataset: 0 = person)
# - save=True → save output
# - project & name → save to a known folder
results = model.predict(
    source=image_path,
    show=True,
    save=True,
    classes=[0],  # only detect persons
    project=r"C:\Users\shyam\yoloproject\YOLO_Outputs",
    name="zidane_Segmentation"
)
