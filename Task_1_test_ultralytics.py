from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image_path = r"C:\Users\shyam\yoloproject\Task_1\Task_1_img.jpg"
results = model.predict(source=image_path, show=True, save=True,
                        project=r"C:\Users\shyam\yoloproject\Task_1",
                        name="Task_1_Detection_result")


# Load the segmentation model
model = YOLO("yolov8n-seg.pt") 

# Path to your local image
image_path = r"C:\Users\shyam\yoloproject\Task_1\Task_1_img.jpg"

# Run segmentation
# - classes=[0] → only 'person' class (COCO dataset: 0 = person)
# - save=True → save output
# - project & name → save to a known folder
results = model.predict(
    source=image_path,
    show=True,
    save=True,
    classes=[0],  # only detect persons
    project=r"C:\Users\shyam\yoloproject\Task_1",
    name="Task_1_Segmentation_result"
)
