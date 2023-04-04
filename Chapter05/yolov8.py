from ultralytics import YOLO

# Load a YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")

# Detect objects on a Wikipedia image
results = model.predict('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/2011_FIA_GT1_Silverstone_2.jpg/1024px-2011_FIA_GT1_Silverstone_2.jpg')

# convert results->numpy_array->Image and display it
from PIL import Image

im = Image.fromarray(results[0].plot()).show()
