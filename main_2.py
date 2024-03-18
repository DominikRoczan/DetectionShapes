from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
# results = model.train(data='C:\\tmp\\dataset\\config.yaml', epochs=1)
root_config = 'C:\\Users\\domin\\OneDrive\\Pulpit\\Python\\DetectionShapes\\dataset\\config.yaml'

results = model.train(data=root_config, epochs=6)