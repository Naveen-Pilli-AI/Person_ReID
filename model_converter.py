from ultralytics import YOLO

model = YOLO('yolov8l.pt') 


model.export(format='onnx')