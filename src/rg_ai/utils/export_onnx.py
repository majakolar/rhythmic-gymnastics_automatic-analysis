from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.export(format="onnx")  # creates 'yolo11n.onnx'

onnx_model = YOLO("yolo11n.onnx")