from ultralytics import YOLO


model = YOLO('yolov8s.yaml').load('yolov8s.pt')

result = model.train(data='/root/face_detection/face_det.yaml', epochs=100, imgsz=640, batch=8)

