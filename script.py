from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="kitti.yaml", epochs=1)