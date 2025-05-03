from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weights_path='yolov8n.pt'):
        self.model = YOLO(weights_path)

    def detect(self, image):
        results = self.model(image, verbose=False)[0]
        return results.boxes.xyxy, results.boxes.cls, results.boxes.conf
