from ultralytics import YOLO


class Detector:
    def __init__(self, path_to_weights: str = 'weights/detector.pt', confidence_level: float = 0.25):
        self.model = YOLO(path_to_weights)
        self.model.conf = confidence_level

    def predict(self, paths_to_images: list):
        outputs = self.model(paths_to_images, task="detection")
        preds = [output.boxes.xyxy.numpy() for output in outputs]

        return preds
