from ultralytics import YOLO
import os
from datetime import datetime

class Detector:
    def __init__(self, path_to_weights: str, path_to_tmp: str, weights_name: str = 'detector.pt', confidence_level: float = 0.25):
        self.model = YOLO(os.path.join(path_to_weights, weights_name))
        self.model.conf = confidence_level
        self.tmp_path = path_to_tmp

    def predict(self, paths_to_images: list):
        print("Detection...")
        outputs = self.model(paths_to_images)

        output = outputs[0] # можно так делать потому что всегоа загружаем только одно изображение
         
        print(output)

        print(output.boxes)
        
        names = [output.names[i] for i in output.boxes.cls.numpy()]
        confs = output.boxes.conf.numpy().tolist()

        preds = [
            {
                "predict_img_path": output.save(filename=f'{self.tmp_path}tmp_{datetime.now()}.jpg'),
                "names": names,
                "confs": confs,
                "coords": output.boxes.xyxy.numpy()
            }
        ]

        return preds

        preds = [
            {
                "predict_img_path": output.save(filename=f'{self.tmp_path}tmp_{datetime.now()}.jpg'),
                "predict_data": dict(zip(map(output.names, output.boxes.cls.numpy()), output.boxes.xyxy.numpy()))
            } for output in outputs]

        return preds
