from datetime import datetime as dt

from Detector import Detector
from Recogniter import Recogniter

from PIL import Image


class Pipeline:
    def __init__(self):
        self.detector = Detector()
        self.recognitor = Recogniter()

    def forward(self, path_to_image: str):
        image = Image.open(path_to_image, 'r').convert("RGB")
        features_coordinates = self.detector.predict([path_to_image])[0]

        recognited_text = []
        for coords in features_coordinates:
            cropped_image = image.crop((coords[0], coords[1], coords[2], coords[3]))
            text = self.recognitor.predict(cropped_image)

            recognited_text.append(text)

        return recognited_text

init_start = dt.now()
pipeline = Pipeline()
print(pipeline.forward("dl.jpeg"))
print(f"init time: {dt.now() - init_start}")
start = dt.now()
print(pipeline.forward("dl.jpeg"))
print(f"iter: {dt.now() - start}")
