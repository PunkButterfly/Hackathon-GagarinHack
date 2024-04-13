from datetime import datetime as dt

from models.Detector import Detector
from models.Recogniter import Recogniter

from PIL import Image


class Pipeline:
    def __init__(self, path_to_weights, path_to_tmp, detector_weights_name: str = 'detector.pt'):
        self.detector = Detector(path_to_weights, path_to_tmp, weights_name=detector_weights_name)
        self.recognitor = Recogniter()

    def forward(self, path_to_image: str):
        image = Image.open(path_to_image, 'r').convert("RGB")

        print(path_to_image)

        predict_result = self.detector.predict([path_to_image])[0]

        features_coordinates = predict_result['coords']
        img_path = predict_result['predict_img_path']

        recognited_text = []
        for coords in features_coordinates:
            cropped_image = image_crop(image, coords)
            text = self.recognitor.predict(cropped_image)

            recognited_text.append(text)

        return recognited_text, img_path
    
def image_crop(image, coords):
    cropped_image = image.crop((coords[0], coords[1], coords[2], coords[3]))
    rotated_image = cropped_image.transpose(Image.ROTATE_90) if (coords[2] - coords[0]) < (coords[3] - coords[1]) else cropped_image

    return rotated_image