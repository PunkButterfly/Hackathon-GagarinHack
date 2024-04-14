from datetime import datetime as dt

from models.Detector import Detector
from models.Recognitor import Recognitor
from models.Alignmentor import Alignmentor
from models.classifier import Classifier

from PIL import Image


class Pipeline:
    def __init__(
            self,
            path_to_weights,
            path_to_tmp,
            detector_weights_name: str = 'detector.pt',
            classifier_weights_name: str = 'v3_weights.pt',
            aligment_wetghts_name: str = "alignmentor.pt"
        ):
        self.detector = Detector(path_to_weights, path_to_tmp, weights_name=detector_weights_name)
        self.recognitor = Recognitor()
        self.aligmentor = Alignmentor(path_to_weights, path_to_tmp, weights_name=aligment_wetghts_name)
        self.classifier_model = Classifier(path_to_weights, weights_name=classifier_weights_name )

    def forward(self, path_to_image: str):
        # image = Image.open(path_to_image, 'r').convert("RGB")

        alignment_image = self.aligmentor.align(path_to_image)

        classifier_probs = self.classifier_model.process_img(alignment_image)

        content_predict_result = self.detector.predict([path_to_image])[0]

        features_coordinates = content_predict_result['coords']
        predict_img_path = content_predict_result['predict_img_path']
        names = content_predict_result['names']
        confs = content_predict_result['confs']

        recognited_text = []
        for name, conf, coords in zip(names, confs, features_coordinates):
            cropped_image = image_crop(alignment_image, coords)
            text = self.recognitor.predict(cropped_image)

            recognited_text.append((text, name, conf))

        return classifier_probs, recognited_text, predict_img_path
    
def image_crop(image, coords):
    cropped_image = image.crop((coords[0], coords[1], coords[2], coords[3]))
    rotated_image = cropped_image.transpose(Image.ROTATE_90) if (coords[2] - coords[0]) < (coords[3] - coords[1]) else cropped_image

    return rotated_image