from Detector import Detector
from Recogniter import Recognitor

from PIL import Image
from multiprocessing import Pool
from datetime import datetime as dt


class Pipeline:
    def __init__(self):
        self.detector = Detector()
        self.recognitor = Recognitor()

    def forward(self, path_to_image: str):
        image = Image.open(path_to_image, 'r').convert("RGB")
        features_coordinates = self.detector.predict([path_to_image])[0]

        with Pool() as pool:
            cropped_images = pool.starmap(image_crop, [(image, coords) for coords in features_coordinates])
            recognited_text = pool.map(self.recognitor.predict, cropped_images)

        return recognited_text


def image_crop(image, coords):
    cropped_image = image.crop((coords[0], coords[1], coords[2], coords[3]))
    rotated_image = cropped_image.transpose(Image.ROTATE_90) if (coords[2] - coords[0]) < (coords[3] - coords[1]) else cropped_image
    rotated_image.save(f"save_{coords[1]}.png")

    return rotated_image


if __name__ == '__main__':
    pipeline = Pipeline()
    start = dt.now()
    print(start)
    print(pipeline.forward("pass.jpeg"))
    print(f"iter: {dt.now() - start}")
