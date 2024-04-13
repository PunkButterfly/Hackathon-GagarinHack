import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class Recognitor:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed', is_parallelizable=True)
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

    def predict(self, image):
        with torch.no_grad():
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            ids = self.model.generate(pixel_values, do_sample=True, use_cache=True, max_new_tokens=100)
            text = self.processor.decode(ids[0], skip_special_tokens=True)

        return text
