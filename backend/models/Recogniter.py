from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class Recogniter:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('raxtemur/trocr-base-ru')
        self.model = VisionEncoderDecoderModel.from_pretrained('raxtemur/trocr-base-ru')

    def predict(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]

        return text
