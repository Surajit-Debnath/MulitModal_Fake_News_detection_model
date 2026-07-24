from PIL import Image

from app.embedder import TextEmbedder
from app.transforms import image_transform


class Predictor:
    def __init__(self):
        self.embedder = TextEmbedder()

    def preprocess_text(self, text: str):
        embedding = self.embedder.encode(text)

        # Make sure the tensor has shape (1, 768)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        return embedding

    def preprocess_image(self, image: Image.Image):
        image = image.convert("RGB")

        image = image_transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        return image