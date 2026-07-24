import torch

from PIL import Image

from app.embedder import TextEmbedder
from app.transforms import image_transform

from app.model_loader import ModelLoader


class FakeNewsPredictor:

    def __init__(self, checkpoint_path):

        self.loader = ModelLoader(checkpoint_path)

        self.model = self.loader.get_model()
        self.device = self.loader.get_device()

        self.embedder = TextEmbedder(device=str(self.device))

    @torch.no_grad()
    def predict(self, text: str, image: Image.Image):

        # ---------- Text ----------
        text_embedding = self.embedder.encode(text)

        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)

        text_embedding = text_embedding.to(self.device)

        # ---------- Image ----------
        image = image.convert("RGB")

        image_tensor = image_transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        # ---------- Model ----------
        logits = self.model(
            text_embedding,
            image_tensor,
        )

        probs = torch.softmax(logits, dim=1)

        confidence, prediction = torch.max(
            probs,
            dim=1,
        )

        labels = {
            0: "Real",
            1: "Fake",
        }

        return {
            "prediction": labels[prediction.item()],
            "confidence": float(confidence.item()),
        }