import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
    ):
      
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode(self, text):
       

        embeddings = self.model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        return embeddings