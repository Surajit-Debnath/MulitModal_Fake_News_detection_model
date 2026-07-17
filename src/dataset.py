import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.transforms import image_transform

class FakeNewsDataset(Dataset):
  
    def __init__(
        self,
        dataframe,
        images_dir,
        embeddings,
        label_column="2_way_label",
    ):

        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.embeddings=embeddings
        self.label_column = label_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        text_embedding = self.embeddings[row["id"]]

        image_path = os.path.join(
            self.images_dir,
            row["id"] + ".jpg"
        )

        image = Image.open(image_path).convert("RGB")
        image = image_transform(image)

        label = torch.tensor(
            row[self.label_column],
            dtype=torch.long
        )

        sample = {
            "id": row["id"],
            "text": text_embedding,
            "image": image,
            "label": label,
        }

        return sample