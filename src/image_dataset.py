import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageOnlyFakeNewsDataset(Dataset):

    def __init__(
        self,
        dataframe,
        images_dir,
        transform,
        label_column="2_way_label",
    ):

        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.label_column = label_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_path = os.path.join(
            self.images_dir,
            row["id"] + ".jpg"
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(
            row[self.label_column],
            dtype=torch.long
        )

        sample = {
            "id": row["id"],
            "image": image,
            "label": label,
        }

        return sample
