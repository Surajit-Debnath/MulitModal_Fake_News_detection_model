import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import FakeNewsDataset


def create_dataloader(
    csv_path,
    embeddings,
    images_dir="./images",
    batch_size=32,
    shuffle=True,
    num_workers=0,
):

    dataframe = pd.read_csv(csv_path)

    dataset = FakeNewsDataset(
        dataframe=dataframe,
        images_dir=images_dir,
        embeddings=embeddings,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader