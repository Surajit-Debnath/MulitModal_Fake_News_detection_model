import pandas as pd
from torch.utils.data import DataLoader
from src.image_dataset import ImageOnlyFakeNewsDataset
from src.transforms import efficientnet_train_transform, efficientnet_eval_transform


def create_image_dataloader(
    csv_path,
    images_dir="./images",
    batch_size=32,
    shuffle=True,
    num_workers=0,
    train=True,
):

    dataframe = pd.read_csv(csv_path)
    transform = efficientnet_train_transform if train else efficientnet_eval_transform

    dataset = ImageOnlyFakeNewsDataset(
        dataframe=dataframe,
        images_dir=images_dir,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader
