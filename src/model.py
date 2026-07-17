import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

NUM_CLASSES = 2
LEARNING_RATE = 1e-4
DROPOUT_P = 0.1
RESNET_OUT_DIM = 2048

losses = []

print("CUDA Available:", torch.cuda.is_available())


class JointTextImageModel(nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        text_module,
        image_module,
        text_feature_dim,
        image_feature_dim,
        fusion_output_size,
        dropout_p,
        hidden_size=512,
    ):
        super().__init__()

        self.text_module = text_module
        self.image_module = image_module

        self.fusion = nn.Linear(
            text_feature_dim + image_feature_dim,
            fusion_output_size,
        )

        self.fc1 = nn.Linear(fusion_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.dropout = nn.Dropout(dropout_p)
        self.loss_fn = loss_fn

    def forward(self, text, image, label):

        text_features = torch.relu(self.text_module(text))
        image_features = torch.relu(self.image_module(image))

        combined = torch.cat((text_features, image_features), dim=1)

        fused = self.dropout(torch.relu(self.fusion(combined)))

        hidden = torch.relu(self.fc1(fused))

        logits = self.fc2(hidden)

        loss = self.loss_fn(logits, label)

        return logits, loss


class MultimodalFakeNewsDetectionModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()

        if hparams:
            self.hparams.update(hparams)

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get(
            "image_feature_dim",
            self.text_feature_dim,
        )

        self.model = self._build_model()

    def forward(self, text, image, label):
        return self.model(text, image, label)

    def training_step(self, batch, batch_idx):

        text = batch["text"]
        image = batch["image"]
        label = batch["label"]

        pred, loss = self.model(text, image, label)

        pred_label = torch.argmax(pred, dim=1)
        accuracy = (pred_label == label).float().mean()

        self.log(
         "train_loss",
         loss,
         prog_bar=True,
         on_step=False,
         on_epoch=True,
        )

        self.log(
            "train_acc",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step_end(self, batch_parts):
        return sum(batch_parts) / len(batch_parts)
    

    def validation_step(self, batch, batch_idx):

        text = batch["text"]
        image = batch["image"]
        label = batch["label"]

        pred, loss = self.model(text, image, label)

        pred_label = torch.argmax(pred, dim=1)
        accuracy = (pred_label == label).float().mean()

        self.log(
            "val_loss",
             loss,
             prog_bar=True,
             on_epoch=True,
        )

        self.log(
            "val_acc",
             accuracy,
             prog_bar=True,
             on_epoch=True,
            )

        return loss
    
    def test_step(self, batch, batch_idx):

        text = batch["text"]
        image = batch["image"]
        label = batch["label"]

        pred, loss = self.model(text, image, label)

        pred_label = torch.argmax(pred, dim=1)

        accuracy = (pred_label == label).float().mean()

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
        )

        self.log(
            "test_acc",
            accuracy,
            prog_bar=True,
            on_epoch=True,
        )

        return loss

    

    def configure_optimizers(self):
        return torch.optim.Adam(
           filter(lambda p: p.requires_grad, self.parameters()),
            lr=LEARNING_RATE,
        )

    def _build_model(self):

        text_module = nn.Linear(
            self.embedding_dim,
            self.text_feature_dim,
        )

        image_module = torchvision.models.resnet152(
          weights=torchvision.models.ResNet152_Weights.DEFAULT
        )

        # Freeze all pretrained layers
        for param in image_module.parameters():
            param.requires_grad = False

        # Replace the classifier
        image_module.fc = nn.Linear(
            RESNET_OUT_DIM,
            self.image_feature_dim,
        )

        # Train only the new classifier
        for param in image_module.fc.parameters():
            param.requires_grad = True

        return JointTextImageModel(
            num_classes=self.hparams.get(
                "num_classes",
                NUM_CLASSES,
            ),
            loss_fn=nn.CrossEntropyLoss(),
            text_module=text_module,
            image_module=image_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            fusion_output_size=self.hparams.get(
                "fusion_output_size",
                512,
            ),
            dropout_p=self.hparams.get(
                "dropout_p",
                DROPOUT_P,
            ),
        )


class PrintCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Training Started...")

    def on_train_end(self, trainer, pl_module):
        print("Training Finished.")

        for loss in losses:
            print(loss)