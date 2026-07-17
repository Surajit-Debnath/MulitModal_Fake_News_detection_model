import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

NUM_CLASSES = 2
LEARNING_RATE = 1e-4
DROPOUT_P = 0.2
EFFICIENTNET_OUT_DIM = 1280


class ImageOnlyFakeNewsDetectionModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()

        if hparams:
            self.hparams.update(hparams)

        self.num_classes = self.hparams.get("num_classes", NUM_CLASSES)
        self.dropout_p = self.hparams.get("dropout_p", DROPOUT_P)
        self.learning_rate = self.hparams.get("learning_rate", LEARNING_RATE)

        self.loss_fn = nn.CrossEntropyLoss()
        self.backbone = self._build_backbone()

    def forward(self, image):
        return self.backbone(image)

    def extract_features(self, image):
        """Penultimate-layer (1280-d) image embedding, for use as the image side of fusion."""
        features = self.backbone.features(image)
        features = self.backbone.avgpool(features)
        return torch.flatten(features, 1)

    def _step(self, batch, stage):

        image = batch["image"]
        label = batch["label"]

        logits = self.backbone(image)
        loss = self.loss_fn(logits, label)

        pred_label = torch.argmax(logits, dim=1)
        accuracy = (pred_label == label).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", accuracy, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    def _build_backbone(self):

        backbone = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze all pretrained layers
        for param in backbone.parameters():
            param.requires_grad = False

        # Replace the classifier
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(EFFICIENTNET_OUT_DIM, self.num_classes),
        )

        # Train only the new classifier
        for param in backbone.classifier.parameters():
            param.requires_grad = True

        return backbone
