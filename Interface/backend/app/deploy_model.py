import torch
import torch.nn as nn
import torchvision

NUM_CLASSES = 2
DROPOUT_P = 0.1
RESNET_OUT_DIM = 2048


class DeploymentModel(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        text_feature_dim=300,
        image_feature_dim=300,
        fusion_output_size=512,
        hidden_size=512,
        num_classes=NUM_CLASSES,
        dropout_p=DROPOUT_P,
    ):
        super().__init__()

        # ---------- Text branch ----------
        self.text_module = nn.Linear(
            embedding_dim,
            text_feature_dim,
        )

        # ---------- Image branch ----------
        self.image_module = torchvision.models.resnet152(
            weights=torchvision.models.ResNet152_Weights.DEFAULT
        )

        # Freeze pretrained backbone
        for param in self.image_module.parameters():
            param.requires_grad = False

        # Replace classifier
        self.image_module.fc = nn.Linear(
            RESNET_OUT_DIM,
            image_feature_dim,
        )

        # ---------- Fusion ----------
        self.fusion = nn.Linear(
            text_feature_dim + image_feature_dim,
            fusion_output_size,
        )

        self.fc1 = nn.Linear(
            fusion_output_size,
            hidden_size,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            num_classes,
        )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):

        text_features = torch.relu(
            self.text_module(text)
        )

        image_features = torch.relu(
            self.image_module(image)
        )

        combined = torch.cat(
            (text_features, image_features),
            dim=1,
        )

        fused = self.dropout(
            torch.relu(self.fusion(combined))
        )

        hidden = torch.relu(
            self.fc1(fused)
        )

        logits = self.fc2(hidden)

        return logits