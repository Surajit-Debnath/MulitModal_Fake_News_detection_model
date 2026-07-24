import torch

from app.deploy_model import DeploymentModel


class ModelLoader:
    def __init__(self, checkpoint_path, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = DeploymentModel()

        self._load_weights(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        lightning_state = checkpoint["state_dict"]

        model_state = {}

        for key, value in lightning_state.items():
            if key.startswith("model."):
                key = key.replace("model.", "", 1)

            model_state[key] = value

        missing, unexpected = self.model.load_state_dict(
            model_state,
            strict=False,
        )

        print("Model loaded successfully!")

        if missing:
            print("\nMissing keys:")
            print(missing)

        if unexpected:
            print("\nUnexpected keys:")
            print(unexpected)

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device