import segmentation_models_pytorch as smp
import torch.nn as nn


class ChangeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1
        )

    def forward(self, x):
        return self.model(x)