import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([4.0]).cuda()
        )

        self.dice = smp.losses.DiceLoss(
            mode='binary',
            smooth=1.0
        )

    def forward(self, preds, targets):

        bce_loss = self.bce(preds, targets)

        dice_loss = self.dice(preds, targets)

        total_loss = (
            0.7 * bce_loss +
            0.3 * dice_loss
        )

        return total_loss