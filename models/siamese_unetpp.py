import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseUNetPP(nn.Module):
    def __init__(self, encoder_name='efficientnet-b3', encoder_weights='imagenet'):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )

        # encoder_channels = self.encoder.out_channels
        encoder_channels = [
            c * 3 for c in self.encoder.out_channels
        ]

        self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )

        self.segmentation_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, pre, post):
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)

        diff_features = []

        for pre_feat, post_feat in zip(pre_features, post_features):
            # diff = torch.abs(pre_feat - post_feat)
            # diff_features.append(diff)
            # fusion = torch.cat(
            #     [pre_feat, post_feat],
            #     dim=1
            # )

            fusion = torch.cat(
                [
                    pre_feat,
                    post_feat,
                    torch.abs(pre_feat - post_feat)
                ],
                dim=1
            )

            diff_features.append(fusion)

        decoder_output = self.decoder(*diff_features)

        mask = self.segmentation_head(decoder_output)

        return mask