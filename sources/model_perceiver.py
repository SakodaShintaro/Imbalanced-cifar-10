#!/usr/bin/env python3
import torch
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder, PerceiverImagePreprocessor


class PostProcessor(torch.nn.Module):
    def __init__(self, image_channels, image_height, image_width):
        super().__init__()
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x, *args, **keywords):
        x = x.squeeze(1)
        return x


class PerceiverImageReconstructModel(PerceiverModel):
    def __init__(self, image_channels, image_height, image_width):
        config = PerceiverConfig(d_model=256, d_latents=160, num_self_attends_per_block=4)
        super().__init__(config)
        self.config = config
        self.input_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="conv",
            spatial_downsample=1,
            out_channels=126,
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs=dict(
                num_bands=32,
                max_resolution=(image_height, image_width),
                sine_only=False,
                concat_pos=True,
            ),
        )
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=10,
            num_channels=config.d_latents,
            use_query_residual=True,
            trainable_position_encoding_kwargs=dict(
                index_dims=1, num_channels=config.d_latents
            ))
        self.output_postprocessor = PostProcessor(image_channels, image_height, image_width)


class PerceiverRapperModel(nn.Module):
    def __init__(self, image_channels, image_height, image_width):
        super(PerceiverRapperModel, self).__init__()
        self.model = PerceiverImageReconstructModel(image_channels, image_height, image_width)

    def forward(self, x):
        out = self.model(x)
        return out.logits


if __name__ == "__main__":
    BATCH_SIZE = 2
    IMAGE_CHANNEL = 3
    IMAGE_HEIGHT = IMAGE_WIDTH = 32
    model = PerceiverRapperModel(IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH).cuda()
    x = torch.ones([BATCH_SIZE, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH]).cuda()
    out = model(x)
    print(out.shape)
