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
        config = PerceiverConfig(d_model=64, d_latents=80, num_self_attends_per_block=2)
        super().__init__(config)
        self.config = config
        self.input_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="conv1x1",
            spatial_downsample=1,
            in_channels=image_channels,
            out_channels=config.d_model,
            position_encoding_type="trainable",
            concat_or_add_pos="add",
            project_pos_dim=config.d_model,
            trainable_position_encoding_kwargs=dict(
                index_dims=image_height * image_width, num_channels=config.d_model
            )
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
