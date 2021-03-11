import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DwithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depthwise=False):
        super(Conv2DwithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=kernel_size // 2, groups=in_channels if depthwise else 1)
        self.norm_ = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduction=8):
        super(SqueezeExcite, self).__init__()
        reduced_chs = in_chs // reduction
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, reduction):
        super(ResidualBlock, self).__init__()
        self.conv_and_norm0_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.conv_and_norm1_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.se_module = SqueezeExcite(channel_num, reduction)

    def forward(self, x):
        t = x
        t = self.conv_and_norm0_.forward(t)
        t = F.relu(t)
        t = self.conv_and_norm1_.forward(t)
        t = self.se_module(t)
        t = F.relu(x + t)
        return t


class InvertedResidual(nn.Module):
    def __init__(self, channel_num, exp_ratio=1):
        super(InvertedResidual, self).__init__()
        mid_chs = channel_num * exp_ratio

        self.act = nn.ReLU(inplace=True)

        # Point-wise expansion
        self.conv_bn1 = Conv2DwithBatchNorm(channel_num, mid_chs, kernel_size=1)

        # Depth-wise convolution
        self.conv_bn2 = Conv2DwithBatchNorm(mid_chs, mid_chs, kernel_size=3, depthwise=True)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_chs)

        # Point-wise linear projection
        self.conv_bn3 = Conv2DwithBatchNorm(mid_chs, channel_num, kernel_size=1)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_bn1(x)
        x = self.act(x)

        # Depth-wise convolution
        x = self.conv_bn2(x)
        x = self.act(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_bn3(x)

        x += residual

        return x


class CNNModel(nn.Module):
    def __init__(self, input_size, input_channel_num, down_sampling_num=2, block_num=4, channel_num=128, class_num=10):
        super(CNNModel, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(down_sampling_num):
            self.encoder.add_module(f"conv{i}", Conv2DwithBatchNorm(input_channel_num if i == 0 else channel_num, channel_num, kernel_size=3))
            self.encoder.add_module(f"relu{i}", nn.ReLU())
            self.encoder.add_module(f"pool{i}", nn.MaxPool2d(2, 2))

        for i in range(block_num):
            self.encoder.add_module(f"block{i}", ResidualBlock(channel_num, kernel_size=3, reduction=8))
            # self.encoder.add_module(f"block{i}", InvertedResidual(channel_num, 1))

        representation_size = input_size // (down_sampling_num * 2)
        self.classifier = nn.Sequential()
        self.classifier.add_module("classifier_conv", nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=1, padding=0))
        self.classifier.add_module("classifier_relu", nn.ReLU())
        self.classifier.add_module("classifier_flatten", nn.Flatten(1))
        self.classifier.add_module("classifier_linear", torch.nn.Linear(channel_num * representation_size * representation_size, class_num))

    def encode(self, x):
        return self.encoder(x)

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)
