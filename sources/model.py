import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DwithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2DwithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, reduction):
        super(ResidualBlock, self).__init__()
        self.conv_and_norm0_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.conv_and_norm1_ = Conv2DwithBatchNorm(channel_num, channel_num, kernel_size)
        self.linear0_ = nn.Linear(channel_num, channel_num // reduction, bias=False)
        self.linear1_ = nn.Linear(channel_num // reduction, channel_num, bias=False)

    def forward(self, x):
        t = x
        t = self.conv_and_norm0_.forward(t)
        t = F.relu(t)
        t = self.conv_and_norm1_.forward(t)

        y = F.avg_pool2d(t, [t.shape[2], t.shape[3]])
        y = y.view([-1, t.shape[1]])
        y = self.linear0_.forward(y)
        y = F.relu(y)
        y = self.linear1_.forward(y)
        y = torch.sigmoid(y)
        y = y.view([-1, t.shape[1], 1, 1])
        t = t * y

        t = F.relu(x + t)
        return t


class CNNModel(nn.Module):
    def __init__(self, input_size, input_channel_num, down_sampling_num=2, block_num=4, channel_num=128, class_num=10):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential()
        for i in range(down_sampling_num):
            self.model.add_module(f"conv{i}", Conv2DwithBatchNorm(input_channel_num if i == 0 else channel_num, channel_num, kernel_size=3))
            self.model.add_module(f"relu{i}", nn.ReLU())
            self.model.add_module(f"pool{i}", nn.MaxPool2d(2, 2))

        for i in range(block_num):
            self.model.add_module(f"block{i}", ResidualBlock(channel_num, kernel_size=3, reduction=8))

        representation_size = input_size // (down_sampling_num * 2)

        self.model.add_module("classifier_conv", nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=1, padding=0))
        self.model.add_module("classifier_relu", nn.ReLU())
        self.model.add_module("classifier_flatten", nn.Flatten(1))
        self.model.add_module("classifier_linear", torch.nn.Linear(channel_num * representation_size * representation_size, class_num))

    def forward(self, x):
        return self.model(x)
