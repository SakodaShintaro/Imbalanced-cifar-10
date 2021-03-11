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
    def __init__(self, input_size, input_channel_num, hidden_size, class_num, freeze_encoder=False):
        super(CNNModel, self).__init__()
        down_channel_num = [128, 128]
        self.conv1 = Conv2DwithBatchNorm(in_channels=input_channel_num, out_channels=down_channel_num[0], kernel_size=3)
        self.conv2 = Conv2DwithBatchNorm(in_channels=down_channel_num[0], out_channels=down_channel_num[1], kernel_size=3)

        self.blocks = nn.Sequential()
        block_num = 4
        for i in range(block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(down_channel_num[-1], kernel_size=3, reduction=8))

        self.conv_classifier = nn.Conv2d(in_channels=down_channel_num[1], out_channels=down_channel_num[1], kernel_size=1, padding=0)
        representation_size = input_size // (len(down_channel_num) * 2)
        self.linear_classifier = torch.nn.Linear(down_channel_num[1] * representation_size * representation_size, class_num)
        self.freeze_encoder = freeze_encoder
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Residual Block
        x = self.blocks(x)

        # classifier
        x = self.conv_classifier(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_classifier(x)

        return x
