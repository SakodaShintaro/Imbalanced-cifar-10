import torch
import torch.nn as nn


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, input_channel_num, hidden_size, class_num, freeze_encoder=False):
        super(LinearModel, self).__init__()
        self.linear0 = torch.nn.Linear(input_size * input_size * input_channel_num, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_reconstruct = torch.nn.Linear(hidden_size, input_size * input_size * input_channel_num)
        self.linear_classifier = torch.nn.Linear(hidden_size, class_num)
        self.freeze_encoder = freeze_encoder

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(1)
        x = self.linear0(x)
        x = torch.nn.functional.relu(x)

        x = self.linear1(x)
        x = torch.nn.functional.relu(x)

        if self.freeze_encoder:
            x = x.detach()

        # reconstruct
        r = self.linear_reconstruct(x)
        r = torch.tanh(r)
        r = r.view(x_shape)

        # classifier
        c = self.linear_classifier(x)

        return r, c


class CNNModel(nn.Module):
    def __init__(self, input_size, input_channel_num, hidden_size, class_num, freeze_encoder=False):
        super(CNNModel, self).__init__()
        down_channel_num = [32, 16]
        self.conv1 = nn.Conv2d(in_channels=input_channel_num, out_channels=down_channel_num[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=down_channel_num[0], out_channels=down_channel_num[1], kernel_size=3, padding=1)

        self.t_conv1 = nn.ConvTranspose2d(in_channels=down_channel_num[1], out_channels=down_channel_num[0], kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=down_channel_num[0], out_channels=input_channel_num, kernel_size=2, stride=2)

        self.conv_classifier = nn.Conv2d(in_channels=down_channel_num[1], out_channels=down_channel_num[1], kernel_size=1, padding=0)
        representation_size = input_size // (len(down_channel_num) * 2)
        self.linear_classifier = torch.nn.Linear(down_channel_num[1] * representation_size * representation_size, class_num)
        self.freeze_encoder = freeze_encoder
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)

        # representation
        if self.freeze_encoder:
            x = x.detach()

        # reconstruct
        r = self.t_conv1(x)
        r = torch.nn.functional.relu(r)
        r = self.t_conv2(r)
        r = torch.tanh(r)

        # classifier
        c = self.conv_classifier(x)
        c = torch.nn.functional.relu(c)
        c = torch.flatten(c, 1)
        c = self.linear_classifier(c)

        return r, c
