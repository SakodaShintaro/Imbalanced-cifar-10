import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.linear0 = torch.nn.Linear(input_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.linear0(x)
        x = torch.nn.functional.relu(x)
        x = self.linear1(x)
        x = torch.sigmoid(x)
        return x
