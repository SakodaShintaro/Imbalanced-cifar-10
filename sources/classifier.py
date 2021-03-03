import torch


class Classifier(torch.nn.Module):
    def __init__(self, auto_encoder, hidden_size, class_num):
        super(Classifier, self).__init__()
        self.auto_encoder = auto_encoder
        self.linear = torch.nn.Linear(hidden_size, class_num)

    def forward(self, x):
        x = self.auto_encoder.get_embed(x)
        x = self.linear(x)
        return x
