import torch


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, class_num, freeze_encoder=False):
        super(Model, self).__init__()
        self.linear0 = torch.nn.Linear(input_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_reconstruct = torch.nn.Linear(hidden_size, input_size)
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
