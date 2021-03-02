import torch
import torchvision
import numpy as np
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, data_num_of_imbalanced_class):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=True,
                                                    download=True, transform=transform)
        self.data = self.cifar10.data
        self.targets = self.cifar10.targets
        self.targets = np.array(self.targets)

        self.transform = transform

        # 削除
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
        ]
        class_to_label = dict()
        for i, name in enumerate(class_names):
            class_to_label[name] = i

        imbalanced_class_labels = [class_to_label[name]
                                   for name in ["bird", "deer", "truck"]]

        mask = np.ones(len(self.data), dtype=bool)
        data_num = [0 for _ in range(len(class_names))]
        for i, (data, target) in enumerate(zip(self.data, self.targets)):
            if target in imbalanced_class_labels and data_num[target] >= data_num_of_imbalanced_class:
                mask[i] = False
                continue

            data_num[target] += 1

        self.data = self.data[mask]
        self.targets = self.targets[mask]

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.data)
