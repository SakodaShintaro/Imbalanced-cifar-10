import torch
import torchvision
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, data_num_of_imbalanced_class, copy_imbalanced_class):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        self.data = self.cifar10.data
        self.targets = self.cifar10.targets
        self.targets = np.array(self.targets)

        self.transform = transform

        # erase data
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        class_to_label = {name: i for i, name in enumerate(class_names)}

        imbalanced_class_labels = [class_to_label[name] for name in ["bird", "deer", "truck"]]

        mask = np.ones(len(self.data), dtype=bool)
        data_num = [0 for _ in range(len(class_names))]
        for i, (data, target) in enumerate(zip(self.data, self.targets)):
            if target in imbalanced_class_labels and data_num[target] >= data_num_of_imbalanced_class:
                mask[i] = False
                continue

            data_num[target] += 1

        self.data = self.data[mask]
        self.targets = self.targets[mask]

        # copy imbalanced class
        if copy_imbalanced_class:
            imbalanced_class_indices = [(target in imbalanced_class_labels) for target in self.targets]
            append_data = self.data[imbalanced_class_indices]
            append_targets = self.targets[imbalanced_class_indices]
            copy_num = (5000 // data_num_of_imbalanced_class) - 1
            print(append_data.shape)
            print(append_targets.shape)

            append_data = np.tile(append_data, reps=(copy_num, 1, 1, 1))
            append_targets = np.tile(append_targets, reps=(copy_num, ))
            print(append_data.shape)
            print(append_targets.shape)

            self.data = np.concatenate([self.data, append_data], 0)
            self.targets = np.concatenate([self.targets, append_targets], 0)
            print(self.data.shape)
            print(self.targets.shape)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.data)
