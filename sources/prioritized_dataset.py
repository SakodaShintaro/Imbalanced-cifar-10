import torch
import torchvision
import numpy as np


class SegmentTree:
    def __init__(self, size):
        self.n_ = 2**(size.bit_length())
        self.sum_ = [0 for _ in range(self.n_ * 2)]

    def update(self, x, v):
        self.sum_[x + self.n_ - 1] = v
        i = (x + self.n_ - 2) // 2
        while True:
            self.sum_[i] = self.sum_[2 * i + 1] + self.sum_[2 * i + 2]
            if i == 0:
                break
            i = (i - 1) // 2

    def getIndex(self, value, k=0):
        if k >= self.n_ - 1:
            return k - (self.n_ - 1)
        return (self.getIndex(value, 2 * k + 1) if value <= self.sum_[2 * k + 1] else self.getIndex(value - self.sum_[2 * k + 1], 2 * k + 2))

    def getSum(self):
        return self.sum_[0]


class PrioritizedDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.data = subset.dataset.data
        self.targets = subset.dataset.targets
        self.targets = np.array(self.targets)
        self.transform = subset.dataset.transform
        self.segtree = SegmentTree(len(self.data))
        for i in range(len(self.data)):
            self.segtree.update(i, 10)

        self.trained_indices = list()

    def __getitem__(self, index):
        # Ignore the argument index and determine the index from the segment tree
        rnd = np.random.rand() * self.segtree.getSum()
        # print(f"sum = {self.segtree.getSum()}, rnd = {rnd}")
        index = self.segtree.getIndex(rnd)

        # Save this index to update loss information
        self.trained_indices.append(index)

        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.data)

    def update(self, losses):
        for index, loss in zip(self.trained_indices, losses):
            self.segtree.update(index, loss.item())
        self.trained_indices.clear()


class PrioritizedDataloader:
    def __init__(self, prioritized_dataset, batch_size):
        self.dataset_ = prioritized_dataset
        self.batch_size = batch_size
        self.i_ = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i_ >= len(self.dataset_.data):
            self.i_ = 0
            raise StopIteration()

        data = list()
        label = list()
        for i in range(self.batch_size):
            x, y = self.dataset_.__getitem__(0)
            data.append(x)
            y = torch.tensor(y)

            label.append(y)
            self.i_ += 1

        data = torch.stack(data)
        label = torch.stack(label)
        return data, label
