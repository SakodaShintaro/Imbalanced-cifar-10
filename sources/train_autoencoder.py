import torch
import torchvision
import argparse
from model import AutoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    root_dir = "../data"

    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                            download=True, transform=transform)
    print(trainset)
    exit(1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    image_size = 32
    image_channel = 3

    m = AutoEncoder(image_size * image_size * image_channel, args.hidden_size)
    print(m)

    for minibatch in trainloader:
        x, y = minibatch
        x = x.flatten(1)
        print(x.shape, y.shape)
        out = m.forward(x)
        print(out)


if __name__ == "__main__":
    main()
